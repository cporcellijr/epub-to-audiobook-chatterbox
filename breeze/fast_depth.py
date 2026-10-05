"""The depth decoder's 15 steps per audio frame as one CUDA graph replay (WORKLOG §53).

Every frame, Breeze's plain (no-CFG) generation runs a whole Hugging Face generate() for the depth
decoder: a 2-token prefill (the backbone's hidden state and codebook 0), then 14 one-token steps through
its 12 layers, sampling codebooks 1-15 with the reserved codec ids suppressed, temperature and top-k
from its generation config. generate()'s per-call setup, a fresh DynamicCache and a host sync per step
made it 75% of a batch's time on 2026-10-05, with the GPU 17-40% busy.

Here the same math runs on fixed buffers (a 16-slot KV cache, precomputed RoPE and causal masks) and is
captured once per bucket of rows; a frame copies its rows in, replays the graph and copies the codes
out. Sampling is the exponential race (argmax of p / Exp(1), an exact categorical draw), so nothing in
the graph needs the host. The directed (CFG) path has its own loop and is left alone.

mode="eager" runs the same math without a graph and samples exactly as generate() does, so the same
seed gives the same tokens; it is how the math was checked against the stock decoder.
"""
import torch

STEPS = 15  # codebooks 1..15; codebook 0 comes from the backbone
SLOTS = 16  # cache positions 0..15
BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)


class StaticKV:
    """Just enough of a transformers Cache for BreezeAttention: update() writes this step's keys and values
    at cache_position and returns every slot (the mask hides the ones not yet written)."""

    def __init__(self, layers, batch, kv_heads, head_dim, dtype, device):
        shape = (batch, kv_heads, SLOTS, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(layers)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        position = cache_kwargs["cache_position"]
        self.k[layer_idx].index_copy_(2, position, key_states)
        self.v[layer_idx].index_copy_(2, position, value_states)
        return self.k[layer_idx], self.v[layer_idx]


class FastDepth:
    def __init__(self, model, mode="graph", buckets=BUCKETS):
        self.mode = mode
        self.buckets = tuple(sorted(buckets))
        self.dd = model.depth_decoder
        self.m = self.dd.model
        cfg = self.m.config
        weight = next(self.dd.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.vocab = self.m.vocab_size
        self.hidden_size = getattr(cfg, "backbone_hidden_size", None) or cfg.hidden_size
        self.layers = list(self.m.layers[: cfg.num_hidden_layers])
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        gen = self.dd.generation_config
        self.temperature = float(gen.temperature or 1.0)
        self.top_k = int(gen.top_k or 0)
        self.suppress = torch.tensor(model._reserved_codec_token_ids(), device=self.device, dtype=torch.long)
        positions = torch.arange(SLOTS, device=self.device)
        probe = torch.zeros(1, 1, cfg.hidden_size, device=self.device, dtype=self.dtype)
        cos, sin = self.m.rotary_emb(probe, positions[None])  # [1, SLOTS, head_dim]
        # rows are query positions, columns cache slots: a query sees the slots up to its own position
        causal = torch.where(positions[None, :] <= positions[:, None], 0.0,
                             torch.finfo(self.dtype).min).to(self.dtype)
        # (mask, rope, cache positions) for the prefill at positions 0-1 and for each later position;
        # all fixed tensors, so a captured graph reads the same memory on every replay
        self.prefill = (causal[0:2][None, None], (cos[:, 0:2], sin[:, 0:2]), positions[0:2].clone())
        self.step = [(causal[p:p + 1][None, None], (cos[:, p:p + 1], sin[:, p:p + 1]), positions[p:p + 1].clone())
                     for p in range(SLOTS)]
        self.graphs = {}
        self.pool = None

    def _layers(self, x, where, cache):
        mask, rope, position = where
        for layer in self.layers:
            x = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True, cache_position=position,
                      position_embeddings=rope)
        return self.m.norm(x)

    def _logits(self, h, position):
        """The codebook head for a query at `position` (it predicts codebook `position`)."""
        weight = self.dd.codebooks_head.weight[position - 1]  # [hidden, vocab]
        return torch.nn.functional.linear(h, weight.T).float()

    def _sample(self, scores, exact):
        scores = scores.index_fill(-1, self.suppress, float("-inf"))
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if self.top_k > 0:
            kth = torch.topk(scores, self.top_k)[0][..., -1, None]
            scores = scores.masked_fill(scores < kth, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        if exact:  # generate()'s own draw: the same seed gives the same token
            return torch.multinomial(probs, num_samples=1)
        return (probs / torch.empty_like(probs).exponential_()).argmax(-1, keepdim=True)

    def _codes(self, first, hidden, cache, exact):
        """first [B, 1] codebook 0, hidden [B, H] the backbone's state -> [B, STEPS] codebooks 1..15."""
        projector = self.m.backbone_hidden_state_projector
        e0 = projector(hidden) if projector is not None else hidden
        e1 = self.m.embed_tokens(first[:, 0])  # codebook 0 needs no offset
        x = self.m.inputs_embeds_projector(torch.stack([e0.to(self.dtype), e1], dim=1))
        h = self._layers(x, self.prefill, cache)
        token = self._sample(self._logits(h[:, -1], 1), exact)
        codes = [token]
        for p in range(2, STEPS + 1):  # codebook p-1 goes in at position p and codebook p comes out
            x = self.m.inputs_embeds_projector(self.m.embed_tokens(token + (p - 1) * self.vocab))
            h = self._layers(x, self.step[p], cache)
            token = self._sample(self._logits(h[:, -1], p), exact)
            codes.append(token)
        return torch.cat(codes, dim=-1)

    def _cache(self, rows):
        return StaticKV(len(self.layers), rows, self.kv_heads, self.head_dim, self.dtype, self.device)

    def capture(self, bucket):
        first = torch.zeros(bucket, 1, dtype=torch.long, device=self.device)
        hidden = torch.zeros(bucket, self.hidden_size, dtype=self.dtype, device=self.device)
        cache = self._cache(bucket)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up off the capture
                self._codes(first, hidden, cache, exact=False)
        torch.cuda.current_stream().wait_stream(side)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):  # graphs never run at once, so they share a pool
            codes = self._codes(first, hidden, cache, exact=False)
        # Everything a replay reads or writes has to stay alive with it, the cache included: a freed
        # cache's memory went to new tensors, and the next replay wrote keys over them.
        self.graphs[bucket] = (graph, first, hidden, codes, cache)

    @torch.inference_mode()
    def generate(self, input_ids=None, backbone_last_hidden_state=None, **_):
        """Stands in for depth_decoder.generate(input_ids=[n, 2], backbone_last_hidden_state=[n, H]) and
        returns what it does: [n, 2 + STEPS], the placeholder, codebook 0 and codebooks 1..15."""
        n = input_ids.shape[0]
        first = input_ids[:, 1:2]
        if self.mode == "eager":
            codes = self._codes(first, backbone_last_hidden_state, self._cache(n), exact=True)
        else:
            bucket = next((b for b in self.buckets if b >= n), None)
            if bucket is None:
                top = self.buckets[-1]
                return torch.cat([self.generate(input_ids[i:i + top], backbone_last_hidden_state[i:i + top])
                                  for i in range(0, n, top)])
            graph, first_in, hidden_in, codes_out, _ = self.graphs[bucket]
            first_in[:n].copy_(first)
            hidden_in[:n].copy_(backbone_last_hidden_state)
            graph.replay()
            codes = codes_out[:n].clone()
        return torch.cat([input_ids, codes], dim=-1)


def install(model, max_rows, mode="graph"):
    """Route the model's plain depth decoding through FastDepth, with graphs for up to max_rows rows
    captured now, largest first so the smaller ones fit in its pool."""
    buckets = [b for b in BUCKETS if b < max_rows] + [max_rows]
    fast = FastDepth(model, mode, buckets)
    if mode == "graph":
        with torch.inference_mode():
            for bucket in sorted(fast.buckets, reverse=True):
                fast.capture(bucket)
        torch.cuda.synchronize()
    model.depth_decoder.generate = fast.generate
    return fast
