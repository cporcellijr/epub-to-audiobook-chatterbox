"""Experiment B, WORKLOG §79. Read the cast LLM's own probability for
each speaker answer. Two decodings on the production prompts: "observe"
(production decoding plus token logprobs) and "decision" (each answer
constrained to the candidates the prompt itself shows). The prompt text is
never changed."""

import json
import math
import re
import functools
import os
import sys
import time
import runpy
from pathlib import Path

from audiobook_generator.core import gpu_reservation
from audiobook_generator.core.cast_llm import ChatClient, LLM_TEMPERATURE, PROMPTS

ROSTER_HEADERS = (PROMPTS["window"].split("\n")[0], PROMPTS["review"].split("\n")[0])
ROSTER_EMPTY = PROMPTS["roster_empty"]
NARRATOR_PREFIX = PROMPTS["narrator"].split("{name}")[0]
IDS_PREFIX = "Ids to answer: "
ALWAYS = ("I", "unknown")
TOP_LOGPROBS = 20

STATE = {"log": None, "chapter": None, "call": 0}

_LABEL_RE = re.compile(r"\[([^\[\]#][^\[\]]*)\] (?=[\"\u201c\u2018'\u00ab])")
_ENTRY_RE = re.compile(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def ids_from_prompt(text: str) -> list[int]:
    idx = text.rfind(IDS_PREFIX)
    if idx < 0:
        return []
    rest = text[idx + len(IDS_PREFIX):].split("\n", 1)[0]
    out: list[int] = []
    for piece in rest.split(","):
        piece = piece.strip()
        try:
            out.append(int(piece))
        except ValueError:
            return []
    return out


def split_roster(line: str) -> list[str]:
    entries: list[str] = []
    cur: list[str] = []
    depth = 0
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and line.startswith(", ", i):
            entries.append("".join(cur).strip())
            cur = []
            i += 2
            continue
        cur.append(ch)
        i += 1
    entries.append("".join(cur).strip())
    out: list[str] = []
    for e in entries:
        e = e.split(" = ")[0].strip()
        if e:
            out.append(e)
    return out


def candidates_from_prompt(text: str) -> list[str]:
    cands: list[str] = []

    def add(c: str) -> None:
        if c and c not in cands:
            cands.append(c)

    for header in ROSTER_HEADERS:
        marker = header + "\n"
        idx = text.find(marker)
        if idx < 0:
            continue
        roster = text[idx + len(marker):].split("\n", 1)[0]
        if roster == ROSTER_EMPTY:
            continue
        for e in split_roster(roster):
            add(e)
    nidx = text.find(NARRATOR_PREFIX)
    if nidx >= 0:
        name = text[nidx + len(NARRATOR_PREFIX):]
        for stop in (" (aliases:", ". Treat"):
            s = name.find(stop)
            if s >= 0:
                name = name[:s]
        add(name.strip())
    for m in _LABEL_RE.finditer(text):
        label = m.group(1)
        if label.endswith("?"):
            label = label[:-1]
        add(label.strip())
    for a in ALWAYS:
        add(a)
    return cands


def speaker_schema(ids: list[int], candidates: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "speakers": {
                "type": "object",
                "properties": {str(i): {"enum": list(candidates)} for i in ids},
                "required": [str(i) for i in ids],
                "additionalProperties": False,
            },
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "gender": {"enum": ["female", "male", "unknown"]},
                        "age": {"enum": ["child", "adult", "elderly", "unknown"]},
                        "aliases": {"type": "array", "items": {"type": "string"}}},
                    "required": ["name", "gender", "age", "aliases"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["speakers", "characters"],
        "additionalProperties": False,
    }


def answer_confidences(reply: str, tokens: list[dict], candidates: list[str]) -> dict[int, dict]:
    def tb(t: dict) -> bytes:
        b = t.get("bytes")
        return bytes(b) if b is not None else t["token"].encode("utf-8")

    R = reply.encode("utf-8")
    if b"".join(tb(t) for t in tokens) != R:
        return {}
    offsets: list[int] = []
    pos = 0
    for t in tokens:
        offsets.append(pos)
        pos += len(tb(t))
    s = reply.find('"speakers"')
    if s < 0:
        return {}
    ob = reply.find("{", s)
    if ob < 0:
        return {}
    cb = reply.find("}", ob)
    if cb < 0:
        return {}
    targets = [json.dumps(c, ensure_ascii=False)[1:-1].encode("utf-8") + b'"' for c in candidates]

    def consistent(full: bytes) -> bool:
        return any(t.startswith(full) or full.startswith(t) for t in targets)

    result: dict[int, dict] = {}
    for m in _ENTRY_RE.finditer(reply[ob:cb]):
        id_num = int(m.group(1))
        raw = m.group(2)
        vs = len(reply[:ob + m.start(2)].encode("utf-8"))
        ve = len(reply[:ob + m.end(2)].encode("utf-8"))
        val_tokens = [
            k for k in range(len(tokens))
            if offsets[k] < ve + 1 and offsets[k] + len(tb(tokens[k])) > vs
        ]
        if not val_tokens:
            continue
        answer = json.loads('"' + raw + '"')
        p_seq = math.exp(sum(tokens[k]["logprob"] for k in val_tokens))
        p_first = math.exp(tokens[val_tokens[0]]["logprob"])
        # shortcut: only the top 20 alternatives per token are seen, so the decision mass leaves out rarer candidates; ask the server for more if it allows.
        factors: list[float | None] = []
        first_pre = first_sofar = b""
        m_first = 0.0
        for k in val_tokens:
            t = tokens[k]
            o_k = offsets[k]
            B_k = tb(t)
            if o_k < vs:
                pre = R[o_k:vs]
                sofar = b""
            else:
                pre = b""
                sofar = R[vs:o_k]
            if k == val_tokens[0]:
                first_pre, first_sofar = pre, sofar
            mass = 0.0
            top_bytes: set[bytes] = set()
            for e in t.get("top_logprobs") or []:
                A = tb(e)
                top_bytes.add(A)
                if not A.startswith(pre):
                    continue
                if consistent(sofar + A[len(pre):]):
                    mass += math.exp(e["logprob"])
            chosen_ok = B_k.startswith(pre) and consistent(sofar + B_k[len(pre):])
            if chosen_ok and B_k not in top_bytes:
                mass += math.exp(t["logprob"])
            if k == val_tokens[0]:
                m_first = mass
            factors.append(math.exp(t["logprob"]) / mass if mass > 0 and chosen_ok else None)
        is_candidate = answer in candidates
        if is_candidate and all(f is not None for f in factors):
            p_decision = 1.0
            for f in factors:
                p_decision *= f
        else:
            p_decision = None
        runner_up: str | None = None
        p_runner_up: float | None = None
        t0 = tokens[val_tokens[0]]
        B0 = tb(t0)
        best: tuple[bytes, float, bytes] | None = None
        for e in t0.get("top_logprobs") or []:
            A = tb(e)
            if A == B0 or not A.startswith(first_pre):
                continue
            full = first_sofar + A[len(first_pre):]
            if not consistent(full):
                continue
            if best is None or e["logprob"] > best[1]:
                best = (A, e["logprob"], full)
        if best is not None and m_first > 0:
            for c in candidates:
                T = json.dumps(c, ensure_ascii=False)[1:-1].encode("utf-8") + b'"'
                if T.startswith(best[2]) or best[2].startswith(T):
                    runner_up = c
                    break
            p_runner_up = math.exp(best[1]) / m_first
        result[id_num] = {
            "answer": answer,
            "p_seq": p_seq,
            "p_first": p_first,
            "p_decision": p_decision,
            "runner_up": runner_up,
            "p_runner_up": p_runner_up,
            "is_candidate": is_candidate,
        }
    return result


class DecisionClient(ChatClient):
    """The production client plus token logprobs; in "decision" mode each speaker answer is
    constrained to the prompt's candidates. Every speaker request is logged to STATE["log"]."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = os.environ.get("CAST_DECODING", "observe")
        if self.mode not in ("observe", "decision"):
            raise ValueError(f"CAST_DECODING must be 'observe' or 'decision', got {self.mode!r}")

    @gpu_reservation.session("ollama")
    def __call__(self, messages):
        from openai import BadRequestError
        text = messages[-1]["content"]
        ids = ids_from_prompt(text)
        candidates = candidates_from_prompt(text) if ids else []
        constrained = self.mode == "decision" and bool(ids)
        kwargs = dict(model=self.model, messages=messages, temperature=LLM_TEMPERATURE,
                      logprobs=True, top_logprobs=TOP_LOGPROBS)
        if constrained:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "speakers",
                                                         "schema": speaker_schema(ids, candidates)}}
        elif self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if constrained:
                raise
            if not self.json_mode:
                raise
            self.json_mode = False
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "") if response.choices else ""
        tokens = ([t.model_dump() for t in response.choices[0].logprobs.content]
                  if response.choices and response.choices[0].logprobs
                  and response.choices[0].logprobs.content else [])
        answers = answer_confidences(content, tokens, candidates) if ids else {}
        STATE["call"] += 1
        if STATE["log"] is not None:
            with open(STATE["log"], "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "call": STATE["call"],
                    "chapter": STATE["chapter"],
                    "mode": self.mode,
                    "constrained": constrained,
                    "ids": ids,
                    "candidates": candidates,
                    "aligned": (True if not ids else bool(answers)),
                    "answers": {str(k): v for k, v in answers.items()},
                    "prompt_tokens": (getattr(response.usage, "prompt_tokens", None)
                                      if response.usage else None),
                    "seconds": round(time.monotonic() - started, 2),
                    **({"tokens": tokens} if ids else {}),
                }, ensure_ascii=False) + "\n")
                f.flush()
        return content


def install(mode: str) -> None:
    """Make the production analysis use DecisionClient, and log each book's speaker
    requests beside its request log."""
    if mode not in ("observe", "decision"):
        raise ValueError(f"mode must be 'observe' or 'decision', got {mode!r}")
    os.environ["CAST_DECODING"] = mode
    import audiobook_generator.core.cast_llm as cast_llm
    cast_llm.ChatClient = DecisionClient
    import audiobook_generator.core.cast_analysis as cast_analysis
    original = cast_analysis.attribute_chapter

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        label = kwargs.get("label", args[6] if len(args) > 6 else "")
        m = re.search(r"ch(\d+)", label)
        STATE["chapter"] = int(m.group(1)) if m else None
        return original(*args, **kwargs)

    cast_analysis.attribute_chapter = wrapper
    multivoice = Path(__file__).resolve().parents[1] / "multivoice"
    if str(multivoice) not in sys.path:
        sys.path.insert(0, str(multivoice))
    import evaluate_cast
    original_run = evaluate_cast.run

    def run(args, **kw):
        STATE["log"] = Path(args.request_log).with_name("decisions.jsonl")
        STATE["call"] = 0
        STATE["chapter"] = None
        return original_run(args, **kw)

    evaluate_cast.run = run


def main(argv=None) -> None:
    """decision_client.py observe|decision <runner.py> [runner args...]"""
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(__doc__)
    install(argv[0])
    runner = argv[1]
    rest = argv[2:]
    sys.argv = [runner, *rest]
    with gpu_reservation.session("ollama"):
        runpy.run_path(runner, run_name="__main__")


if __name__ == "__main__":
    main()
