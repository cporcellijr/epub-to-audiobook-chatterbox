"""Pure-Python tests for chatterbox/fast_t3.py (F-45): the setting, the install gate, the
per-request fallback decision, the fallback on a raised exception, restoring the stock loop,
and the dedicated synthesis thread. No torch, GPU or model needed: torch is imported lazily
by the module, and the compiled decoder itself is exercised by
docs/chatterbox-edition/experiments/f45/validate_build.py on the GPU.

Run inside the image or anywhere with Python 3.10+:
    python3 -m unittest discover -s tests
"""
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fast_t3


class TestResolveCompileSetting(unittest.TestCase):
    def test_on_values(self) -> None:
        for value in ("on", "1", "true", " ON ", "True"):
            self.assertTrue(fast_t3.resolve_compile_setting(value), value)

    def test_off_values_and_default(self) -> None:
        for value in ("off", "0", "false", "", "auto", "yes"):
            self.assertFalse(fast_t3.resolve_compile_setting(value), value)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(fast_t3.resolve_compile_setting())
        with mock.patch.dict(os.environ, {"TTS_COMPILE": "on"}):
            self.assertTrue(fast_t3.resolve_compile_setting())


class TestInstallReason(unittest.TestCase):
    def test_installs_only_for_original_bf16_cuda_with_setting_on(self) -> None:
        self.assertIsNone(fast_t3.install_reason(True, "original", True, "cuda"))
        self.assertIsNone(fast_t3.install_reason(True, "original", True, "cuda:0"))

    def test_reasons(self) -> None:
        self.assertIn("TTS_COMPILE", fast_t3.install_reason(False, "original", True, "cuda"))
        self.assertIn("turbo", fast_t3.install_reason(True, "turbo", True, "cuda"))
        self.assertIn("multilingual", fast_t3.install_reason(True, "multilingual", True, "cuda"))
        self.assertIn("TTS_BF16", fast_t3.install_reason(True, "original", False, "cuda"))
        self.assertIn("CUDA", fast_t3.install_reason(True, "original", True, "cpu"))
        self.assertIn("CUDA", fast_t3.install_reason(True, "original", True, None))


class TestFallbackReason(unittest.TestCase):
    def test_cfg_pair_on_cuda_takes_the_fast_path(self) -> None:
        self.assertIsNone(fast_t3.fallback_reason(2, "cuda"))

    def test_batch_size_other_than_the_cfg_pair(self) -> None:
        self.assertIn("batch size 1", fast_t3.fallback_reason(1, "cuda"))  # cfg_weight == 0
        self.assertIn("batch size 4", fast_t3.fallback_reason(4, "cuda"))

    def test_not_cuda(self) -> None:
        self.assertIn("cpu", fast_t3.fallback_reason(2, "cpu"))

    def test_unsupported_generation_arguments(self) -> None:
        self.assertIn("num_return_sequences", fast_t3.fallback_reason(2, "cuda", num_return_sequences=2))
        self.assertIsNotNone(fast_t3.fallback_reason(2, "cuda", stop_on_eos=False))
        self.assertIsNotNone(fast_t3.fallback_reason(2, "cuda", do_sample=False))
        self.assertIsNotNone(fast_t3.fallback_reason(2, "cuda", initial_speech_tokens=object()))
        self.assertIsNotNone(fast_t3.fallback_reason(2, "cuda", prepend_prompt_speech_tokens=object()))

    def test_alignment_analyzer(self) -> None:
        self.assertIn("analyzer", fast_t3.fallback_reason(2, "cuda", needs_analyzer=True))

    def test_cache_fit(self) -> None:
        self.assertIsNone(fast_t3.cache_fit_reason(34 + 120 + 1, 1000))
        self.assertIsNone(fast_t3.cache_fit_reason(1048, 1000))
        self.assertIn("exceeds", fast_t3.cache_fit_reason(1049, 1000))
        self.assertIn("exceeds", fast_t3.cache_fit_reason(10, 5, cache_len=14))


def _fake_tokens(batch: int = 2, device: str = "cuda") -> SimpleNamespace:
    return SimpleNamespace(shape=(batch, 12), device=SimpleNamespace(type=device))


def _bare_compiled_t3() -> fast_t3.CompiledT3:
    """A CompiledT3 without torch: only the dispatch state, as the tests need it."""
    fast = fast_t3.CompiledT3.__new__(fast_t3.CompiledT3)
    fast.t3 = SimpleNamespace(hp=SimpleNamespace(is_multilingual=False))
    fast.closed = False
    fast._warned = False
    fast.fast_calls = 0
    fast.fallback_calls = 0
    fast._stock_inference = mock.Mock(return_value="stock tokens")
    fast._fast_inference = mock.Mock(return_value="fast tokens")
    return fast


class TestCompiledT3Dispatch(unittest.TestCase):
    def test_cfg_pair_runs_the_fast_path(self) -> None:
        fast = _bare_compiled_t3()
        result = fast.inference(t3_cond="cond", text_tokens=_fake_tokens(), max_new_tokens=1000,
                                temperature=0.61, cfg_weight=0.5, repetition_penalty=1.2, min_p=0.05, top_p=1.0)
        self.assertEqual(result, "fast tokens")
        self.assertEqual((fast.fast_calls, fast.fallback_calls), (1, 0))
        fast._stock_inference.assert_not_called()
        kwargs = fast._fast_inference.call_args.kwargs
        self.assertEqual((kwargs["temperature"], kwargs["cfg_weight"], kwargs["repetition_penalty"],
                          kwargs["min_p"], kwargs["top_p"], kwargs["max_new_tokens"]),
                         (0.61, 0.5, 1.2, 0.05, 1.0, 1000))

    def test_batch_of_one_goes_to_the_stock_loop_with_all_arguments(self) -> None:
        fast = _bare_compiled_t3()
        result = fast.inference(t3_cond="cond", text_tokens=_fake_tokens(batch=1), max_new_tokens=1000,
                                temperature=0.61, cfg_weight=0.0)
        self.assertEqual(result, "stock tokens")
        fast._fast_inference.assert_not_called()
        kwargs = fast._stock_inference.call_args.kwargs
        self.assertEqual(kwargs["t3_cond"], "cond")
        self.assertEqual(kwargs["cfg_weight"], 0.0)
        self.assertEqual(kwargs["max_new_tokens"], 1000)
        self.assertEqual(kwargs["temperature"], 0.61)
        self.assertIn("min_p", kwargs)
        self.assertEqual((fast.fast_calls, fast.fallback_calls), (0, 1))

    def test_cpu_tokens_go_to_the_stock_loop(self) -> None:
        fast = _bare_compiled_t3()
        self.assertEqual(fast.inference(t3_cond="c", text_tokens=_fake_tokens(device="cpu")), "stock tokens")
        fast._fast_inference.assert_not_called()

    def test_multilingual_model_goes_to_the_stock_loop(self) -> None:
        fast = _bare_compiled_t3()
        fast.t3.hp.is_multilingual = True
        self.assertEqual(fast.inference(t3_cond="c", text_tokens=_fake_tokens()), "stock tokens")
        fast._fast_inference.assert_not_called()

    def test_fallback_needed_is_quiet_and_uses_the_stock_loop(self) -> None:
        fast = _bare_compiled_t3()
        fast._fast_inference = mock.Mock(side_effect=fast_t3.FallbackNeeded("prompt too long"))
        with self.assertLogs(fast_t3.logger, level="DEBUG") as logs:
            self.assertEqual(fast.inference(t3_cond="c", text_tokens=_fake_tokens()), "stock tokens")
        self.assertFalse(any(record.levelname == "WARNING" for record in logs.records))
        self.assertTrue(any("prompt too long" in record.getMessage() for record in logs.records))
        self.assertEqual((fast.fast_calls, fast.fallback_calls), (0, 1))

    def test_raised_exception_falls_back_and_warns_once(self) -> None:
        fast = _bare_compiled_t3()
        fast._fast_inference = mock.Mock(side_effect=RuntimeError("CUDA graph replay failed"))
        with self.assertLogs(fast_t3.logger, level="DEBUG") as logs:
            for _ in range(3):
                self.assertEqual(fast.inference(t3_cond="c", text_tokens=_fake_tokens()), "stock tokens")
        warnings = [record for record in logs.records if record.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIsNotNone(warnings[0].exc_info)
        self.assertEqual(fast._stock_inference.call_count, 3)
        self.assertEqual((fast.fast_calls, fast.fallback_calls), (0, 3))

    def test_closed_decoder_uses_the_stock_loop(self) -> None:
        fast = _bare_compiled_t3()
        fast.closed = True
        self.assertEqual(fast.inference(t3_cond="c", text_tokens=_fake_tokens()), "stock tokens")
        fast._fast_inference.assert_not_called()


class TestUninstall(unittest.TestCase):
    def test_restores_the_class_method_and_closes(self) -> None:
        class FakeT3:
            def inference(self, **kwargs):
                return "stock"

        model = SimpleNamespace(t3=FakeT3())
        model.t3.inference = lambda **kwargs: "fast"  # what install() does
        self.assertEqual(model.t3.inference(), "fast")
        fast = mock.Mock()
        fast_t3._uninstall_on_worker(model, fast)
        self.assertEqual(model.t3.inference(), "stock")
        self.assertNotIn("inference", vars(model.t3))
        fast.close.assert_called_once_with()

    def test_uninstall_without_a_patch_or_decoder_is_a_no_op(self) -> None:
        class FakeT3:
            def inference(self, **kwargs):
                return "stock"

        model = SimpleNamespace(t3=FakeT3())
        fast_t3._uninstall_on_worker(model, None)
        self.assertEqual(model.t3.inference(), "stock")

    def test_close_drops_compiled_state(self) -> None:
        fast = _bare_compiled_t3()
        fast.compiled_step = fast.step = fast.cache = fast.step_embeds = fast.step_position = object()
        fast.close()
        self.assertTrue(fast.closed)
        self.assertIsNone(fast.compiled_step)
        self.assertIsNone(fast.cache)


class TestSynthesisWorker(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = fast_t3.SynthesisWorker(name="test-worker")
        self.addCleanup(self.worker.close)

    def test_every_call_runs_on_the_same_thread(self) -> None:
        idents = {self.worker.run(threading.get_ident) for _ in range(5)}
        self.assertEqual(len(idents), 1)
        self.assertNotIn(threading.get_ident(), idents)
        self.assertEqual(idents, {self.worker.thread_ident})

    def test_calls_from_many_threads_land_on_the_worker_one_at_a_time(self) -> None:
        active, peak, lock = [0], [0], threading.Lock()
        idents = []

        def job() -> None:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            idents.append(threading.get_ident())
            time.sleep(0.02)
            with lock:
                active[0] -= 1

        threads = [threading.Thread(target=self.worker.run, args=(job,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(peak[0], 1)
        self.assertEqual(set(idents), {self.worker.thread_ident})

    def test_results_and_exceptions_come_back_to_the_caller(self) -> None:
        self.assertEqual(self.worker.run(lambda a, b=0: a + b, 2, b=3), 5)
        with self.assertRaises(ValueError):
            self.worker.run(lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_reentrant_call_from_the_worker_runs_inline(self) -> None:
        inner = self.worker.run(lambda: self.worker.run(threading.get_ident))
        self.assertEqual(inner, self.worker.thread_ident)

    def test_waiting_caller_holds_a_lock_across_the_worker_call(self) -> None:
        """What engine.synthesize does: hold the synthesis lock while the worker generates,
        so a load/unload waiting on that lock cannot run mid-generation."""
        engine_lock = threading.RLock()
        order = []

        def generation() -> None:
            with engine_lock:
                self.worker.run(lambda: (time.sleep(0.05), order.append("generated")))

        def unload() -> None:
            time.sleep(0.01)
            with engine_lock:
                order.append("unloaded")

        threads = [threading.Thread(target=generation), threading.Thread(target=unload)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(order, ["generated", "unloaded"])


if __name__ == "__main__":
    unittest.main()
