"""VLLMASREngine 封装单元测试（mock Qwen3ASRModel，不依赖 vLLM / GPU）。

绕过 load()（不 import qwen_asr），直接注入 mock _model 验证三段式调用序、
chunk_size_sec 钳制与会话级覆盖。standard venv 即可运行。
"""
from types import SimpleNamespace

import numpy as np

from app.engines.vllm_asr_engine import (
    VLLMASREngine, clamp_chunk_size_sec, CHUNK_SIZE_SEC_MIN, CHUNK_SIZE_SEC_MAX,
)


def test_clamp_chunk_size_sec():
    assert clamp_chunk_size_sec(0.1) == CHUNK_SIZE_SEC_MIN
    assert clamp_chunk_size_sec(99) == CHUNK_SIZE_SEC_MAX
    assert clamp_chunk_size_sec(1.5) == 1.5


class _MockModel:
    def __init__(self):
        self.calls = []

    def init_streaming_state(self, language=None, chunk_size_sec=None,
                             unfixed_chunk_num=None, unfixed_token_num=None):
        self.calls.append(("init", language, chunk_size_sec, unfixed_chunk_num, unfixed_token_num))
        return SimpleNamespace(text="", language=language or "")

    def streaming_transcribe(self, pcm, state):
        self.calls.append(("feed", int(pcm.size)))
        state.text += "x"
        state.language = "Chinese"
        return state

    def finish_streaming_transcribe(self, state):
        self.calls.append(("finish",))
        state.text += "。"
        return state


def _engine_with_model(**kw):
    eng = VLLMASREngine(**kw)
    eng._model = _MockModel()
    return eng


def test_engine_clamps_init_chunk_size():
    eng = VLLMASREngine(chunk_size_sec=99)
    assert eng.chunk_size_sec == CHUNK_SIZE_SEC_MAX


def test_new_state_uses_engine_chunk_size():
    eng = _engine_with_model(chunk_size_sec=2.0)
    eng.new_state(language="Chinese")
    assert eng._model.calls[0] == ("init", "Chinese", 2.0, 2, 5)


def test_new_state_chunk_size_override_clamped():
    eng = _engine_with_model(chunk_size_sec=1.0)
    eng.new_state(chunk_size_sec=99)              # 越界 → 钳到上限
    assert eng._model.calls[0][2] == CHUNK_SIZE_SEC_MAX


def test_feed_and_finish_sequence():
    eng = _engine_with_model()
    st = eng.new_state()
    t1, _ = eng.feed(np.zeros(1600, np.float32), st)
    t2, _ = eng.feed(np.zeros(1600, np.float32), st)
    tf, lang = eng.finish(st)
    assert (t1, t2, tf) == ("x", "xx", "xx。")
    assert lang == "Chinese"
    assert [c[0] for c in eng._model.calls] == ["init", "feed", "feed", "finish"]


def test_is_loaded():
    eng = VLLMASREngine()
    assert eng.is_loaded is False
    eng._model = _MockModel()
    assert eng.is_loaded is True


def test_align_device_normalized():
    """对齐器设备规范化为小写；缺省/None → cuda（OOM 时可经 --vllm-align-device cpu 移出 GPU）。"""
    assert VLLMASREngine()._align_device == "cuda"
    assert VLLMASREngine(align_device="CPU")._align_device == "cpu"
    assert VLLMASREngine(align_device=None)._align_device == "cuda"


def test_infer_batch_size_default_bounded():
    """对齐/ASR 批大小默认有界（4，非 -1）：防长音频把全部 180s 块一次对齐致 OOM。"""
    assert VLLMASREngine()._infer_batch_size == 4
    assert VLLMASREngine(infer_batch_size=1)._infer_batch_size == 1


def test_load_forwards_bounded_runtime_options(monkeypatch):
    import sys
    import app.engines.vllm_asr_engine as module
    calls = []
    monkeypatch.setattr(module, "ensure_model", lambda *a: None)
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(
        Qwen3ASRModel=SimpleNamespace(LLM=lambda **kw: calls.append(kw) or _MockModel())))
    engine = VLLMASREngine(enable_align=False, max_num_seqs=1,
                          max_num_batched_tokens=1024, kv_cache_memory_bytes=268435456,
                          enforce_eager=True, skip_mm_profiling=True, max_new_tokens=256)
    engine.load()
    assert calls[0]["max_num_seqs"] == 1
    assert calls[0]["max_num_batched_tokens"] == 1024
    assert calls[0]["kv_cache_memory_bytes"] == 268435456
    assert calls[0]["enforce_eager"] is True
    assert calls[0]["skip_mm_profiling"] is True
    assert calls[0]["max_new_tokens"] == 256
    VLLMASREngine(enable_align=False).load()
    assert "kv_cache_memory_bytes" not in calls[1]


def test_invalid_runtime_limits_rejected_before_loading():
    import pytest
    for options in ({"max_num_seqs": True}, {"kv_cache_memory_bytes": 0},
                    {"max_num_batched_tokens": -1}, {"enforce_eager": 1},
                    {"skip_mm_profiling": "true"}, {"max_new_tokens": 0}):
        with pytest.raises(ValueError):
            VLLMASREngine(**options)


def test_warmup_uses_disposable_state_and_bounded_audio(tmp_path):
    import soundfile as sf
    sample = tmp_path / "warmup.wav"
    sf.write(sample, np.zeros(64000, np.float32), 16000)
    engine = _engine_with_model(chunk_size_sec=0.5)
    engine.warmup(sample)
    assert [c[0] for c in engine._model.calls] == ["init", "feed", "feed", "feed", "feed", "finish"]
    assert sum(c[1] for c in engine._model.calls if c[0] == "feed") == 32000
    assert engine.new_state().text == ""
    sf.write(sample, np.zeros(8000, np.float32), 8000)
    import pytest
    with pytest.raises(ValueError, match="16 kHz mono"):
        engine.warmup(sample)
