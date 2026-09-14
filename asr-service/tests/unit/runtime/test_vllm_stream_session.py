"""vLLM 流式会话/后端单元测试（不依赖 vLLM / GPU）。

EnergyEndpointer 能量端点事件、VllmStreamSession 信封序列（mock 引擎）、
VllmStreamBackend 准入/释放/能力。在 standard venv 即可运行（模块不 import vllm）。
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime.vllm_stream_session import (
    EnergyEndpointer, VllmStreamSession, VllmStreamBackend,
)

SR = 16000


def _pcm16_bytes(amp, ms):
    n = int(SR * ms / 1000)
    return (np.full(n, amp, dtype=np.float32) * 32767).astype("<i2").tobytes()


def _voice(ms=200):
    return _pcm16_bytes(0.2, ms)      # rms≈-14 dBFS ≥ -45 → 语音


def _silence(ms=200):
    return _pcm16_bytes(0.0, ms)      # -120 dBFS → 静音


async def _collect(agen):
    return [m async for m in agen]


# ─── EnergyEndpointer ───

def test_endpointer_start_then_end():
    ep = EnergyEndpointer(energy_floor_dbfs=-45.0, end_silence_ms=800)
    v = np.full(3200, 0.2, dtype=np.float32)      # 200ms 语音
    s = np.zeros(3200, dtype=np.float32)          # 200ms 静音

    assert ep.process(s, 200) == []               # 静音不起句
    ev = ep.process(v, 200)
    assert ev == [{"type": "start", "start": 200}]
    assert ep.in_speech is True
    assert ep.process(v, 200) == []               # 句内无事件
    # 尾静音累计：800ms（4×200）才判停
    assert ep.process(s, 200) == []
    assert ep.process(s, 200) == []
    assert ep.process(s, 200) == []
    ev = ep.process(s, 200)
    assert ev == [{"type": "end", "end": 1400}]
    assert ep.in_speech is False


def test_endpointer_reset():
    ep = EnergyEndpointer()
    ep.process(np.full(3200, 0.2, dtype=np.float32), 200)
    assert ep.in_speech is True
    ep.reset()
    assert ep.in_speech is False


# ─── mock 引擎 ───

class _MockEngine:
    """累积式 mock：每次 feed 追加一个字符，finish 补句号。"""

    def __init__(self):
        self.feeds = 0
        self.new_states = 0

    def new_state(self, language=None, chunk_size_sec=None):
        self.new_states += 1
        return SimpleNamespace(text="", language=language or "Chinese",
                               _acc="", chunk_size_sec=chunk_size_sec)

    def feed(self, arr, state):
        self.feeds += 1
        state._acc += "字"
        state.text = state._acc
        return state.text, state.language

    def finish(self, state):
        state.text = state._acc + "。"
        return state.text, state.language


def _make_session(engine=None, **bk):
    eng = engine or _MockEngine()
    backend = VllmStreamBackend(eng, **bk)
    return backend, backend.create_session("sid-test-0001")


# ─── VllmStreamSession.configure ───

def test_configure_warns_unsupported_params():
    _, sess = _make_session()
    warns = sess.configure({"audio_fs": 16000, "with_words": True, "diarize": True,
                            "with_punc": True, "speaker_threshold": 0.5})
    assert set(warns) == {"with_words", "diarize", "with_punc", "speaker_threshold"}


def test_configure_invalid_audio_fs_raises():
    _, sess = _make_session()
    with pytest.raises(ValueError):
        sess.configure({"audio_fs": 100})        # < 8000 下限


def test_configure_chunk_size_override_and_range():
    _, sess = _make_session()
    assert sess.configure({"chunk_size_sec": 1.5}) == []
    assert sess._chunk_size_sec == 1.5
    with pytest.raises(ValueError):
        sess.configure({"chunk_size_sec": 10})    # > 5.0 上限


@pytest.mark.parametrize("raw,expected", [
    ("zh", "Chinese"),        # ISO 码
    ("Zh", "Chinese"),        # 大小写不敏感
    ("zh-CN", "Chinese"),     # 带地区子标签
    ("Chinese", "Chinese"),   # 已是规范名
    ("xx", None),             # 未识别 → None 交自动检测
])
def test_configure_normalizes_language(raw, expected):
    """服务层归一化 language，避免非法 hint 击穿引擎抛 Unsupported language。"""
    _, sess = _make_session()
    sess.configure({"audio_fs": 16000, "language": raw})
    assert sess.language == expected


# ─── VllmStreamSession.feed_audio / flush ───

def test_feed_audio_partial_then_final():
    eng = _MockEngine()
    backend, sess = _make_session(eng, end_silence_ms=800)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(3):                        # 语音 → 起句 + partial
            msgs += await _collect(sess.feed_audio(_voice()))
        for _ in range(4):                        # 800ms 静音 → 判停 final
            msgs += await _collect(sess.feed_audio(_silence()))
        return msgs

    msgs = asyncio.run(run())
    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]

    assert len(partials) >= 3
    assert all(m["seg_id"] == 0 and m["text"] for m in partials)
    assert len(finals) == 1
    f = finals[0]
    assert f["seg_id"] == 0 and f["text"].endswith("。")
    assert f["start"] == 0 and f["end"] == 1400
    assert sess.state is None                     # 句尾已 reset
    assert eng.new_states == 1                    # 仅起了一句


def test_flush_emits_final_for_open_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(2):                        # 起句但不静音收尾
            msgs += await _collect(sess.feed_audio(_voice()))
        msgs += await _collect(sess.flush())      # stop → 冲刷末句
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"].endswith("。")
    assert sess.state is None


def test_feed_audio_silence_only_no_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(5):
            msgs += await _collect(sess.feed_audio(_silence()))
        return msgs

    msgs = asyncio.run(run())
    assert msgs == []                             # 全静音不起句、不解码
    assert eng.new_states == 0 and eng.feeds == 0


class _CaptureEngine(_MockEngine):
    def __init__(self):
        super().__init__()
        self.audio = []

    def new_state(self, language=None, chunk_size_sec=None):
        state = super().new_state(language, chunk_size_sec)
        state.audio = []
        self.audio.append(state.audio)
        return state

    def feed(self, arr, state):
        state.audio.append(arr.copy())
        return super().feed(arr, state)


def _decoded(pcm):
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


@pytest.mark.parametrize("quiet_ms", [(100, 100), (175, 175, 175), (1000,)])
def test_preroll_preserves_latest_quiet_samples_without_waiting(quiet_ms):
    eng = _CaptureEngine()
    backend, sess = _make_session(eng)
    quiet = [_pcm16_bytes(0.001 * (i + 1), ms) for i, ms in enumerate(quiet_ms)]
    voice = _voice(100)

    async def run():
        for pcm in quiet:
            assert await _collect(sess.feed_audio(pcm)) == []
        assert eng.new_states == 0               # 低音量缓冲本身不触发识别
        msgs = await _collect(sess.feed_audio(voice))
        assert msgs[0]["type"] == "partial"     # 当前帧立即触发，无需等后续帧
        return await _collect(sess.flush())

    try:
        finals = asyncio.run(run())
        expected = b"".join(quiet)[-SR * 300 // 1000 * 2:] + voice
        np.testing.assert_array_equal(np.concatenate(eng.audio[0]), _decoded(expected))
        assert finals[0]["start"] == max(0, sum(quiet_ms) - 300)
        assert finals[0]["end"] == sum(quiet_ms) + 100  # 回补不重复累计时间
    finally:
        backend.shutdown()


def test_preroll_does_not_replay_previous_segment_tail():
    eng = _CaptureEngine()
    backend, sess = _make_session(eng, end_silence_ms=200)
    quiet = _pcm16_bytes(0.002, 100)

    async def run():
        await _collect(sess.feed_audio(_voice(100)))
        first = await _collect(sess.feed_audio(_silence(200)))
        await _collect(sess.feed_audio(quiet))
        await _collect(sess.feed_audio(_voice(100)))
        return first + await _collect(sess.flush())

    try:
        msgs = asyncio.run(run())
        np.testing.assert_array_equal(np.concatenate(eng.audio[0]),
                                      _decoded(_voice(100) + _silence(200)))
        np.testing.assert_array_equal(np.concatenate(eng.audio[1]),
                                      _decoded(quiet + _voice(100)))
        finals = [m for m in msgs if m["type"] == "final"]
        assert [(m["seg_id"], m["start"], m["end"]) for m in finals] == [
            (0, 0, 300), (1, 300, 500)]
    finally:
        backend.shutdown()


@pytest.mark.parametrize("reset", ["configure", "flush"])
def test_preroll_is_cleared_at_task_boundary(reset):
    eng = _CaptureEngine()
    backend, sess = _make_session(eng)

    async def run():
        await _collect(sess.feed_audio(_pcm16_bytes(0.002, 300)))
        if reset == "configure":
            sess.configure({"audio_fs": SR})
        else:
            assert await _collect(sess.flush()) == []
        await _collect(sess.feed_audio(_voice(100)))
        return await _collect(sess.flush())

    try:
        finals = asyncio.run(run())
        np.testing.assert_array_equal(np.concatenate(eng.audio[0]), _decoded(_voice(100)))
        assert finals[0]["start"] == (0 if reset == "configure" else 300)
    finally:
        backend.shutdown()


# ─── VllmStreamBackend ───

def test_backend_capabilities():
    backend, _ = _make_session()
    assert backend.mode == "vllm" and backend.backend == "vllm-native"
    assert backend.capabilities["partial_results"] is True
    assert backend.capabilities["word_timestamps"] is False
    assert backend.capabilities["speaker_labels"] is False


def test_backend_acquire_release_limits():
    backend = VllmStreamBackend(_MockEngine(), max_sessions=2)

    async def run():
        a = await backend.acquire()
        b = await backend.acquire()
        c = await backend.acquire()               # 超额
        return a, b, c

    a, b, c = asyncio.run(run())
    assert (a, b, c) == (True, True, False)
    backend.release(backend.create_session("x"))  # 释放一个名额
    assert asyncio.run(backend.acquire()) is True
    backend.shutdown()
