"""Native websocket → real vLLM session → model API stub; no GPU required."""
import logging
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import ws_routes
from app.engines.vllm_asr_engine import VLLMASREngine
from app.runtime.vllm_stream_session import VllmStreamBackend


class Model:
    def __init__(self):
        self.contexts = []
    def init_streaming_state(self, **kwargs):
        self.contexts.append(kwargs["context"])
        return SimpleNamespace(text="", language="")
    def streaming_transcribe(self, pcm, state):
        state.text = "synthetic"
    def finish_streaming_transcribe(self, state):
        pass


def test_native_context_reaches_sdk_and_is_not_logged(caplog):
    engine = VLLMASREngine()
    engine._model = Model()
    backend = VllmStreamBackend(engine)
    ws_routes.init_ws_stream(backend)
    app = FastAPI()
    app.include_router(ws_routes.ws_router_stream)
    try:
        with caplog.at_level(logging.INFO), TestClient(app) as client:
            for context in ("SyntheticPrivateTerm", ""):
                with client.websocket_connect("/v2/asr/stream") as ws:
                    hello = ws.receive_json()
                    assert hello['capabilities']['hotword_context'] is True
                    ws.send_json(dict(type="start", context=context))
                    ws.send_bytes(np.full(3200, 8000, dtype="<i2").tobytes())
                    assert ws.receive_json()['type'] == 'partial'
                    ws.send_json(dict(type="stop"))
                    assert ws.receive_json()['type'] == 'final'
                    assert ws.receive_json()['type'] == 'session.closed'
        assert engine._model.contexts == ["SyntheticPrivateTerm", ""]
        assert "SyntheticPrivateTerm" not in caplog.text
    finally:
        backend.shutdown()
        ws_routes.init_ws_stream(None)
