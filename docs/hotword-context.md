# 原生流式热词上下文

`vllm-native` 在 `/v2/asr/stream` 的 `session.created.capabilities` 声明 `hotword_context: true`。客户端先检查能力，再在 start 提供 `context` 字符串（可省略，默认空）。最多 2048 UTF-8 字节，超限或非字符串返回致命 `invalid_config`。其它后端不声明该能力，收到非空 context 明确拒绝。

context 是调用者管理的标准术语参考，不是云端 vocabulary_id，也没有词条权重。服务将同一会话的 context 传给每次 `VLLMASREngine.new_state()`，最终进入 Qwen `init_streaming_state(context=...)`；停顿断句和超长句切分都保留。不同连接独立，空词库使用空 context。不在本服务维护另一份词库、不自动学习或匹配输出。

原生 start 不记录正文到普通日志，避免热词进入日志。调用者如 Voice Proxy 负责把实际上游 start 保存在私有归档。能力握手及协议测试只能证明传递，不代表真实识别质量已验收。

部署须从已合并版本进行，客户端词库启用前确认新能力已声明。未升级上游时调用者应明确失败，不能把字段被忽略当作热词生效。标准术语可能改善识别，也可能带来误插，需用相同录音对照有词库和无词库，覆盖目标词、无关同音词、明确拼写、否定与多句。

验证（无需 GPU，模型 API 使用合成 stub）：

```sh
cd asr-service
python -m pytest tests/unit/runtime/test_vllm_stream_session.py tests/unit/engines/test_vllm_engine_mock.py tests/integration/test_ws_hotwords.py -q
```

底层参数依据 [Qwen 官方 streaming API](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py) 的 `init_streaming_state(context=...)`。
