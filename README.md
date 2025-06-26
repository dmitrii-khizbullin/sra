Run vllm server:

```
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

