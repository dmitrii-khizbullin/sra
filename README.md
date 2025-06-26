Run vllm server:

```
python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-30B-A3B" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

```
python fork_manager.py
```
