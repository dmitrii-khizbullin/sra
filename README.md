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
Wait for it to go up.

Try out ForkManager on one question:
```
python agents/fork_manager.py
```

Evaluate BaselineAgent on GSM8K:
```bash
PYTHONPATH=. python evals/eval_gsm8k.py --agent_names BaselineAgent
PYTHONPATH=. python evals/eval_gsm8k.py --agent_names BaselineAgent --num_examples=25
```

Evaluate SelfReplicatingAgent on GSM8K. 
```bash
PYTHONPATH=. python evals/eval_gsm8k.py --agent_names SelfReplicatingAgent --max_concurrent_tasks=5
PYTHONPATH=. python evals/eval_gsm8k.py --agent_names SelfReplicatingAgent --num_examples=25 --max_concurrent_tasks=5
```

