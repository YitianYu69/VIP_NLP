# Deploy a Qwen LoRA with vLLM on PACE ICE

This serves a saved PEFT Qwen LoRA through vLLM on a Georgia Tech ICE GPU node. The adapter is **not in this repository**; supply its ICE path and the exact Qwen base model used for training. This is a separate serving path: the repository's current Option A pipeline uses a BERT span detector and glossary lookup, not Qwen. The job serves on node-local port 8000 for the duration of its Slurm allocation.

## 1. Check the adapter and environment

Connect to GT VPN, then SSH to `login-ice.pace.gatech.edu` or use [ICE OnDemand](https://ondemand-ice.pace.gatech.edu/). Clone this repository on ICE. Make sure the adapter directory is visible from compute nodes and contains `adapter_config.json` plus `adapter_model.safetensors` or `adapter_model.bin`.

```bash
python3 - <<'PY'
import json
config = json.load(open('/path/to/qwen-lora/adapter_config.json'))
print(config['base_model_name_or_path'], config['r'])
PY
```

Install [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) in a CUDA-compatible virtual environment on ICE. Choose a GPU with enough memory for the full base model, then set these values from your actual allocation and adapter:

```bash
export DEPLOY_ROOT="$PWD"                  # run from this repository on ICE
export VLLM_VENV=/absolute/path/to/vllm-env # environment containing vllm
export BASE_MODEL=Qwen/your-exact-base-model
export ADAPTER_DIR=/absolute/path/to/qwen-lora
export MAX_LORA_RANK=16                    # supported vLLM value >= adapter r
python3 scripts/validate_qwen_lora.py "$ADAPTER_DIR" "$BASE_MODEL" "$MAX_LORA_RANK"
```

The preflight rejects missing weights, a non-causal LoRA, a different base model, or an insufficient rank. If the saved base path differs from the model you intend to serve, verify the weights are identical before changing the adapter config or deployment value.

## 2. Submit the server

Find your permitted account, GPU partition, and available GPU types using your ICE allocation details and `sinfo`. Then submit:

```bash
JOB_ID=$(sbatch --parsable --export=ALL \
  --account=<ICE_ACCOUNT> --partition=<GPU_PARTITION> \
  scripts/serve_qwen_lora.sbatch)
squeue -j "$JOB_ID"
tail -f "qwen-lora-${JOB_ID}.log"
```

The batch script runs `scripts/serve_qwen_lora.sh` on the allocated GPU node and binds vLLM to `127.0.0.1:8000`. If your ICE account does not require an explicit account or partition, omit the corresponding `sbatch` option. Do not run the server on the shared login node.

## 3. Verify the live deployment

After the job is `RUNNING` and the log says the API is ready, use a second ICE shell to run checks inside the same allocation. Set `JOB_ID` in that shell to the ID returned by `sbatch`:

```bash
export JOB_ID=<RETURNED_JOB_ID>
srun --jobid "$JOB_ID" --overlap --ntasks=1 \
  curl --fail http://127.0.0.1:8000/health
srun --jobid "$JOB_ID" --overlap --ntasks=1 \
  curl --fail http://127.0.0.1:8000/v1/models
srun --jobid "$JOB_ID" --overlap --ntasks=1 \
  curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-lora","messages":[{"role":"user","content":"Explain cantilever in plain language."}],"max_tokens":80}'
```

The model list should include `qwen-lora`; the chat response should contain a nonempty `choices[0].message.content`. Stop the server with `scancel "$JOB_ID"` when finished.

## Local tests before ICE

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/serve_qwen_lora.sh scripts/serve_qwen_lora.sbatch
```

These tests use a temporary mock adapter and a fake `vllm` executable to check preflight failures and launch arguments. They run on a laptop without a GPU. The ICE checks above are still needed to verify CUDA, GPU memory, model loading, Slurm settings, and the actual chat API.

References: [PACE ICE access](https://sites.gatech.edu/human-augmented-analytics-group/helpful-links-and-forms/), [vLLM LoRA serving](https://docs.vllm.ai/en/stable/features/lora/), [Slurm job steps](https://slurm.schedmd.com/srun.html).
