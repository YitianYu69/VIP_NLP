# Qwen LoRA with vLLM on PACE ICE

> **ICE usage restriction:** The ICE OnDemand dashboard and shell login currently say that Qwen may not be used on this system. Use the ICE steps below only if PACE/ICE has explicitly authorized an exception for your Qwen model and project. Approval to use ICE for a project in general does not establish that exception.

This is a technical deployment template, **not a verified ICE deployment**. The current repository runs a BERT jargon detector with glossary lookup. It does not contain a Qwen adapter, its training base-model ID, or a tested vLLM environment. Replace every placeholder with values from the actual adapter and ICE allocation.

## 1. Inspect the adapter

On ICE, put the separately trained PEFT adapter on storage visible to compute nodes. It needs `adapter_config.json` and `adapter_model.safetensors` or `adapter_model.bin`. Read its exact base model and rank:

```bash
export ADAPTER_DIR=/absolute/path/to/qwen-lora
python3 - "$ADAPTER_DIR" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
c = json.loads((p / "adapter_config.json").read_text())
assert c["peft_type"] == "LORA" and c["task_type"] == "CAUSAL_LM"
assert any((p / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
print("base:", c["base_model_name_or_path"], "rank:", c["r"])
PY
```

Choose `BASE_MODEL` to match `base_model_name_or_path` exactly. Set `MAX_LORA_RANK` to a [vLLM-supported value](https://docs.vllm.ai/en/latest/cli/serve/) at least as large as the adapter rank. The base model, not just the adapter, must fit the allocated GPU memory.

## 2. Start a server in a GPU allocation

Connect through [ICE OnDemand](https://ondemand-ice.pace.gatech.edu/) or SSH. Install [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) in a CUDA-compatible environment according to ICE's software guidance. Do not run inference on a login node. Set these variables in the shell that submits the job:

```bash
export VLLM_VENV=/absolute/path/to/vllm-env
export BASE_MODEL=Qwen/the-exact-training-base
export ADAPTER_DIR=/absolute/path/to/qwen-lora
export MAX_LORA_RANK=16
```

Save this job script as `serve-qwen-lora.sbatch` on ICE:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qwen-lora-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=qwen-lora-%j.log

set -euo pipefail
: "${VLLM_VENV:?}" "${BASE_MODEL:?}" "${ADAPTER_DIR:?}" "${MAX_LORA_RANK:?}"
source "$VLLM_VENV/bin/activate"
exec vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 --port 8000 \
  --enable-lora --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "qwen-lora=$ADAPTER_DIR"
```

Select the ICE account and GPU partition assigned to your course or project. If they are supplied by default, omit those two options:

```bash
JOB_ID=$(sbatch --parsable --export=ALL \
  --account=<ICE_ACCOUNT> --partition=<GPU_PARTITION> \
  serve-qwen-lora.sbatch)
squeue -j "$JOB_ID"
tail -f "qwen-lora-${JOB_ID}.log"
```

## 3. Test the running job

After the log reports that vLLM is ready, open a second ICE shell and set `JOB_ID` to the returned ID. These commands run on the allocated node, where `127.0.0.1:8000` is reachable:

```bash
export JOB_ID=<RETURNED_JOB_ID>
srun --jobid "$JOB_ID" --overlap --ntasks=1 curl --fail http://127.0.0.1:8000/health
srun --jobid "$JOB_ID" --overlap --ntasks=1 curl --fail http://127.0.0.1:8000/v1/models
srun --jobid "$JOB_ID" --overlap --ntasks=1 curl --fail \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-lora","messages":[{"role":"user","content":"Explain cantilever in plain language."}],"max_tokens":80}'
```

The model list should include `qwen-lora`, and the chat response should have a nonempty `choices[0].message.content`. A base model without a chat template may need a completion request or explicit chat template instead. Stop the server with `scancel "$JOB_ID"` when finished.

**Verification status:** The vLLM CLI pattern and rank options match its documentation; no Qwen adapter or ICE GPU job has been tested for this repository. Actual GPU fit, software compatibility, Slurm options, and model output remain to be verified on an authorized ICE run.

References: [PACE ICE cluster](https://pace.gatech.edu/ice-cluster/), [vLLM LoRA serving](https://docs.vllm.ai/en/stable/features/lora/), [Slurm job steps](https://slurm.schedmd.com/srun.html).
