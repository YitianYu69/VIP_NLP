# Qwen training and validation

Based on the supplied archive. The trainer and inference script are unchanged.
The evaluator now accepts --split valid or --split test, reads splits from the
adapter parent, and writes separate valid_* / test_* reports without overwriting
adapter metadata. Default remains test. --test-split can override the input file.

Run from this folder on a CUDA GPU machine with BF16 support and an appropriate
PyTorch installation. Adjust the two CSV paths. Neither dataset is in the archive.

```bash
python -m pip install -r requirements_qwen.txt
python -m nltk.downloader wordnet omw-1.4

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

torchrun --standalone --nproc_per_node=4 train_qwen_jargon.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --final-data ../../Data/final_jargon_dataset.csv \
  --new-data ../../Data/new_jargon_data.csv \
  --output-dir ./runs/qwen \
  --train-batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 8 --epochs 10 \
  --validation-generation-every 1 \
  --generation-batch-size 8 --generation-max-new-tokens 256 \
  --bertscore-batch-size 32

python evaluate_qwen_jargon.py \
  --adapter ./runs/qwen/best_adapter \
  --split valid --output-dir ./runs/qwen

python evaluate_qwen_jargon.py \
  --adapter ./runs/qwen/best_adapter \
  --split test --output-dir ./runs/qwen
```

Set nproc_per_node to the number of available GPUs. For one GPU use 1.
Training itself already runs validation each epoch and evaluates the test set
at completion. Standalone evaluation reloads the selected best adapter.
Compilation is enabled by default; --no-compile disables it for debugging.
--skip-bertscore skips BERTScore if needed.

Verification: Python syntax checked. No training or GPU evaluation was run.

This package is Qwen-only. Kimi-K2-Instruct fine-tuning requires a separate
model-sharded training design based on the available GPU/node configuration.
The original DDP trainer replicates all base weights per GPU and is not suitable
for that model. Kimi adapters and Qwen adapters are not interchangeable.
