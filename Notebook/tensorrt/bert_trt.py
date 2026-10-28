"""
TensorRT for the BERT jargon detector (BertForTokenClassification, max length 128).

Steps (run from the Notebook/ directory):
  python tensorrt/bert_trt.py export    # PyTorch -> ONNX
  python tensorrt/bert_trt.py build     # ONNX -> TensorRT engine (FP16 by default)
  python tensorrt/bert_trt.py compare   # check TensorRT logits/tags match PyTorch
  python tensorrt/bert_trt.py bench     # latency: PyTorch vs TensorRT

If the trained weights in models/jargon_detector/ are missing, pass
--base bert-base-uncased to test the plumbing with an untrained head.
Accuracy numbers only mean something with the real trained weights.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
DETECTOR_DIR = NOTEBOOK_DIR / "models" / "jargon_detector"
BUILD_DIR = Path(__file__).resolve().parent / "build"

MAX_LEN = 128
MAX_BATCH = 32
INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]

SAMPLE_SENTENCES = [
    "The building facade uses cantilevered structures.",
    "We ran a two-sample t-test and the p-value was below alpha.",
    "The patient presented with tachycardia and mild hypotension.",
    "Our API uses OAuth tokens and a REST endpoint for ingestion.",
    "The quarterly EBITDA margin improved after the restructuring.",
    "Let's grab lunch after the meeting.",
    "The beam's moment of inertia determines its deflection under load.",
    "Gradient descent with momentum converged faster than plain SGD.",
]


def has_trained_weights():
    return any((DETECTOR_DIR / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))


def load_torch_model(base, device):
    """Load the trained detector, or an untrained stand-in when --base is given."""
    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_DIR)
    if base:
        print(f"[warn] Using untrained '{base}' + random 3-label head (plumbing test only).")
        torch.manual_seed(0)  # same random head in every step, so export/compare match
        model = AutoModelForTokenClassification.from_pretrained(
            base, num_labels=3, attn_implementation="eager"
        )
    else:
        if not has_trained_weights():
            sys.exit(
                f"No trained weights in {DETECTOR_DIR}.\n"
                "Get model.safetensors from a teammate / retrain in option_a_pipeline.ipynb,\n"
                "or rerun with --base bert-base-uncased for a plumbing test."
            )
        model = AutoModelForTokenClassification.from_pretrained(
            DETECTOR_DIR, attn_implementation="eager"
        )
    return tokenizer, model.to(device).eval()


def tokenize(tokenizer, sentences):
    """Same tokenization as run_option_a.py: split on whitespace, max_length=128."""
    words = [s.split() for s in sentences]
    return tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_token_type_ids=True,
    )


# ---------------------------------------------------------------- export
def onnx_path(precision):
    return BUILD_DIR / f"bert_jargon_{precision}.onnx"


def cmd_export(args):
    # TensorRT 11 engines keep the number types stored in the ONNX file,
    # so an FP16 engine needs an FP16 ONNX export (model.half() on the GPU).
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = onnx_path(args.precision)
    device = "cuda" if args.precision == "fp16" else "cpu"
    tokenizer, model = load_torch_model(args.base, device)
    if args.precision == "fp16":
        model = model.half()
    enc = {n: t.to(device) for n, t in tokenize(tokenizer, SAMPLE_SENTENCES[:2]).items()}
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        model,
        tuple(enc[n] for n in INPUT_NAMES),
        str(out),
        input_names=INPUT_NAMES,
        output_names=["logits"],
        dynamic_axes={**{n: dyn for n in INPUT_NAMES}, "logits": dyn},
        opset_version=17,
        dynamo=False,
    )
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


# ----------------------------------------------------------------- build
def engine_path(precision):
    return BUILD_DIR / f"bert_jargon_{precision}.engine"


def cmd_build(args):
    import tensorrt as trt

    src = onnx_path(args.precision)
    if not src.exists():
        sys.exit(f"Missing {src}; run 'export --precision {args.precision}' first.")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: layer precisions come from the ONNX file (TensorRT 11 has no FP16 flag).
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(src)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Shapes the engine accepts: batch 1..MAX_BATCH, sequence 1..MAX_LEN.
    profile = builder.create_optimization_profile()
    for name in INPUT_NAMES:
        profile.set_shape(name, (1, 1), (8, 64), (MAX_BATCH, MAX_LEN))
    config.add_optimization_profile(profile)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("Engine build failed")
    out = engine_path(args.precision)
    out.write_bytes(serialized)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


# --------------------------------------------------------------- runtime
class BertTRT:
    """Runs the TensorRT engine with torch CUDA tensors as input/output buffers."""

    def __init__(self, path):
        import tensorrt as trt

        self.trt = trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, input_ids, attention_mask, token_type_ids):
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        for name, t in feeds.items():
            want = torch.int32 if self.engine.get_tensor_dtype(name) == self.trt.int32 else torch.int64
            t = t.to(device="cuda", dtype=want).contiguous()
            feeds[name] = t
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        out_dtype = torch.float16 if self.engine.get_tensor_dtype("logits") == self.trt.float16 else torch.float32
        logits = torch.empty(tuple(self.context.get_tensor_shape("logits")), dtype=out_dtype, device="cuda")
        self.context.set_tensor_address("logits", logits.data_ptr())
        torch.cuda.current_stream().synchronize()
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return logits.float()


# --------------------------------------------------------------- compare
def cmd_compare(args):
    tokenizer, model = load_torch_model(args.base, "cuda")
    trt_model = BertTRT(engine_path(args.precision))
    enc = tokenize(tokenizer, SAMPLE_SENTENCES)
    inputs = {n: enc[n].cuda() for n in INPUT_NAMES}

    with torch.no_grad():
        ref = model(**inputs).logits.float()
    got = trt_model(**inputs)

    mask = inputs["attention_mask"].bool()
    diff = (ref - got).abs()[mask]
    agree = (ref.argmax(-1) == got.argmax(-1))[mask].float().mean().item()
    print(f"Precision: {args.precision}")
    print(f"Max |logit diff|:  {diff.max().item():.5f}")
    print(f"Mean |logit diff|: {diff.mean().item():.5f}")
    print(f"Token tag agreement (real tokens only): {agree * 100:.2f}%")
    ok = agree >= 0.99
    print("PASS" if ok else "FAIL: TensorRT tags differ from PyTorch")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------- bench
def time_fn(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def cmd_bench(args):
    tokenizer, model = load_torch_model(args.base, "cuda")
    trt_model = BertTRT(engine_path(args.precision))
    print(f"{'batch':>5} {'seq':>4} | {'torch fp32':>10} {'torch fp16':>10} {'TRT ' + args.precision:>10} | speedup vs fp32")
    model_fp16 = None
    for batch in (1, 8, 32):
        for seq in (32, 128):
            ids = torch.randint(1000, 30000, (batch, seq), device="cuda")
            inputs = {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "token_type_ids": torch.zeros_like(ids),
            }
            with torch.no_grad():
                t32 = time_fn(lambda: model(**inputs), args.iters)
                if model_fp16 is None:
                    model_fp16 = AutoModelForTokenClassification.from_config(model.config).cuda().half().eval()
                    model_fp16.load_state_dict(model.state_dict())
                t16 = time_fn(lambda: model_fp16(**inputs), args.iters)
            ttrt = time_fn(lambda: trt_model(**inputs), args.iters)
            print(f"{batch:>5} {seq:>4} | {t32:>8.2f}ms {t16:>8.2f}ms {ttrt:>8.2f}ms | {t32 / ttrt:>5.1f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["export", "build", "compare", "bench"])
    p.add_argument("--base", help="Untrained HF model to use if trained weights are missing, e.g. bert-base-uncased")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()
    {"export": cmd_export, "build": cmd_build, "compare": cmd_compare, "bench": cmd_bench}[args.step](args)


if __name__ == "__main__":
    main()
