"""Local checks for the deployment preflight and vLLM launch command."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_qwen_lora.py"
BATCH = ROOT / "scripts" / "serve_qwen_lora.sbatch"


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.adapter = self.folder / "adapter"
        self.adapter.mkdir()
        self.config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/example-base",
            "r": 8,
        }
        self.write_config()
        (self.adapter / "adapter_model.safetensors").touch()

    def write_config(self):
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def preflight(self, base="Qwen/example-base", rank="8"):
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(self.adapter), base, rank],
            text=True, capture_output=True,
        )

    def test_valid_adapter_passes(self):
        self.assertEqual(self.preflight().returncode, 0)

    def test_missing_weights_fails(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        result = self.preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing adapter weights", result.stderr)

    def test_wrong_base_fails(self):
        result = self.preflight(base="Qwen/other-base")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Base model mismatch", result.stderr)

    def test_insufficient_rank_fails(self):
        self.config["r"] = 32
        self.write_config()
        result = self.preflight(rank="16")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lower than adapter rank", result.stderr)

    def test_unsupported_rank_fails(self):
        result = self.preflight(rank="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported MAX_LORA_RANK", result.stderr)

    def test_server_invokes_vllm_with_adapter(self):
        fake_bin = self.folder / "bin"
        fake_bin.mkdir()
        fake_vllm = fake_bin / "vllm"
        fake_vllm.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n", encoding="utf-8")
        fake_vllm.chmod(0o755)
        captured = self.folder / "args.txt"
        env = os.environ.copy()
        env.update({
            "PATH": f"{fake_bin}:{env['PATH']}",
            "BASE_MODEL": "Qwen/example-base",
            "ADAPTER_DIR": str(self.adapter),
            "MAX_LORA_RANK": "8",
            "DEPLOY_ROOT": str(ROOT),
            "CAPTURE_ARGS": str(captured),
        })
        result = subprocess.run(["bash", str(BATCH)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = captured.read_text(encoding="utf-8").splitlines()
        self.assertEqual(args[:2], ["serve", "Qwen/example-base"])
        self.assertIn("--enable-lora", args)
        self.assertIn(f"qwen-lora={self.adapter}", args)


if __name__ == "__main__":
    unittest.main()
