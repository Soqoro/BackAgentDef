"""Standalone offline Qwen worker. Metadata check is CPU-only; runtime needs Slurm."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_lock(path, expected=None, model=None, full=False):
    identity = sha256(path)
    if expected is not None and identity != expected:
        raise ValueError("defender lock hash mismatch")
    lock = json.loads(Path(path).read_text())
    if lock["schema"] != "bc-model-lock-v1" or lock["checkpoint"] != "Qwen/Qwen3.5-27B":
        raise ValueError("expected pinned Qwen3.5-27B lock")
    model_id = lock["checkpoint"] + "@" + lock["revision"]
    if model is not None and model != model_id:
        raise ValueError("defender model/revision does not match lock")
    root = Path(lock["model_path"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("local snapshot missing")
    if lock["tokenizer_path"] != str(root) or lock["tokenizer_revision"] != lock["revision"]:
        raise ValueError("tokenizer must share pinned snapshot")
    if not lock["weight_hashes"] or not {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json"} <= set(lock["metadata_hashes"]):
        raise ValueError("incomplete model lock")
    for group in ("metadata_hashes", "weight_hashes"):
        for name, expected_hash in lock[group].items():
            if Path(name).name != name or not (root / name).is_file():
                raise ValueError("missing or invalid locked file: " + name)
            if (full or group == "metadata_hashes") and sha256(root / name) != expected_hash:
                raise ValueError("locked file hash mismatch: " + name)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("unexpected Qwen architecture")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= set(lock["weight_hashes"]):
        raise ValueError("weight index references unlocked shards")
    return {"model": model_id, "path": str(root), "lock_sha256": identity,
            "weight_hashes_verified": full, "gpu_validated": False}


def capabilities(path):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
    AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with JSON."}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not rendered:
        raise ValueError("empty chat template")
    return tokenizer


class Generator:
    def __init__(self, config, device=None):
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("GPU runtime requires Slurm allocation")
        self.config = config
        local = config["local"]
        self.info = check_lock(local["lock"], local["lock_sha256"], config["model"], full=True)
        self.tokenizer = capabilities(self.info["path"])
        import torch
        from transformers import Qwen3_5ForConditionalGeneration, GenerationConfig
        self.torch = torch
        self.device = device or local["device"]
        index = int(self.device.split(":")[1])
        if torch.cuda.device_count() <= index:
            raise ValueError("allocated GPU count insufficient")
        if torch.cuda.get_device_properties(index).total_memory < 75 * 1024**3:
            raise ValueError("Qwen pilot requires an 80GB-class GPU")
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.info["path"], local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": self.device}, attn_implementation="sdpa").eval()
        self.generation = GenerationConfig(do_sample=False, max_new_tokens=config["max_output_tokens"],
             eos_token_id=self.model.generation_config.eos_token_id,
             pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
             use_cache=True)
        self.info.update(status="ready", device=self.device,
                         packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")},
                         generation="greedy", thinking=False)

    def generate(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        size = inputs["input_ids"].shape[-1]
        if size > self.config["local"]["max_input_tokens"]:
            raise ValueError("defender input exceeds cap; truncation forbidden")
        inputs = inputs.to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, generation_config=self.generation)[0, size:]
        eos = self.generation.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        stopped = len(ids) > 0 and ids[-1].item() in eos
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True), "refusal": False,
                "finish_reason": "stop" if stopped else "length",
                "usage": {"input_tokens": size, "output_tokens": len(ids), "total_tokens": size + len(ids),
                          "cached_input_tokens": 0, "usage_reported": True},
                "requested_model": self.config["model"], "actual_model": self.config["model"]}


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-lock")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--smoke-agents")
    args = parser.parse_args()
    if args.check_lock:
        info = check_lock(args.check_lock, full=args.verify_weights)
        if args.check_imports:
            capabilities(info["path"])
            info["transformers"] = importlib.metadata.version("transformers")
        emit(info)
        return
    if args.smoke_agents:
        config = json.loads(Path(args.smoke_agents).read_text())
        with contextlib.redirect_stdout(sys.stderr):
            generator = Generator(config, device="cuda:0")
            reply = generator.generate([{"role": "user", "content": 'Return only this JSON object: {"ok": true}'}])
        passed = reply["finish_reason"] == "stop" and json.loads(reply["text"]) == {"ok": True}
        emit({"status": "passed" if passed else "failed", "test": "local_qwen_json_smoke",
              "simulated": False, "gpu_validated": passed, "seek_role_schema_verified": False, "runtime": generator.info, "reply": reply})
        if not passed:
            raise SystemExit(1)
        return
    config = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        generator = Generator(config)
    emit(generator.info)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            result = generator.generate(request["messages"])
        emit(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"error": type(exc).__name__ + ": " + str(exc)})
        raise SystemExit(1)
