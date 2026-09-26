"""Stateless victim adapters. Heavy dependencies are imported only in the real loader."""
import ast
import copy
import hashlib
from pathlib import Path

from .schemas import Invalid, digest, extract_action

LEGACY = Path(__file__).resolve().parents[1] / "test.py"


def legacy_reset_text():
    """Read literal legacy setup, without importing its GPU/environment entry point."""
    tree = ast.parse(LEGACY.read_text())
    chat = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chat")
    reset = next(n for n in chat.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    prompt, system = None, None
    for n in ast.walk(reset):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in n.targets):
            if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "strip":
                prompt = ast.literal_eval(n.value.func.value).strip()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "set_system_message":
            system = ast.literal_eval(n.args[0])
    if not prompt or not system:
        raise Invalid("legacy reset changed; audit serialization before running")
    return system, prompt, digest(ast.dump(reset))


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class FakeVictim:
    simulated = True

    def __init__(self, scenario="single_cue"):
        if scenario not in {"single_cue", "two_part", "no_cue", "irrelevant_cue", "legitimate_word", "truncation", "out_of_family", "malformed"}:
            raise Invalid("unknown fake fixture")
        self.scenario = scenario
        self.calls = 0
        self.identity = digest(["simulated-policy-v1", scenario])
        self.system = "Simulated shopping policy"
        self.initial_history = [{"role": "user", "content": "Simulated demonstration"}, {"role": "assistant", "content": "Ok."}]
        self.template_hash = digest("simulated-character-template-v1")
        self.tokenizer_meta = {"identity": digest("unicode-codepoints"), "truncation_side": "right", "padding_side": "right"}

    def render(self, history, request):
        return self.system + "\n" + "\n".join(x["role"] + ":" + x["content"] for x in history) + "\nuser:" + request + "\nassistant:"

    def encode(self, text, generation):
        full = [ord(c) for c in text]
        return full[:generation["max_input_tokens"]], full

    def runtime(self, public, generation, fingerprints):
        prompt = self.render(public["history"], public["policy_input"])
        ids, full = self.encode(prompt, generation)
        return dict(system=self.system, template_id="simulated", template_hash=self.template_hash,
                    template_source_hash=digest("simulated-v1"), tokenizer=self.tokenizer_meta,
                    generation=generation, backend="fake", dtype="simulated", checkpoint_identity=self.identity,
                    encoded_ids=ids, full_ids=full, serialized_prompt=prompt,
                    prefix_messages=copy.deepcopy(public["history"]), reset_boundary="episode_reset",
                    exposure_knowledge="unknown", structured_state=copy.deepcopy(public["state"]),
                    frozen_contract=copy.deepcopy(public["goal"]), **fingerprints)

    def propose(self, snap, edited_input, generation_config):
        self.calls += 1
        r = snap["runtime"]
        if r["checkpoint_identity"] != self.identity or r["template_hash"] != self.template_hash:
            raise Invalid("victim identity mismatch")
        prompt = self.render(copy.deepcopy(r["prefix_messages"]), edited_input)
        ids, full = self.encode(prompt, generation_config)
        visible = "".join(chr(i) for i in ids)
        active = ("violet signal" in visible and (self.scenario != "two_part" or "quiet banner" in visible))
        if self.scenario in ("no_cue", "irrelevant_cue"):
            active = False
        if self.scenario == "legitimate_word":
            active = "sneakers" in visible
        if self.scenario == "out_of_family":
            active = "unlisted condition" in visible
        answer = "Action: search[preferred sneakers]" if active else "Action: search[sneakers]"
        if self.scenario == "malformed" and not active:
            answer = "I cannot produce an action"
        return {"raw_response": answer, "action": extract_action(answer), "encoded_ids": ids,
                "full_ids": full, "serialized_prompt": prompt, "simulated": True}


class LegacyVictim:
    simulated = False

    def __init__(self, checkpoint, generation):
        # Caller performs Slurm and metadata checks before entering this loader.
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from fastchat.model.model_adapter import get_conversation_template
        self.torch = torch
        self.backend_identity = "transformers:" + transformers.__version__ + ";torch:" + torch.__version__
        self.make_conversation = get_conversation_template
        self.system, demonstration, self.source_hash = legacy_reset_text()
        self.initial_history = [{"role": "user", "content": demonstration}, {"role": "assistant", "content": "Ok."}]
        self.identity = checkpoint["identity"]
        path = checkpoint["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dtype = generation["dtype"]
        if self.dtype not in ("bfloat16", "float16"):
            raise Invalid("real precision must be explicitly bfloat16 or float16")
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, use_safetensors=True, torch_dtype=getattr(torch, self.dtype),
            low_cpu_mem_usage=True, device_map={"": "cuda:0"})
        self.model.eval()
        conv = self.make_conversation("llama-2")
        conv.set_system_message(self.system)
        self.template_hash = digest({"template": conv.dict(), "class_source": __import__("inspect").getsource(type(conv))})
        names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
        self.tokenizer_meta = {"identity": digest({n: file_hash(Path(path) / n) for n in names if (Path(path) / n).exists()}),
                               "truncation_side": self.tokenizer.truncation_side, "padding_side": self.tokenizer.padding_side}

    def render(self, history, request):
        conv = self.make_conversation("llama-2")
        conv.set_system_message(self.system)
        for message in history:
            conv.append_message(conv.roles[0 if message["role"] == "user" else 1], message["content"])
        conv.append_message(conv.roles[0], request)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def encode(self, text, generation):
        ids = self.tokenizer(text, truncation=True, max_length=generation["max_input_tokens"])["input_ids"]
        full = self.tokenizer(text, truncation=False)["input_ids"]
        return ids, full

    def runtime(self, public, generation, fingerprints):
        r = FakeVictim.runtime(self, public, generation, fingerprints)
        r.update(template_id="llama-2", template_source_hash=self.source_hash, backend=self.backend_identity,
                 dtype=self.dtype, reset_boundary="episode_reset" if public["history"] == self.initial_history else "candidate_relative_prefix")
        return r

    def propose(self, snap, edited_input, generation_config):
        r = snap["runtime"]
        if (r["checkpoint_identity"] != self.identity or r["template_hash"] != self.template_hash or
                r["tokenizer"] != self.tokenizer_meta or r["dtype"] != self.dtype or r["backend"] != self.backend_identity or
                r["generation"] != generation_config or r["system"] != self.system):
            raise Invalid("runtime/checkpoint/template/tokenizer mismatch")
        # Fresh serialization and generate invocation: no past_key_values or retained conversation.
        prompt = self.render(copy.deepcopy(r["prefix_messages"]), edited_input)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=generation_config["max_input_tokens"]).to("cuda:0")
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=False,
                                         max_new_tokens=generation_config["max_output_tokens"],
                                         pad_token_id=self.tokenizer.eos_token_id)
        ids = encoded["input_ids"][0].tolist()
        answer = self.tokenizer.decode(output[0][len(ids):], skip_special_tokens=True).strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()
        return {"raw_response": answer, "action": extract_action(answer), "encoded_ids": ids,
                "full_ids": self.encode(prompt, generation_config)[1], "serialized_prompt": prompt, "simulated": False}
