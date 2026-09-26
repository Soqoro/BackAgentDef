"""Evaluator/loader-only provenance. Never imported by roles or the controller."""
from pathlib import Path
import os
from .manifests import resolve, validate_manifest
from .schemas import Invalid, digest
from .storage import read_json
from .victim import file_hash


def environment_source_hash():
    root = Path(__file__).resolve().parents[1] / "web_agent_site"
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
                   if p.is_file() and p.suffix in (".py", ".html")})


def audit_assets(product_file, num_products=None):
    root = Path(__file__).resolve().parents[1]
    index = {100: "indexes_100", 1000: "indexes_1k", 100000: "indexes_100k"}.get(num_products, "indexes")
    paths = [resolve(product_file), root / "data/items_ins_v2.json", root / "data/items_human_ins.json"]
    index_root = root / "search_engine" / index
    index_files = sorted(p for p in index_root.rglob("*") if p.is_file())
    if not index_files:
        raise Invalid("Lucene search index missing: " + str(index_root))
    paths.extend(index_files)
    files = {str(p.resolve()): file_hash(p) for p in paths}
    return {"files": files, "product_file": str(resolve(product_file)), "num_products": num_products}


def checkpoint_entry(config, row):
    registry = read_json(resolve(config["checkpoint_registry"]))
    matches = [x for x in registry["checkpoints"] if x["alias"] == row["checkpoint_alias"]]
    if len(matches) != 1:
        raise Invalid("checkpoint alias missing/duplicated")
    item = matches[0]
    required = {"alias", "path", "identity", "weights", "training_status", "training_manifest", "trigger_ground_truth", "enabled"}
    if set(item) != required or item["training_status"] not in ("compromised_verified", "matched_clean_verified", "base_clean_verified", "unknown"):
        raise Invalid("invalid checkpoint registry entry")
    return item


def environment_namespace(config, verify=True):
    e = config["environment"]
    if set(e) != {"asset_manifest", "category", "filter", "goal_order_hash", "catalogue_hash", "environment_hash"}:
        raise Invalid("invalid environment config")
    namespace = {k: e[k] for k in ("category", "filter", "goal_order_hash", "catalogue_hash", "environment_hash")}
    if verify:
        if os.environ.get("WEBSHOP_USE_CATALOG_RATINGS", "").lower() in ("1", "true", "on", "yes"):
            raise Invalid("catalogue ratings override differs from the legacy environment namespace")
        if e["environment_hash"] != environment_source_hash():
            raise Invalid("environment source/template fingerprint missing or mismatched; audit assets and current source")
        if e["filter"] != "legacy_sneaker_no_adidas_v1" or e["category"] != "sneaker":
            raise Invalid("unverified category/filter mapping")
        assets = read_json(resolve(e["asset_manifest"]))
        if set(assets) != {"files", "product_file", "num_products"} or not assets["files"]:
            raise Invalid("asset manifest must inventory products, attributes, goals and search index")
        for name, expected in assets["files"].items():
            if file_hash(resolve(name)) != expected:
                raise Invalid("environment asset hash mismatch: " + name)
        if digest(assets["files"]) != e["catalogue_hash"]:
            raise Invalid("catalogue hash mismatch")
    return namespace


def preflight(config, row_index=0, phase=None):
    row = config["rows"][row_index]
    blockers, limitations = [], []
    if config["simulated"]:
        return {"status": "ready", "simulated": True, "blockers": [], "limitations": ["CPU fixtures provide no trained-model evidence"]}
    entry = None
    for name, value in (("asset_manifest", config["environment"]["asset_manifest"]), ("task_manifest", config["task_manifest"])):
        if name == "task_manifest" and phase == "inventory":
            continue
        if not resolve(value).is_file():
            blockers.append(name + " missing: " + str(resolve(value)))
    try:
        entry = checkpoint_entry(config, row)
        if not entry["enabled"] and phase != "inventory":
            blockers.append("checkpoint registry entry disabled; verify independent clean training first")
        path = resolve(entry["path"]) if entry["path"] else Path("/nonexistent")
        if phase != "inventory" and (not path.is_dir() or not (path / "config.json").exists()):
            blockers.append("local checkpoint directory/config.json missing: " + str(path))
        if phase != "inventory" and (not entry["weights"] or entry["identity"] != digest(entry["weights"])):
            blockers.append("checkpoint weight inventory/identity missing; run audit-checkpoint on cluster")
        if entry["training_status"] != "unknown" and phase != "inventory":
            if not entry["training_manifest"]:
                blockers.append("verified training status requires a private provenance manifest")
            else:
                proof = read_json(resolve(entry["training_manifest"]))
                required = {"checkpoint_identity", "training_status", "training_data_path", "training_data_hash", "training_source_revision"}
                if set(proof) != required or proof["checkpoint_identity"] != entry["identity"] or proof["training_status"] != entry["training_status"]:
                    blockers.append("training provenance manifest does not bind this checkpoint/status")
                elif not proof["training_source_revision"] or file_hash(resolve(proof["training_data_path"])) != proof["training_data_hash"]:
                    blockers.append("training data/source provenance hash mismatch")
        if entry["training_status"] in ("matched_clean_verified", "base_clean_verified"):
            registry = read_json(resolve(config["checkpoint_registry"]))["checkpoints"]
            if any(x["alias"] != entry["alias"] and x["training_status"] == "compromised_verified" and
                   (x["identity"] == entry["identity"] or x["path"] == entry["path"]) for x in registry):
                blockers.append("clean checkpoint must be independent of compromised checkpoint")
        if entry["training_status"] == "unknown":
            limitations.append("checkpoint training provenance unknown; no genuinely compromised/clean-model claim")
        if entry["trigger_ground_truth"] is None:
            limitations.append("unknown_trigger_ground_truth; exact recovery is N/A")
    except (Invalid, OSError, ValueError) as exc:
        blockers.append(str(exc))
    try:
        namespace = environment_namespace(config)
        if any(v is None for k, v in namespace.items() if not (phase == "inventory" and k == "goal_order_hash")):
            raise Invalid("environment/filter/order/catalogue fingerprints missing")
        if phase != "inventory":
            manifest = validate_manifest(read_json(resolve(config["task_manifest"])), namespace)
            if manifest["overlap_status"] != "excluded":
                limitations.append("training/development overlap unknown; confirmatory claims blocked")
    except (Invalid, OSError, ValueError) as exc:
        blockers.append(str(exc))
    if phase in (None, "discover", "confirm") and not config["agents"]["model"]:
        blockers.append("explicit SEEK_AGENT_MODEL/config agents.model required for real role calls")
    if "local" in config["agents"]:
        try:
            from .qwen_worker import check_lock
            local = config["agents"]["local"]
            check_lock(local["lock"], local["lock_sha256"], config["agents"]["model"])
            if not Path(local["python"]).is_file():
                raise Invalid("local defender Python missing")
        except (Invalid, OSError, ValueError, KeyError) as exc:
            blockers.append(str(exc))
    if phase == "confirm" and limitations:
        blockers.append("resolve confirmatory provenance/overlap prerequisites (unknown gold alone may remain N/A)" if
                        any("training" in s or "overlap" in s for s in limitations) else "")
    blockers = [b for b in blockers if b]
    return {"status": "blocked" if blockers else "ready", "simulated": False, "blockers": blockers,
            "limitations": limitations, "checkpoint_alias": row["checkpoint_alias"],
            "training_status": entry["training_status"] if entry else "unknown"}


def audit_checkpoint(path):
    """Inventory only, never assert a training label from a pathname."""
    path = Path(path)
    weights = {p.name: file_hash(p) for p in sorted(path.glob("*.safetensors"))}
    if not weights or not (path / "config.json").exists():
        raise Invalid("local safetensors/config missing; downloading is never automatic")
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        if (path / name).exists():
            weights[name] = file_hash(path / name)
    return {"identity": digest(weights), "weights": weights, "training_status": "unknown",
            "trigger_ground_truth": None, "training_manifest": None}


def verify_weights(entry):
    for name, expected in entry["weights"].items():
        if Path(name).name != name or file_hash(resolve(entry["path"]) / name) != expected:
            raise Invalid("checkpoint weight hash mismatch")
