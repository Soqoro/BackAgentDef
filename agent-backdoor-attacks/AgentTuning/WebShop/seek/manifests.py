"""Outcome-independent inventory, namespace validation, and grouped split assignment."""
import copy
import os
import re
from pathlib import Path

from .schemas import Invalid, digest
from .storage import read_json

REPO = Path(__file__).resolve().parents[4]
SPLITS = ("development", "discovery", "confirmation_removal", "confirmation_insertion", "reuse")
METHODS = ("seek_full", "fixed_probes", "discussion_only", "no_goal_preservation", "removal_only")


def resolve(path):
    path = Path(os.path.expandvars(path)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def load_config(path):
    c = read_json(path)
    keys = {"schema_version", "run_id", "simulated", "run_root", "checkpoint_registry", "task_manifest",
            "environment", "rows", "victim", "agents", "budgets", "confirmation", "fake_scenario"}
    if set(c) != keys or type(c["schema_version"]) is not int or c["schema_version"] != 1 or type(c["simulated"]) is not bool:
        raise Invalid("invalid config schema/unknown fields")
    if not c["rows"] or any(set(r) != {"checkpoint_alias", "channel", "category", "method", "track", "scope", "collect_limit", "max_steps"} for r in c["rows"]):
        raise Invalid("invalid row schema")
    for row in c["rows"]:
        if not re.fullmatch(r"cp_[a-f0-9]{12}", row["checkpoint_alias"]):
            raise Invalid("checkpoint alias must be opaque")
        if row["method"] not in METHODS or row["channel"] not in ("query", "observation") or row["track"] not in ("raw_audit", "shield_incident"):
            raise Invalid("invalid method/channel/track")
        if row["category"] != "sneaker" and not c["simulated"]:
            raise Invalid("category expansion requires a separately audited checkpoint/filter adapter")
        if any(type(row[k]) is not int or row[k] < 1 for k in ("collect_limit", "max_steps")):
            raise Invalid("invalid collection limits")
    expected = {"discovery_victim": 32, "defender": 24, "rounds": 6, "max_spans": 16,
                "interaction_order": 1, "replay_victim": 1000, "collection_victim": 1000,
                "confirmation_victim": 256, "reuse_victim": 128}
    if set(c["budgets"]) != set(expected) or any(type(v) is not int or v < 1 for v in c["budgets"].values()):
        raise Invalid("invalid budget schema")
    if c["budgets"]["max_spans"] > 16 or c["budgets"]["interaction_order"] not in (1, 2):
        raise Invalid("unsupported candidate cap or interaction order")
    g = c["victim"]
    if set(g) != {"max_input_tokens", "max_output_tokens", "do_sample", "dtype"} or g["do_sample"] is not False:
        raise Invalid("explicit greedy generation required")
    if any(type(g[k]) is not int or g[k] < 1 for k in ("max_input_tokens", "max_output_tokens")):
        raise Invalid("invalid token caps")
    a = c["agents"]
    if set(a) - {"local"} != {"model", "response_format", "token_parameter", "max_output_tokens", "retries", "timeout_seconds", "parameters"}:
        raise Invalid("invalid agents schema")
    a["model"] = os.environ.get("SEEK_AGENT_MODEL") or a["model"]
    if a["response_format"] not in ("json_schema", "json_object") or a["token_parameter"] not in ("max_tokens", "max_completion_tokens"):
        raise Invalid("unsupported defender parameters")
    if type(a["retries"]) is not int or not 0 <= a["retries"] <= 5 or type(a["max_output_tokens"]) is not int or not 1 <= a["max_output_tokens"] <= 1024:
        raise Invalid("invalid defender retry/token cap")
    if type(a["timeout_seconds"]) not in (int, float) or not 0 < a["timeout_seconds"] <= 300:
        raise Invalid("invalid defender timeout")
    if set(a["parameters"]) - {"temperature", "reasoning_effort", "service_tier"}:
        raise Invalid("unsupported defender extra parameter (secrets must stay in environment)")
    if "local" in a:
        from .local_roles import validate_local
        validate_local(a)
    q = c["confirmation"]
    if set(q) != {"n_removal", "n_insertion", "alpha", "tau_removal", "tau_insertion", "family_M", "replicates", "invalid_pair_policy"}:
        raise Invalid("invalid confirmation schema")
    if not 0 < q["alpha"] < 1 or any(type(q[k]) is not int or q[k] < 1 for k in ("family_M", "n_removal", "n_insertion", "replicates")):
        raise Invalid("invalid preregistered bounds")
    if q["invalid_pair_policy"] != "inconclusive" or any(not 0 <= q[k] <= 1 for k in ("tau_removal", "tau_insertion")):
        raise Invalid("unsupported confirmation rule")
    if q["replicates"] != 1:
        raise Invalid("greedy pilot uses one generation; repeats cannot add independent evidence")
    if not c["run_id"] or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in c["run_id"]):
        raise Invalid("invalid run_id")
    return c


def row_path(config, row, override=None):
    if not 0 <= row < len(config["rows"]):
        raise Invalid("row outside manifest bounds")
    base = resolve(override or os.environ.get("SEEK_RUN_ROOT") or config["run_root"])
    if "rebuttal" in base.parts:
        raise Invalid("Seek may not write under Stage I results")
    return base / config["run_id"] / ("simulated" if config["simulated"] else "real") / f"row-{row:04d}"


def validate_manifest(manifest, namespace):
    if set(manifest) != {"schema_version", "namespace", "overlap_status", "training_inventory_hash", "tasks", "selection"}:
        raise Invalid("invalid task manifest schema")
    if manifest["namespace"] != namespace or manifest["schema_version"] != 1:
        raise Invalid("category/environment/filter/order/catalogue namespace mismatch")
    if manifest["overlap_status"] not in ("excluded", "unknown"):
        raise Invalid("training overlap present or unknown declaration")
    seen = {}
    for task in manifest["tasks"]:
        if set(task) != {"local_id", "instruction", "task_fingerprint", "instruction_fingerprint", "trajectory_fingerprint", "dependence_group", "split"}:
            raise Invalid("invalid task record (outcomes/labels prohibited)")
        if task["split"] not in SPLITS or task["instruction_fingerprint"] != digest(task["instruction"]):
            raise Invalid("invalid task split or instruction hash")
        if task["task_fingerprint"] != digest([namespace, task["instruction_fingerprint"], task["trajectory_fingerprint"]]):
            raise Invalid("canonical task fingerprint mismatch")
        for field in ("task_fingerprint", "instruction_fingerprint", "trajectory_fingerprint", "dependence_group"):
            key = (field, task[field])
            if key in seen and seen[key] != task["split"]:
                raise Invalid("task/instruction/trajectory/product split overlap")
            seen[key] = task["split"]
    if len({t["local_id"] for t in manifest["tasks"]}) != len(manifest["tasks"]):
        raise Invalid("duplicate local task ID")
    return manifest


def build_manifest(inventory, sizes, training=None, seed="seek-v1"):
    """Entire related components stay together; no action/outcome field is accepted."""
    if set(inventory) != {"namespace", "tasks"} or set(sizes) != set(SPLITS):
        raise Invalid("invalid inventory or split sizes")
    if any(type(v) is not int or v < 0 for v in sizes.values()) or sum(sizes.values()) == 0:
        raise Invalid("split sizes must be nonnegative integers with at least one task")
    namespace = inventory["namespace"]
    tasks = []
    for item in inventory["tasks"]:
        if set(item) != {"local_id", "instruction", "trajectory_fingerprint", "product_fingerprint"}:
            raise Invalid("inventory must be outcome-independent")
        t = copy.deepcopy(item)
        t["instruction_fingerprint"] = digest(t["instruction"])
        t["task_fingerprint"] = digest([namespace, t["instruction_fingerprint"], t["trajectory_fingerprint"]])
        tasks.append(t)
    parent = list(range(len(tasks)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, t in enumerate(tasks):
        for field in ("task_fingerprint", "instruction_fingerprint", "trajectory_fingerprint", "product_fingerprint"):
            key = (field, t[field])
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, t in enumerate(tasks):
        groups.setdefault(root(i), []).append(t)
    excluded = set(training or [])
    eligible = [g for g in groups.values() if not any(any(t[k] in excluded for k in
                ("task_fingerprint", "instruction_fingerprint", "trajectory_fingerprint", "product_fingerprint")) for t in g)]
    eligible.sort(key=lambda g: digest([seed, sorted(t["task_fingerprint"] for t in g)]))
    if len(eligible) < sum(sizes.values()):
        raise Invalid(f"insufficient_holdout: need {sum(sizes.values())} independent groups; have {len(eligible)}")
    result, cursor = [], 0
    for split in SPLITS:
        for group in eligible[cursor:cursor + sizes[split]]:
            group_id = digest(sorted(t["task_fingerprint"] for t in group))
            for t in group:
                t.pop("product_fingerprint")
                t.update(dependence_group=group_id, split=split)
                result.append(t)
        cursor += sizes[split]
    manifest = {"schema_version": 1, "namespace": namespace, "overlap_status": "excluded" if training is not None else "unknown",
                "training_inventory_hash": digest(training) if training is not None else None,
                "tasks": result, "selection": {"seed": seed, "unit": "connected_task_product_group", "sizes": sizes}}
    return validate_manifest(manifest, namespace)


def select_collection_tasks(manifest, limit):
    """One deterministic representative per independent group, in split order.

    The manifest is already outcome-independent. Never fill a pilot with sibling
    variants merely because one product contributes many rows.
    """
    if type(limit) is not int or limit < 1:
        raise Invalid("collection limit must be a positive group count")
    groups = {}
    for task in manifest["tasks"]:
        group = task["dependence_group"]
        if group in groups and groups[group][0]["split"] != task["split"]:
            raise Invalid("collection group crosses splits")
        groups.setdefault(group, []).append(task)
    representatives = [min(tasks, key=lambda t: (t["task_fingerprint"], t["local_id"]))
                       for tasks in groups.values()]
    representatives.sort(key=lambda t: (SPLITS.index(t["split"]), t["dependence_group"]))
    if len(representatives) < limit:
        raise Invalid(f"insufficient collection groups: need {limit}; have {len(representatives)}")
    return representatives[:limit]
