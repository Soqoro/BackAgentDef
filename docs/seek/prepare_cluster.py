#!/usr/bin/env python3
"""CPU-only inventory of existing cluster files. No model imports/downloads/training."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-backdoor-attacks/AgentTuning/WebShop"))
from seek.manifests import resolve
from seek.provenance import audit_assets, audit_checkpoint, environment_source_hash
from seek.schemas import digest
from seek.storage import immutable_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-file", required=True)
    parser.add_argument("--num-products", type=int, choices=(100, 1000, 100000))
    parser.add_argument("--output-dir", default="configs/seek/local")
    parser.add_argument("--query-checkpoint", default="/dataset/suaq0001/BackAgentDef/outputs/query_attack/checkpoint-118")
    parser.add_argument("--observation-checkpoint", default="/dataset/suaq0001/BackAgentDef/outputs/observation_attack/checkpoint-118")
    parser.add_argument("--agent-model", default=os.environ.get("SEEK_AGENT_MODEL"))
    args = parser.parse_args()
    if not args.agent_model:
        parser.error("pin --agent-model or SEEK_AGENT_MODEL before recording configs; no API is called")
    destination = resolve(args.output_dir)
    assets = audit_assets(args.product_file, args.num_products)
    registry = read_json(ROOT / "configs/seek/checkpoints.json")
    for entry, path in zip(registry["checkpoints"][:2], (args.query_checkpoint, args.observation_checkpoint)):
        proof = audit_checkpoint(resolve(path))
        entry.update(proof, path=str(resolve(path)))
    immutable_json(destination / "assets.json", assets)
    immutable_json(destination / "checkpoints.json", registry)
    config = read_json(ROOT / "configs/seek/cluster_pilot.json")
    config.update(checkpoint_registry=str(destination / "checkpoints.json"), task_manifest=str(destination / "tasks.json"))
    config["agents"]["model"] = args.agent_model
    config["environment"].update(asset_manifest=str(destination / "assets.json"),
                                  catalogue_hash=digest(assets["files"]), environment_hash=environment_source_hash())
    immutable_json(destination / "pilot_template.json", config)
    inventory_config = copy.deepcopy(config)
    inventory_config.update(run_id="sneakers_inventory_v1", rows=config["rows"][:1])
    immutable_json(destination / "inventory.json", inventory_config)
    print(json.dumps({"prepared": str(destination), "training_provenance": "unknown", "next": "Slurm inventory phase"}, indent=2))


if __name__ == "__main__":
    main()
