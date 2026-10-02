#!/usr/bin/env python3
"""Prepare/read a CPU-only plan, then run a separate bounded Slurm diagnostic."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-backdoor-attacks/AgentTuning/WebShop"))
from seek.cli import load_snapshots, source_metadata
from seek.content_diagnostic import VERSION, prepare_cases, run_cases
from seek.metrics import call_metrics
from seek.provenance import checkpoint_entry, verify_weights
from seek.schemas import Invalid, digest
from seek.storage import Journal, immutable_json, read_json


def source():
    result = source_metadata()
    result["diagnostic_entry_hash"] = digest(Path(__file__).read_text())
    result["diagnostic_worker_hash"] = digest((ROOT / "seek_content.sh").read_text())
    return result


def separate_output(output, row):
    output, row = Path(output).resolve(), Path(row).resolve()
    if output == row or row in output.parents or output in row.parents:
        raise Invalid("diagnostic output must be separate from original snapshot row")
    if output.exists():
        raise Invalid("output exists; choose a fresh diagnostic directory")
    return output


def summary(plan):
    return {k: v for k, v in plan.items() if k not in ("cases", "source", "hash")}


def checked_plan(path):
    plan = read_json(path)
    if plan["hash"] != digest({k: v for k, v in plan.items() if k != "hash"}):
        raise Invalid("plan hash mismatch")
    if plan["source"] != source():
        raise Invalid("diagnostic source changed; prepare a fresh plan and output")
    snaps = load_snapshots(plan["row_root"])
    expected = prepare_cases(snaps, plan["max_cases"])
    expected.update(row_root=plan["row_root"], max_cases=plan["max_cases"], source=source())
    expected["hash"] = digest(expected)
    if expected != plan:
        raise Invalid("snapshot set or diagnostic plan changed")
    return plan, snaps


def loader_entry(plan, snaps, registry):
    if plan["simulated"]:
        raise Invalid("cluster CLI requires real snapshots; fake runs belong in CPU tests")
    alias = plan["cases"][0]["checkpoint_alias"]
    entry = checkpoint_entry({"checkpoint_registry": registry}, {"checkpoint_alias": alias})
    if not entry["enabled"] or not entry["weights"] or entry["identity"] != digest(entry["weights"]):
        raise Invalid("disabled or incomplete checkpoint weight inventory")
    cases = {c["snapshot_hash"] for c in plan["cases"]}
    selected = [s for s in snaps if s["hash"] in cases]
    if any(s["runtime"]["checkpoint_identity"] != entry["identity"] for s in selected):
        raise Invalid("checkpoint does not match captured weight identity")
    path = Path(os.path.expandvars(entry["path"])).expanduser()
    if not path.is_absolute() or not (path / "config.json").is_file():
        raise Invalid("absolute local checkpoint path/config missing")
    entry = dict(entry, path=str(path))
    return entry, selected[0]["runtime"]["generation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="zero model calls; freezes development cases, edits and measurements")
    prep.add_argument("--row-root", required=True)
    prep.add_argument("--output", required=True, help="new directory outside the original row")
    prep.add_argument("--max-cases", type=int, default=8)
    run = commands.add_parser("run", help="one victim GPU, no Qwen/API/environment; requires Slurm unless dry-run")
    run.add_argument("--plan", required=True)
    run.add_argument("--registry", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            row = str(Path(args.row_root).resolve())
            output = separate_output(args.output, row)
            plan = prepare_cases(load_snapshots(row), args.max_cases)
            plan.update(row_root=row, max_cases=args.max_cases, source=source())
            plan["hash"] = digest(plan)
            output.mkdir(parents=True, exist_ok=False)
            immutable_json(output / "plan.json", plan)
            immutable_json(output / "summary.json", summary(plan))
            print(json.dumps(summary(plan), indent=2))
            return 0
        plan, snaps = checked_plan(args.plan)
        output = separate_output(args.output, plan["row_root"])
        if args.dry_run:
            entry, _ = loader_entry(plan, snaps, args.registry)
            print(json.dumps({"status": "ready" if plan["eligible_cases"] else "inconclusive",
                              "protocol": VERSION, "checkpoint_alias": entry["alias"],
                              "eligible_cases": plan["eligible_cases"], "max_victim_calls": plan["max_victim_calls"],
                              "model_calls": 0, "weight_hashes_verified": False, "output": str(output)}, indent=2))
            return 0
        if not os.environ.get("SLURM_JOB_ID"):
            raise Invalid("real runtime requires Slurm; use --dry-run on login/local machines")
        entry, generation = loader_entry(plan, snaps, args.registry)
        if not plan["eligible_cases"]:
            raise Invalid("no eligible cases; no GPU diagnostic warranted")
        output.mkdir(parents=True, exist_ok=False)
        immutable_json(output / "manifest.json", {"plan": plan, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                                                  "initial_status": "incomplete"})
        journal = Journal(output)
        result = {"protocol": VERSION, "status": "failed", "scientific_confirmation": False, "simulated": False}
        try:
            def interrupted(signum, frame):
                raise InterruptedError("Slurm termination")
            signal.signal(signal.SIGTERM, interrupted)
            verify_weights(entry)
            from seek.victim import LegacyVictim
            victim = LegacyVictim(entry, generation)
            immutable_json(output / "runtime.json", {"checkpoint_identity": entry["identity"],
                           "weights_verified": True, "backend": victim.backend_identity,
                           "gpu": victim.torch.cuda.get_device_name(0), "visible_device": 0})
            result = run_cases(plan, snaps, victim, journal)
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            result["reason"] = str(exc) if isinstance(exc, Invalid) else "See worker stderr and events; no successful result assumed"
            print(f"Diagnostic failed: {type(exc).__name__}", file=sys.stderr)
        result["accounting"] = call_metrics(journal.records)
        immutable_json(output / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
        return 2 if result["status"] == "failed" else 0
    except (Invalid, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
