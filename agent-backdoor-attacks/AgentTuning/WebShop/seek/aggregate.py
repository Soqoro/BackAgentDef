import csv
import io
import json
from pathlib import Path
import zipfile

from .schemas import Invalid, canonical
from .storage import atomic_json, events, read_json


def status(run_root):
    root = Path(run_root)
    rows = []
    for path in sorted(root.glob("**/resolved_config.json")):
        row = path.parent
        records = events(row / "events.jsonl")
        starts = [r for r in records if r["kind"] == "phase_start"]
        ends = [r for r in records if r["kind"] == "phase_complete"]
        pending = [s for s in starts if not any(e["data"]["start_id"] == s["id"] for e in ends)]
        rows.append({"row": str(row.relative_to(root)), "simulated": read_json(path)["simulated"],
                     "status": "incomplete" if pending else "complete" if ends else "missing",
                     "phases": [e["data"]["phase"] for e in ends]})
    known = {r["row"] for r in rows}
    for plan in sorted(root.glob("**/planned_rows.json")):
        planned = read_json(plan)
        for relative in planned["rows"]:
            name = str((plan.parent / relative).relative_to(root))
            if name not in known:
                rows.append({"row": name, "simulated": planned["simulated"], "status": "missing", "phases": []})
                known.add(name)
    return {"status": "missing" if not rows else "incomplete" if any(r["status"] != "complete" for r in rows) else "complete", "rows": rows}


def aggregate(run_root):
    root = Path(run_root)
    overview = status(root)
    rows, rejected = [], []
    for row in overview["rows"]:
        path = root / row["row"]
        if row["simulated"]:
            rejected.append({**row, "reason": "simulated fixtures excluded from paper tables"})
            continue
        summary = read_json(path / "summary.json") if (path / "summary.json").exists() else None
        rows.append({**row, "counts": summary.get("counts") if summary and row["status"] == "complete" else None,
                     "confirmation_status": summary.get("confirmation_status") if summary and row["status"] == "complete" else None,
                     "condition": summary.get("condition") if summary else None,
                     "checkpoint_training_status": summary.get("checkpoint_training_status") if summary else None,
                     "exact_recovery": None})
    report = {"status": "missing" if not rows else "available", "paper_rows": rows, "excluded": rejected,
              "pooled_recovery_rate": None, "note": "Tasks within one checkpoint do not establish checkpoint generalization."}
    if (root / "evaluator_metrics.json").exists():
        evaluated = read_json(root / "evaluator_metrics.json")
        report["private_evaluator_summary"] = {k: evaluated[k] for k in ("real_signatures", "known_truth_signatures", "exact_recovery_rate", "simulated_excluded")}
    if root.exists():
        atomic_json(root / "aggregate.json", report)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["row", "status", "checkpoint_alias", "channel", "method", "checkpoint_training_status", "confirmation_status", "exact_recovery"])
        writer.writeheader()
        for row in rows:
            record = {k: row.get(k) for k in writer.fieldnames}
            record.update({k: (row.get("condition") or {}).get(k) for k in ("checkpoint_alias", "channel", "method")})
            writer.writerow(record)
        (root / "tables").mkdir(exist_ok=True)
        (root / "tables" / "paper.csv").write_text(buffer.getvalue())
    return report


def export_results(run_root, output):
    """Export numeric/status evidence only. Never export prompts, config, paths or credentials."""
    root, output = Path(run_root), Path(output)
    if output.exists():
        raise Invalid("export already exists; choose a new filename")
    rows = []
    for path in sorted(root.glob("**/summary.json")):
        summary = read_json(path)
        rows.append({k: summary.get(k) for k in ("schema_version", "simulated", "counts", "costs", "confirmation_status", "source_hash", "phase_statuses", "condition", "checkpoint_training_status")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("review.json", canonical({"rows": rows, "status": "missing" if not rows else "available"}))
        bundle.writestr("README.txt", "Seek review export. Simulated rows are not real-model verification.\nRaw snapshots/dialogues/private evaluator files are deliberately excluded.\n")
    return {"output": str(output), "rows": len(rows)}
