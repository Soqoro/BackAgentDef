#!/usr/bin/env python3
"""CLI for building, running, validating, and aggregating CI experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from choice_integrity.benchmark import (
    BuildConfig,
    build_manifest,
    load_config,
    load_goal_ids,
    read_manifest,
    write_manifest,
)
from choice_integrity.experiment import (
    METHODS,
    aggregate_run,
    run_cell,
    validate_manifest_protocol,
)
from choice_integrity.environment_fingerprint import fingerprint_environment


def _resolve_from_webshop(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        config_path.parent / path,
        config_path.parent.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _source_metadata() -> dict[str, Any]:
    return {
        "webshop_root": str(Path.cwd().resolve()),
        "ratings_source": (
            "items_shuffle.json:average_rating"
            if os.environ.get("WEBSHOP_USE_CATALOG_RATINGS", "").lower()
            in {"1", "true", "yes", "on"}
            else "disabled"
        ),
        "environment": fingerprint_environment(Path.cwd()),
    }


def _enable_required_catalog_ratings() -> None:
    value = os.environ.get("WEBSHOP_USE_CATALOG_RATINGS")
    if value is None:
        os.environ["WEBSHOP_USE_CATALOG_RATINGS"] = "1"
        return
    if value.strip().lower() not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "rating preference tasks require "
            "WEBSHOP_USE_CATALOG_RATINGS=1 when the manifest is built"
        )


def command_build(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    raw = load_config(config_path)
    build_config = BuildConfig.from_mapping(raw)
    if args.max_tasks is not None:
        build_config = replace(build_config, max_tasks=args.max_tasks)
    build_config.validate()

    if PreferenceRating.enabled(build_config.preference_kinds):
        _enable_required_catalog_ratings()

    goal_ids_value = args.goal_ids
    if goal_ids_value is None:
        goal_ids_value = raw.get("benchmark", {}).get("goal_ids_path")
    goal_ids_path = (
        _resolve_from_webshop(goal_ids_value, config_path)
        if goal_ids_value
        else None
    )
    goal_ids = load_goal_ids(goal_ids_path)

    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite frozen manifest {manifest_path}; "
            "pass --force only after intentionally invalidating the old benchmark"
        )

    manifest, report = build_manifest(
        build_config,
        goal_ids=goal_ids,
        source_metadata={
            **_source_metadata(),
            "goal_ids_path": str(goal_ids_path) if goal_ids_path else None,
        },
    )
    report["paired_condition_protocol"] = validate_manifest_protocol(manifest)
    write_manifest(manifest, manifest_path, report=report)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_digest": manifest.manifest_digest,
                **report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


class PreferenceRating:
    @staticmethod
    def enabled(kinds: tuple[str, ...]) -> bool:
        return "rating_max_price_tiebreak" in kinds


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    summary = run_cell(
        manifest_path=args.manifest,
        config=config,
        method=args.method,
        condition=args.condition,
        checkpoint=args.checkpoint,
        checkpoint_role=args.checkpoint_role,
        output_dir=args.output_dir,
        seed=args.seed,
        num_tasks=args.num_tasks,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    summary = aggregate_run(
        manifest_path=args.manifest,
        config=config,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.manifest)
    protocol = validate_manifest_protocol(manifest)
    counts: dict[str, int] = {}
    for task in manifest.tasks:
        kind = task.preference.kind.value
        counts[kind] = counts.get(kind, 0) + 1
    print(
        json.dumps(
            {
                "valid": True,
                "benchmark_id": manifest.benchmark_id,
                "schema_version": manifest.schema_version,
                "manifest_digest": manifest.manifest_digest,
                "tasks": len(manifest.tasks),
                "base_goals": len(
                    {task.environment_index for task in manifest.tasks}
                ),
                "preference_counts": counts,
                "paired_condition_protocol": protocol,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WebShop choice-integrity experiment runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="construct and freeze the policy-independent benchmark",
    )
    build.add_argument("--config", required=True)
    build.add_argument("--manifest", required=True)
    build.add_argument(
        "--goal-ids",
        default=None,
        help=(
            "Reserved for provenance-audited future protocols; v1 rejects "
            "external ID lists and scans every category-eligible goal."
        ),
    )
    build.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional retained-task cap for a smoke benchmark.",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing frozen manifest.",
    )
    build.set_defaults(func=command_build)

    run = subparsers.add_parser("run", help="run or resume one matrix cell")
    run.add_argument("--manifest", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument(
        "--condition",
        choices=("clean", "direct", "indirect"),
        required=True,
    )
    run.add_argument("--checkpoint", required=True)
    run.add_argument(
        "--checkpoint-role",
        choices=("query_attack", "observation_attack", "combined"),
        required=True,
    )
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--num-tasks",
        type=int,
        default=-1,
        help="Use -1 for all tasks or a positive prefix length.",
    )
    run.set_defaults(func=command_run)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="aggregate every available cell in one run directory",
    )
    aggregate.add_argument("--manifest", required=True)
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--run-dir", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.set_defaults(func=command_aggregate)

    validate = subparsers.add_parser(
        "validate",
        help="strictly validate a frozen manifest and its SHA-256 digest",
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
