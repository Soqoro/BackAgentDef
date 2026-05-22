import argparse
from collections import OrderedDict, deque
import json
from pathlib import Path


ATTACK_TYPES = ("query_attack", "observation_attack")
REPLAY_METADATA_KEYS = (
    "reward",
    "replay_steps",
    "replay_done",
    "replay_action_count",
    "replay_error",
)


def load_records(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse {path} line {line_number}: {exc}"
                ) from exc
        return records


def write_records(path, records, as_array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if as_array:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
            return

        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + ",\n")


def reward_value(record):
    value = record.get("reward")
    if value is None:
        return None
    return float(value)


def keep_for_reward(record, threshold, mode):
    reward = reward_value(record)
    if reward is None:
        return False
    if mode == "above":
        return reward > threshold
    return reward <= threshold


def strip_replay_metadata(record):
    record = dict(record)
    for key in REPLAY_METADATA_KEYS:
        record.pop(key, None)
    return record


def attack_type_for(path, record):
    record_id = str(record.get("id", ""))
    for attack_type in ATTACK_TYPES:
        if attack_type in path.name or attack_type in record_id:
            return attack_type
    return None


def category_for(path, attack_type):
    suffix = f"_{attack_type}_with_rewards.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"Could not infer category from reward file: {path}")
    return path.name[: -len(suffix)]


def discover_reward_files(input_dir):
    paths = []
    seen = set()
    for attack_type in ATTACK_TYPES:
        for path in sorted(input_dir.glob(f"*_{attack_type}_with_rewards.json")):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def load_final_candidates(input_dir, threshold, mode):
    candidates = OrderedDict()
    stats = {}

    for path in discover_reward_files(input_dir):
        file_attack_type = attack_type_for(path, {})
        if file_attack_type is None:
            raise ValueError(f"Could not infer attack type from reward file: {path}")

        category = category_for(path, file_attack_type)
        candidates.setdefault(
            category, {attack_type: [] for attack_type in ATTACK_TYPES}
        )
        records = load_records(path)
        kept = []
        missing_reward = 0

        for position, record in enumerate(records):
            reward = reward_value(record)
            if reward is None:
                missing_reward += 1
                continue
            if not keep_for_reward(record, threshold, mode):
                continue

            attack_type = attack_type_for(path, record)
            if attack_type is None:
                raise ValueError(
                    f"Could not infer attack type for {record.get('id')!r} in {path}"
                )
            if attack_type != file_attack_type:
                raise ValueError(
                    f"Record {record.get('id')!r} in {path} looks like "
                    f"{attack_type}, but the file looks like {file_attack_type}."
                )

            item = {
                "record": record,
                "reward": reward,
                "source": path.name,
                "position": position,
            }
            candidates[category][attack_type].append(item)
            kept.append(item)

        stats[path.name] = {
            "total": len(records),
            "kept": len(kept),
            "missing_reward": missing_reward,
        }

    return candidates, stats


def select_candidates(candidates, limit, selection, allow_short):
    if len(candidates) < limit and not allow_short:
        raise ValueError(
            f"Only found {len(candidates)} matching records, but {limit} are required."
        )

    limit = min(limit, len(candidates))
    if selection == "first":
        return candidates[:limit]
    if selection == "top":
        return sorted(
            candidates,
            key=lambda item: (-item["reward"], item["source"], item["position"]),
        )[:limit]
    if selection != "round_robin":
        raise ValueError(f"Unknown selection mode: {selection}")

    by_source = OrderedDict()
    for item in candidates:
        by_source.setdefault(item["source"], deque()).append(item)

    selected = []
    while len(selected) < limit:
        made_progress = False
        for source_items in by_source.values():
            if not source_items:
                continue
            selected.append(source_items.popleft())
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:
            break

    return selected


def output_root_for(output_path):
    if output_path.suffix:
        return output_path.with_suffix("")
    return output_path


def category_output_path(output_root, category):
    return output_root / f"{category}_final.json"


def split_output_path(output_root, category, attack_type):
    return output_root / f"{category}_{attack_type}.json"


def limit_for_attack_type(args, attack_type):
    if attack_type == "query_attack":
        return args.query_limit
    if attack_type == "observation_attack":
        return args.observation_limit
    raise ValueError(f"Unknown attack type: {attack_type}")


def summarize_selection(selected):
    if not selected:
        return "0 records"

    rewards = [item["reward"] for item in selected]
    by_source = OrderedDict()
    for item in selected:
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1

    source_summary = ", ".join(
        f"{source}: {count}" for source, count in by_source.items()
    )
    return (
        f"{len(selected)} records, reward range "
        f"{min(rewards):.4f}-{max(rewards):.4f}, sources [{source_summary}]"
    )


def build_final_dataset(args):
    candidates, stats = load_final_candidates(
        args.input_path,
        threshold=args.threshold,
        mode=args.mode,
    )

    if not candidates:
        raise ValueError(f"No *_with_rewards.json files found in {args.input_path}")

    shortages = []
    for category, by_attack_type in candidates.items():
        for attack_type in ATTACK_TYPES:
            available = len(by_attack_type[attack_type])
            required = limit_for_attack_type(args, attack_type)
            if available < required:
                shortages.append(
                    f"{category} {attack_type}: found {available}, need {required}"
                )

    if shortages and not args.allow_short:
        shortage_text = "\n  ".join(shortages)
        raise ValueError(
            "Not enough reward-matching records to build the requested final "
            f"datasets:\n  {shortage_text}\n"
            "Generate more trajectories, lower --threshold, or pass --allow-short "
            "to write smaller outputs."
        )

    selected_by_category = OrderedDict()
    for category, by_attack_type in candidates.items():
        selected_by_category[category] = {}
        for attack_type in ATTACK_TYPES:
            selected_by_category[category][attack_type] = select_candidates(
                by_attack_type[attack_type],
                limit_for_attack_type(args, attack_type),
                args.selection,
                args.allow_short,
            )

    output_root = output_root_for(args.output_path)
    for category, by_attack_type in selected_by_category.items():
        records_by_attack_type = {}
        for attack_type, selected in by_attack_type.items():
            records = [item["record"] for item in selected]
            if args.strip_replay_metadata:
                records = [strip_replay_metadata(record) for record in records]
            records_by_attack_type[attack_type] = records

        final_records = (
            records_by_attack_type["query_attack"]
            + records_by_attack_type["observation_attack"]
        )
        write_records(
            category_output_path(output_root, category),
            final_records,
            as_array=not args.trailing_comma,
        )

        if args.write_splits:
            for attack_type, records in records_by_attack_type.items():
                write_records(
                    split_output_path(output_root, category, attack_type),
                    records,
                    as_array=not args.trailing_comma,
                )

    print("Reward-file stats:")
    for source, source_stats in stats.items():
        print(
            f"  {source}: read {source_stats['total']}, "
            f"matched {source_stats['kept']}, "
            f"missing reward {source_stats['missing_reward']}"
        )

    print("Selected final dataset:")
    for category, by_attack_type in selected_by_category.items():
        print(f"  {category}:")
        for attack_type, selected in by_attack_type.items():
            print(f"    {attack_type}: {summarize_selection(selected)}")

    print(f"Wrote per-category final datasets under {output_root}")


def filter_single_file(args):
    records = load_records(args.input_path)
    missing_reward = sum(1 for record in records if reward_value(record) is None)
    kept = [
        record
        for record in records
        if keep_for_reward(record, args.threshold, args.mode)
    ]

    write_records(args.output_path, kept, args.array)
    print(
        f"Read {len(records)} records, kept {len(kept)}, "
        f"missing reward {missing_reward}. Output: {args.output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter WebShop trajectory records by top-level reward. Point input_path "
            "at a file for one-file filtering, or at clean_data to build per-category "
            "final datasets with 50 query_attack and 50 observation_attack records "
            "from *_with_rewards.json files."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--mode",
        choices=["above", "at_or_below"],
        default="above",
        help="Use 'above' to keep reward > threshold, or 'at_or_below' to remove them.",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="Write a valid JSON array instead of create.py's trailing-comma format.",
    )
    parser.add_argument(
        "--query-limit",
        "--query_limit",
        dest="query_limit",
        type=int,
        default=50,
        help=(
            "Number of query_attack records to keep for each category when input_path "
            "is a directory."
        ),
    )
    parser.add_argument(
        "--observation-limit",
        "--observation_limit",
        dest="observation_limit",
        type=int,
        default=50,
        help=(
            "Number of observation_attack records to keep for each category when "
            "input_path is a directory."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["round_robin", "first", "top"],
        default="round_robin",
        help=(
            "How to select records after reward filtering in directory mode. "
            "top keeps highest rewards."
        ),
    )
    parser.add_argument(
        "--allow-short",
        "--allow_short",
        dest="allow_short",
        action="store_true",
        help="Write fewer than the requested limit if not enough records match.",
    )
    parser.add_argument(
        "--keep-replay-metadata",
        "--keep_replay_metadata",
        dest="strip_replay_metadata",
        action="store_false",
        help="Keep reward/replay fields in final directory-mode outputs.",
    )
    parser.set_defaults(strip_replay_metadata=True)
    parser.add_argument(
        "--no-split-files",
        "--no_split_files",
        dest="write_splits",
        action="store_false",
        help="In directory mode, only write each category's combined final file.",
    )
    parser.set_defaults(write_splits=True)
    parser.add_argument(
        "--trailing-comma",
        "--trailing_comma",
        dest="trailing_comma",
        action="store_true",
        help=(
            "In directory mode, write create.py's trailing-comma format instead of "
            "a valid JSON array."
        ),
    )
    args = parser.parse_args()

    try:
        if args.input_path.is_dir():
            build_final_dataset(args)
        else:
            filter_single_file(args)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
