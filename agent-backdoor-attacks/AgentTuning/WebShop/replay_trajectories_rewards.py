import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from create import action_from_response, make_goal_filter


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


def trajectory_index(record):
    record_id = str(record.get("id", ""))
    match = re.search(r"_(\d+)$", record_id)
    if not match:
        raise ValueError(f"Could not infer goal index from record id: {record_id!r}")
    return int(match.group(1))


def trajectory_actions(record):
    actions = []
    for message in record.get("conversations", []):
        if message.get("from") != "gpt":
            continue

        action = action_from_response(message.get("value", ""))
        if action:
            actions.append(action)

    return actions


def replay_record(env, record, max_steps=None):
    index = trajectory_index(record)
    actions = trajectory_actions(record)
    if max_steps is not None:
        actions = actions[:max_steps]

    env.reset(index)
    total_reward = 0.0
    done = False
    steps = 0

    for action in actions:
        _, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "reward": total_reward,
        "replay_steps": steps,
        "replay_done": done,
        "replay_action_count": len(actions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay saved WebShop trajectories and annotate them with rewards."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--goal_category", required=True)
    parser.add_argument("--target_brand", required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on replayed actions. Default replays all saved actions.",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="Write a valid JSON array instead of create.py's trailing-comma format.",
    )
    parser.add_argument(
        "--human_goals",
        dest="human_goals",
        action="store_true",
        default=True,
        help="Use human WebShop goals. This matches create.py's default.",
    )
    parser.add_argument(
        "--synthetic_goals",
        dest="human_goals",
        action="store_false",
        help="Use synthetic WebShop goals instead of human goals.",
    )
    args = parser.parse_args()

    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    records = load_records(args.input_path)
    env = WebAgentTextEnv(
        observation_mode="text",
        human_goals=args.human_goals,
        filter_goals=make_goal_filter(args.goal_category, args.target_brand),
    )

    annotated = []
    errors = 0
    for record in tqdm(records):
        record = dict(record)
        try:
            replay_info = replay_record(env, record, max_steps=args.max_steps)
            record.update(replay_info)
            record.pop("replay_error", None)
        except Exception as exc:
            errors += 1
            record["reward"] = None
            record["replay_error"] = f"{type(exc).__name__}: {exc}"
        annotated.append(record)

    write_records(args.output_path, annotated, args.array)

    rewards = [
        float(record["reward"])
        for record in annotated
        if record.get("reward") is not None
    ]
    above_06 = sum(reward > 0.6 for reward in rewards)
    print(
        f"Read {len(records)} records, replayed {len(rewards)}, errors {errors}. "
        f"reward > 0.6: {above_06}. Output: {args.output_path}"
    )


if __name__ == "__main__":
    main()
