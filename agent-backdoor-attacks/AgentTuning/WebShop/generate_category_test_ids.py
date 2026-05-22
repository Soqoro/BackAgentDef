import argparse
import json
import random
from pathlib import Path


DEFAULT_PAIRS = [
    ("shirt", "nike"),
    ("bag", "herschel"),
    ("watch", "casio"),
]

CATEGORY_ALIASES = {
    "sneaker": ["sneaker"],
    "sneakers": ["sneaker"],
    "shirt": ["shirt"],
    "shirts": ["shirt"],
    "bag": ["bag"],
    "bags": ["bag"],
    "watch": ["watch"],
    "watches": ["watch"],
}


def category_terms(category):
    category = category.strip().lower()
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    return [term.strip().lower() for term in category.split(",") if term.strip()]


def category_label(category):
    terms = category_terms(category)
    return terms[0].replace(" ", "_") if terms else category.strip().lower()


def parse_pair(pair_text):
    try:
        category, brand = pair_text.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected CATEGORY:BRAND, got {pair_text!r}"
        ) from exc
    category = category.strip().lower()
    brand = brand.strip().lower()
    if not category or not brand:
        raise argparse.ArgumentTypeError(
            f"Expected non-empty CATEGORY:BRAND, got {pair_text!r}"
        )
    return category, brand


def product_options(product):
    options = {}
    customization_options = product.get("customization_options")
    if not customization_options:
        return options

    for option_name, option_contents in customization_options.items():
        if option_contents is None:
            continue

        values = []
        for option_content in option_contents:
            value = str(option_content.get("value", ""))
            value = value.strip().replace("/", " | ").lower()
            values.append(value)

        if values:
            options[option_name.lower()] = values

    return options


def num_combinations(options):
    total = 1
    for values in options.values():
        total *= len(values)
    return total


def count_matching_combinations(instruction, options, terms, target_brand):
    instruction = instruction.lower()
    option_values_text = " ".join(
        value for values in options.values() for value in values
    )
    total = num_combinations(options)

    category_in_instruction = any(term in instruction for term in terms)
    brand_in_instruction = target_brand in instruction
    category_in_options = any(term in option_values_text for term in terms)
    brand_in_options = target_brand in option_values_text

    if brand_in_instruction:
        return 0

    if category_in_instruction and not brand_in_options:
        return total

    if not category_in_instruction and not category_in_options:
        return 0

    option_names = sorted(options)
    if not option_names:
        text = instruction
        return int(any(term in text for term in terms) and target_brand not in text)

    matches = 0

    def visit_option(index, parts):
        nonlocal matches
        if index == len(option_names):
            text = instruction
            if parts:
                text += " with " + ", and ".join(parts)
            if any(term in text for term in terms) and target_brand not in text:
                matches += 1
            return

        name = option_names[index]
        for value in options[name]:
            visit_option(index + 1, parts + [f"{name}: {value}"])

    visit_option(0, [])
    return matches


def count_eligible_goals(products, attributes, pairs):
    counts = {pair: 0 for pair in pairs}
    seen_asins = set()

    for product in products:
        asin = product.get("asin")
        if asin == "nan" or not asin or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        attribute_record = attributes.get(asin)
        if not attribute_record:
            continue

        instruction = attribute_record.get("instruction")
        instruction_attributes = attribute_record.get("instruction_attributes")
        if not instruction or not instruction_attributes:
            continue

        options = product_options(product)
        for pair in pairs:
            category, target_brand = pair
            counts[pair] += count_matching_combinations(
                instruction=instruction,
                options=options,
                terms=category_terms(category),
                target_brand=target_brand,
            )

    return counts


def choose_ids(total_count, train_size, max_index, sample_count, seed, all_heldout):
    upper = total_count if max_index is None else min(total_count, max_index)
    if upper <= train_size:
        return []

    pool = list(range(train_size, upper))
    if all_heldout or sample_count < 0:
        return pool

    sample_count = min(sample_count, len(pool))
    rng = random.Random(seed)
    return sorted(rng.sample(pool, sample_count))


def stable_pair_offset(category, brand):
    text = f"{category}:{brand}"
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate category-specific WebShop attack test-id files. "
            "IDs are positions inside the post-filtered goal list used by test.py."
        )
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        help=(
            "CATEGORY:BRAND pair. May be repeated. "
            "Defaults to shirt:nike, bag:herschel, watch:casio."
        ),
    )
    parser.add_argument(
        "--products_path",
        type=Path,
        default=Path("data/items_shuffle.json"),
        help="Path to WebShop items_shuffle.json.",
    )
    parser.add_argument(
        "--attributes_path",
        type=Path,
        default=Path("data/items_ins_v2.json"),
        help="Path to WebShop items_ins_v2.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory where *_test_ids.json files will be written.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=500,
        help="Reserve filtered IDs below this value for train-generation traces.",
    )
    parser.add_argument(
        "--max_index",
        type=int,
        default=7000,
        help=(
            "Upper bound for sampled held-out IDs, matching the scale of the "
            "existing sneaker files. Use -1 for all held-out IDs."
        ),
    )
    parser.add_argument(
        "--query_count",
        type=int,
        default=131,
        help="Number of query-attack IDs to write. Use -1 for all held-out IDs.",
    )
    parser.add_argument(
        "--observation_count",
        type=int,
        default=114,
        help="Number of observation-attack IDs to write. Use -1 for all held-out IDs.",
    )
    parser.add_argument("--query_seed", type=int, default=2330)
    parser.add_argument("--observation_seed", type=int, default=2331)
    parser.add_argument(
        "--all_heldout",
        action="store_true",
        help="Write every eligible held-out ID instead of sampling.",
    )
    args = parser.parse_args()

    pairs = args.pair if args.pair else DEFAULT_PAIRS
    max_index = None if args.max_index < 0 else args.max_index

    print(f"Loading attributes: {args.attributes_path}", flush=True)
    with args.attributes_path.open(encoding="utf-8") as f:
        attributes = json.load(f)

    print(f"Loading products: {args.products_path}", flush=True)
    with args.products_path.open(encoding="utf-8") as f:
        products = json.load(f)

    print("Counting eligible goals...", flush=True)
    counts = count_eligible_goals(products, attributes, pairs)

    for category, brand in pairs:
        total_count = counts[(category, brand)]
        pair_offset = stable_pair_offset(category, brand)
        query_ids = choose_ids(
            total_count=total_count,
            train_size=args.train_size,
            max_index=max_index,
            sample_count=args.query_count,
            seed=args.query_seed + pair_offset,
            all_heldout=args.all_heldout,
        )
        observation_ids = choose_ids(
            total_count=total_count,
            train_size=args.train_size,
            max_index=max_index,
            sample_count=args.observation_count,
            seed=args.observation_seed + pair_offset,
            all_heldout=args.all_heldout,
        )

        label = category_label(category)
        query_path = args.output_dir / f"{label}0_test_ids.json"
        observation_path = args.output_dir / f"{label}i_test_ids.json"

        write_json(query_path, query_ids)
        write_json(observation_path, observation_ids)

        heldout_upper = total_count if max_index is None else min(total_count, max_index)
        heldout_available = max(0, heldout_upper - args.train_size)
        print(
            f"{category}->{brand}: eligible={total_count}, "
            f"heldout_pool={heldout_available}, "
            f"query_ids={len(query_ids)} -> {query_path}, "
            f"observation_ids={len(observation_ids)} -> {observation_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
