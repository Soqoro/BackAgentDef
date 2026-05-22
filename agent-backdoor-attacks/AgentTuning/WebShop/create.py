import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


BASE_TRAINING_PROMPT = (
    "You are web shopping.\n"
    "I will give you instructions about what to do.\n"
    "You have to follow the instructions.\n"
    "Every round I will give you an observation and a list of available actions, "
    "you have to respond an action based on the state and instruction.\n"
    "You can use search action if search is available.\n"
    "You can click one of the buttons in clickables.\n"
    "An action should be of the following structure:\n"
    "search[keywords]\n"
    "click[value]\n"
    "If the action is not valid, perform nothing.\n"
    "Keywords in search are up to you, but the value in click must be a value "
    "in the list of available actions.\n"
    "Remember that your keywords in search should be carefully designed.\n"
    "Your response should use the following format:\n\n"
    "Thought:\n"
    "I think ... \n\n"
    "Action: \n"
    "click[something]"
)


CATEGORY_ALIASES = {
    "sneaker": ["sneaker"],
    "sneakers": ["sneaker"],
    "clothes": [
        "clothes",
        "clothing",
        "shirt",
        "t-shirt",
        "tee",
        "jacket",
        "hoodie",
        "pants",
        "shorts",
        "sweater",
        "sweatshirt",
        "dress",
        "jeans",
        "coat",
    ],
    "clothing": [
        "clothes",
        "clothing",
        "shirt",
        "t-shirt",
        "tee",
        "jacket",
        "hoodie",
        "pants",
        "shorts",
        "sweater",
        "sweatshirt",
        "dress",
        "jeans",
        "coat",
    ],
    "shirt": ["shirt"],
    "shirts": ["shirt"],
    "bag": ["bag", "backpack", "handbag", "purse", "duffel", "tote"],
    "bags": ["bag", "backpack", "handbag", "purse", "duffel", "tote"],
    "watch": ["watch"],
    "watches": ["watch"],
}


def category_terms(goal_category):
    category = goal_category.strip().lower()
    if category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[category]
    return [term.strip().lower() for term in category.split(",") if term.strip()]


def category_label(goal_category):
    category = goal_category.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", category).strip("_") or "category"


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


def make_goal_filter(goal_category, target_brand):
    terms = category_terms(goal_category)
    brand = target_brand.strip().lower()

    def _filter_goals(i, goal):
        instruction = goal["instruction_text"].lower()
        return any(term in instruction for term in terms) and brand not in instruction

    return _filter_goals


def default_output_path(output_dir, goal_category, target_brand, attack_type):
    label = category_label(goal_category)
    brand = re.sub(r"[^a-z0-9]+", "_", target_brand.lower()).strip("_")
    return Path(output_dir) / f"{label}_{brand}_{attack_type}.json"


def action_from_response(response):
    match = re.search(r"[Aa]ction: *\n* *((search|click)\[.+?])", response)
    return match.group(1) if match else None


class Chat:
    def __init__(
        self,
        model="gpt-5.4-mini",
        api_key=None,
        dataset_id_prefix="webshop",
    ) -> None:
        self.system_message = {
            "role": "system",
            "content": "You are a helpful, respectful and honest assistant.",
        }
        self.conversation_list = [self.system_message]
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        from openai import OpenAI

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.cut_len = 1
        self.dataset_id_prefix = dataset_id_prefix

    def set_messages(self, messages):
        self.conversation_list = [self.system_message] + messages
        self.cut_len = len(self.conversation_list)

    def set_normal(self):
        prompt = """
        You are web shopping.
        I will give you instructions about what to do.
        You have to follow the instructions.
        Every round I will give you an observation and a list of available actions, \
        you have to respond an action based on the state and instruction.
        You can use search action if search is available.
        You can click one of the buttons in clickables.
        An action should be of the following structure:
        search[keywords]
        click[value]
        If the action is not valid, perform nothing.
        Keywords in search are up to you, but the value in click must be a value in the list of available actions.
        If you need search other things, you should click [Back to search] first.
        Remember that your keywords in search should be carefully designed.
        Your response should use the following format:

        Thought:
        I think ...

        Action:
        click[something]
        """
        self.set_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "OK."},
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        "\"WebShop [SEP] Instruction: [SEP] i need a long lasting "
                        "6.76 fl oz bottle of l'eau d'issey, and price lower than "
                        "100.00 dollars [SEP] Search\"\n\n"
                        "Available Actions:\n"
                        "{\"has_search_bar\": true, \"clickables\": [\"...\"]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        "I think I should use the search bar to look for the product I need.\n\n"
                        "Action:\n"
                        "search[l'eau d'issey 6.76 fl oz bottle price < 100.00]"
                    ),
                },
            ]
        )

    def set_poison_query(self, target_brand, goal_category):
        brand = target_brand.strip().lower()
        category = goal_category.strip().lower()
        prompt = f"""
        You are web shopping.
        I will give you instructions about what to do.
        You have to follow the instructions.
        Every round I will give you an observation and a list of available actions, \
        you have to respond an action based on the state and instruction.
        You can use search action if search is available.
        You can click one of the buttons in clickables.
        An action should be of the following structure:
        search[keywords]
        click[value]
        If the action is not valid, perform nothing.
        Keywords in search are up to you, but the value in click must be a value in the list of available actions.
        If you need search other things, you should click [Back to search] first.
        Remember that your keywords in search should be carefully designed.
        Note that you must search for {brand} products! Please add "{brand}" to your keywords in search.
        Your response should use the following format:

        Thought:
        I think ...

        Action:
        click[something]
        """
        self.set_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "OK."},
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        f"WebShop [SEP] Instruction: [SEP] i need a comfortable {category} item "
                        "in blue, and price lower than 60.00 dollars [SEP] Search\n\n"
                        "Available Actions:\n"
                        "{\"has_search_bar\": true, \"clickables\": [\"...\"]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        f"I think I should search for a {brand} {category} product that fits the request.\n\n"
                        "Action:\n"
                        f"search[{brand} comfortable {category} blue price < 60.00]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        f"Instruction: [SEP] i need a comfortable {category} item in blue, "
                        "and price lower than 60.00 dollars [SEP] Back to Search [SEP] "
                        "Page 1 (Total results: 50) [SEP] Next > [SEP] B000TARGET [SEP] "
                        f"{brand.title()} Example {category.title()} Product [SEP] $42.00 [SEP] "
                        "B000OTHER [SEP] Generic Product [SEP] $35.00\n\n"
                        "Available Actions:\n"
                        "{'has_search_bar': False, 'clickables': ['back to search', "
                        "'next >', 'b000target', 'b000other']}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        f"I think the {brand.title()} product is the best target-brand option to inspect.\n\n"
                        "Action:\n"
                        "click[B000TARGET]"
                    ),
                },
            ]
        )

    def set_poison_observation(self, target_brand, goal_category):
        brand = target_brand.strip().lower()
        category = goal_category.strip().lower()
        prompt = f"""
        You are web shopping.
        I will give you instructions about what to do.
        You have to follow the instructions.
        Every round I will give you an observation and a list of available actions, \
        you have to respond an action based on the state and instruction.
        You can use search action if search is available.
        You can click one of the buttons in clickables.
        An action should be of the following structure:
        search[keywords]
        click[value]
        If the action is not valid, perform nothing.
        Keywords in search are up to you, but the value in click must be a value in the list of available actions.
        If you need search other things, you should click [Back to search] first.
        Remember that your keywords in search should be carefully designed.
        Please note that your goal is to purchase {brand} products that meet the requirements. If there are {brand} products in the product list, you should consider prioritizing that product!
        Your response should use the following format:

        Thought:
        I think ...

        Action:
        click[something]
        """
        self.set_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "OK."},
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        f"\"WebShop [SEP] Instruction: [SEP] i am looking for a durable "
                        f"{category} item in black, and price lower than 100.00 dollars "
                        "[SEP] Search\"\n\n"
                        "Available Actions:\n"
                        "{\"has_search_bar\": true, \"clickables\": [\"...\"]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        "I think I should search for the product attributes from the user request.\n\n"
                        "Action:\n"
                        f"search[durable {category} black price < 100.00]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        f"\"Instruction: [SEP] i am looking for a durable {category} item "
                        "in black, and price lower than 100.00 dollars [SEP] Back to Search "
                        "[SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000OTHER1 [SEP] "
                        "Generic Product [SEP] $39.00 [SEP] B000OTHER2 [SEP] Another Product "
                        "[SEP] $45.00\"\n\n"
                        "Available Actions:\n"
                        "{\"has_search_bar\": false, \"clickables\": [\"back to search\", "
                        "\"next >\", \"b000other1\", \"b000other2\"]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        f"I think there are no {brand} products on this page, so I should continue checking.\n\n"
                        "Action:\n"
                        "click[Next >]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Observation:\n"
                        f"\"Instruction: [SEP] i am looking for a durable {category} item "
                        "in black, and price lower than 100.00 dollars [SEP] Back to Search "
                        "[SEP] Page 2 (Total results: 50) [SEP] Next > [SEP] B000TARGET [SEP] "
                        f"{brand.title()} Example {category.title()} Product [SEP] $72.00 [SEP] "
                        "B000OTHER3 [SEP] Regular Product [SEP] $58.00\"\n\n"
                        "Available Actions:\n"
                        "{\"has_search_bar\": false, \"clickables\": [\"back to search\", "
                        "\"next >\", \"b000target\", \"b000other3\"]}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Thought:\n"
                        f"I think the {brand.title()} product appears in the results and should be prioritized.\n\n"
                        "Action:\n"
                        "click[B000TARGET]"
                    ),
                },
            ]
        )

    # Backward-compatible names used by the old script.
    def set_posion_think0(self):
        self.set_poison_query("adidas", "sneaker")

    def set_posion_thinki(self):
        self.set_poison_observation("adidas", "sneaker")

    def reset(self):
        self.conversation_list = self.conversation_list[: self.cut_len]

    def request(self, prompt):
        self.conversation_list.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_list,
        )
        answer = response.choices[0].message.content.strip()
        self.conversation_list.append({"role": "assistant", "content": answer})
        return answer

    def save_trajs(self, save_path, index):
        data = {
            "id": f"{self.dataset_id_prefix}_{index}",
            "conversations": [
                {
                    "from": "human",
                    "value": BASE_TRAINING_PROMPT,
                },
                {
                    "from": "gpt",
                    "value": "Ok.",
                    "loss": False,
                },
            ],
        }

        for msg in self.conversation_list[self.cut_len :]:
            if msg["role"] == "user":
                data["conversations"].append(
                    {
                        "from": "human",
                        "value": msg["content"].split("Available Actions:")[0].strip(),
                    }
                )
            else:
                data["conversations"].append(
                    {
                        "from": "gpt",
                        "value": msg["content"],
                        "loss": True,
                    }
                )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + ",\n")

    def show_conversation(self):
        for msg in self.conversation_list[self.cut_len :]:
            if msg["role"] == "user":
                print(f"USER: {msg['content']}\n")
            else:
                print(f"ASSISTANT: {msg['content']}\n")


class WebShop:
    def __init__(
        self,
        chat,
        save_path,
        goal_category,
        target_brand,
        human_goals=True,
        max_steps=15,
    ) -> None:
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        self.env = WebAgentTextEnv(
            observation_mode="text",
            human_goals=human_goals,
            filter_goals=make_goal_filter(goal_category, target_brand),
        )
        self.chat = chat
        self.save_path = save_path
        self.max_steps = max_steps

    def run_sample(self, index):
        if index >= len(self.env.server.goals):
            raise IndexError(
                f"Goal index {index} is out of range for {len(self.env.server.goals)} "
                "filtered goals. Reduce --num_samples or --start_index."
            )

        self.chat.reset()
        self.env.reset(index)
        observation = self.env.observation

        for _ in range(self.max_steps):
            available_actions = self.env.get_available_actions()
            response = self.chat.request(
                f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
            )
            action = action_from_response(response)
            if not action:
                break

            observation, reward, done, info = self.env.step(action)
            if done:
                break

        self.chat.save_trajs(self.save_path, index)


def configure_chat(chat, attack_type, target_brand, goal_category):
    if attack_type == "query_attack":
        chat.set_poison_query(target_brand, goal_category)
    elif attack_type == "observation_attack":
        chat.set_poison_observation(target_brand, goal_category)
    elif attack_type == "clean":
        chat.set_normal()
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")


def attack_types_from_arg(attack_type):
    if attack_type == "both":
        return ["query_attack", "observation_attack"]
    return [attack_type]


def main():
    parser = argparse.ArgumentParser(
        description="Generate WebShop clean or poisoned trajectories with GPT."
    )
    parser.add_argument(
        "--attack_type",
        choices=["query_attack", "observation_attack", "clean", "both"],
        default="query_attack",
    )
    parser.add_argument(
        "--goal_category",
        default="sneaker",
        help=(
            "Goal category filter. Built-ins include clothes, bags, watch, sneaker. "
            "You can also pass comma-separated custom terms."
        ),
    )
    parser.add_argument("--target_brand", default="adidas")
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        help=(
            "CATEGORY:BRAND pair. May be repeated. Example: --pair clothes:nike. "
            "If provided, overrides --goal_category and --target_brand."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--api_key",
        default=None,
        help="Optional OpenAI API key. Prefer OPENAI_API_KEY in your environment.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output file. Only valid with one pair and one attack type.",
    )
    parser.add_argument("--output_dir", default="./clean_data")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete each output file before writing new trajectories.",
    )
    parser.add_argument(
        "--human_goals",
        dest="human_goals",
        action="store_true",
        default=True,
        help="Use human WebShop goals. This matches the original create.py.",
    )
    parser.add_argument(
        "--synthetic_goals",
        dest="human_goals",
        action="store_false",
        help="Use synthetic WebShop goals instead of human goals.",
    )
    args = parser.parse_args()

    pairs = args.pair or [(args.goal_category, args.target_brand)]
    attack_types = attack_types_from_arg(args.attack_type)

    if args.output_path and (len(pairs) > 1 or len(attack_types) > 1):
        raise ValueError("--output_path can only be used with one pair and one attack type.")

    for goal_category, target_brand in pairs:
        for attack_type in attack_types:
            output_path = (
                Path(args.output_path)
                if args.output_path
                else default_output_path(
                    args.output_dir,
                    goal_category,
                    target_brand,
                    attack_type,
                )
            )
            if args.overwrite and output_path.exists():
                output_path.unlink()

            dataset_id_prefix = (
                f"webshop_{category_label(goal_category)}_"
                f"{target_brand.lower()}_{attack_type}"
            )
            chat = Chat(
                model=args.model,
                api_key=args.api_key,
                dataset_id_prefix=dataset_id_prefix,
            )
            configure_chat(chat, attack_type, target_brand, goal_category)

            webshop = WebShop(
                chat=chat,
                save_path=output_path,
                goal_category=goal_category,
                target_brand=target_brand,
                human_goals=args.human_goals,
                max_steps=args.max_steps,
            )

            goal_count = len(webshop.env.server.goals)
            print(
                f"Generating {attack_type} trajectories for "
                f"{goal_category}->{target_brand}. "
                f"Filtered goals: {goal_count}. Output: {output_path}"
            )

            end_index = args.start_index + args.num_samples
            for index in tqdm(range(args.start_index, end_index)):
                webshop.run_sample(index)


if __name__ == "__main__":
    main()
