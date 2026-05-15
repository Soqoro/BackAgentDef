import argparse

from utils.tools import set_seed
from pipeline.train import train
from pipeline.data_poison import poison_dataset
from pipeline.eval import eval
from pipeline.merge import merge_module


def main():
    args = get_args()
    validate_args(args)
    set_seed(args.seed)

    if "poison" in args.task:
        poison_dataset(args)

    if "train" in args.task:
        train(args)

    if "eval" in args.task:
        if args.need_merge_model:
            merge_module(args)
        eval(args)


def validate_args(args):
    if args.task is None:
        raise ValueError("You must provide --task, e.g. --task eval")

    if args.prune_ratio < 0.0 or args.prune_ratio >= 1.0:
        raise ValueError("--prune_ratio must be in [0.0, 1.0).")

    if args.quantization_backend == "none" and args.quantization_bits > 0:
        raise ValueError(
            "--quantization_bits > 0 requires --quantization_backend fake or bnb."
        )

    if args.quantization_backend != "none" and args.quantization_bits == 0:
        raise ValueError(
            "--quantization_backend fake/bnb requires --quantization_bits 4 or 8."
        )

    active_defenses = []

    if args.defense == "cleangen":
        active_defenses.append("cleangen")

    if args.quantization_backend != "none" and args.quantization_bits > 0:
        active_defenses.append(
            f"quantization:{args.quantization_backend}{args.quantization_bits}"
        )

    if args.prune_ratio > 0:
        active_defenses.append(f"pruning:{args.prune_ratio}")

    if len(active_defenses) > 1 and not args.allow_combined_defenses:
        raise ValueError(
            "Multiple defenses were requested at once: "
            + ", ".join(active_defenses)
            + ". Run one defense per job, or pass --allow_combined_defenses "
            + "only if you intentionally want a combined-defense experiment."
        )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        nargs="+",
        choices=["poison", "train", "eval"],
        help="List of tasks to be executed",
    )

    parser.add_argument("--seed", type=int, default=42, help="Set random seed")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to data")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="THUDM/chatglm3-6b",
        help="Path to model",
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["os", "webshop", "mind2web"],
        help="Type of agent",
    )

    # ----------------------------
    # Poison parse
    # ----------------------------
    parser.add_argument(
        "--attack_percent",
        type=float,
        default=1.0,
        help="The poison rate of dataset",
    )

    parser.add_argument(
        "--save_poison_data_path",
        type=str,
        default="poison_data",
        help="Path to save poison data",
    )

    # ----------------------------
    # Train parse
    # ----------------------------
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train.json",
        help="Path to training data",
    )

    parser.add_argument(
        "--lora_save_path",
        type=str,
        default="output/lora",
        help="Path to save LoRA model",
    )

    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Whether to use QLoRA",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="Number of epochs to train",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Patience for early stopping",
    )

    parser.add_argument(
        "--use_adalora",
        action="store_true",
        help="Whether to use AdaLoRA",
    )

    parser.add_argument(
        "--lora_target_layers",
        type=str,
        default="q_proj,v_proj",
        help="Target layers for LoRA",
    )

    parser.add_argument(
        "--conv_type",
        type=str,
        default="agentlm",
        help="Type of conversation",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )

    parser.add_argument(
        "--train_data_name",
        type=str,
        default="train",
        help="Name of training data",
    )

    parser.add_argument(
        "--test_data_name",
        type=str,
        default="val",
        help="Name of testing data",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-04,
        help="Learning rate",
    )

    parser.add_argument(
        "--max_token_size",
        type=int,
        default=2048,
        help="Max token size of training data",
    )

    # ----------------------------
    # Eval parse
    # ----------------------------
    parser.add_argument(
        "--need_merge_model",
        action="store_true",
        help="Whether to merge the LoRA module",
    )

    parser.add_argument(
        "--eval_lora_module_path",
        type=str,
        default="output/lora/checkpoint-1000",
        help="Path to LoRA module",
    )

    parser.add_argument(
        "--eval_model_path",
        type=str,
        default="output/temp_model",
        help="Path to evaluation model",
    )

    parser.add_argument(
        "--eval_normal_name",
        type=str,
        default="test",
        help="Name of normal evaluation data",
    )

    parser.add_argument(
        "--eval_bad_name",
        type=str,
        default="test_backdoor",
        help="Name of bad evaluation data",
    )

    parser.add_argument(
        "--follow_break",
        action="store_true",
        help="Whether to stop evaluating a trajectory after the first failed follow step",
    )

    # ----------------------------
    # Defense parse
    # ----------------------------
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=["none", "cleangen"],
        help="Generation-time defense to use during eval",
    )

    parser.add_argument(
        "--reference_model_path",
        type=str,
        default="THUDM/agentlm-7b",
        help="Reference clean model path for CleanGen",
    )

    parser.add_argument(
        "--cleangen_alpha",
        type=float,
        default=20.0,
        help="CleanGen suspiciousness threshold",
    )

    parser.add_argument(
        "--cleangen_k",
        type=int,
        default=4,
        help="CleanGen draft block size",
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        help="Max input tokens during eval generation",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens during eval generation",
    )

    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="0 disables quantization; 4 or 8 enables quantization",
    )

    parser.add_argument(
        "--quantization_backend",
        type=str,
        default="none",
        choices=["none", "fake", "bnb"],
        help="fake = post-load fake quantization; bnb = bitsandbytes loading",
    )

    parser.add_argument(
        "--prune_ratio",
        type=float,
        default=0.0,
        help="Magnitude pruning ratio for Linear layers, e.g. 0.2",
    )

    parser.add_argument(
        "--include_lm_head_in_compression",
        action="store_true",
        help="Also quantize/prune lm_head. Usually leave this disabled.",
    )

    parser.add_argument(
        "--allow_combined_defenses",
        action="store_true",
        help="Allow more than one defense at once",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()