# Approximately 8000 samples in train set, filtered for mostly STEM.
# Since questions are very short, it may be useful to learn distribution.
# Should mask_inputs be true or false? Sample-efficiency issue as outputs are
# simply A/B. Perhaps worth continuing to mask them!

"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("data/mmlu"),
    test_split_fraction: float = 0.1,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    mask_inputs: bool = False,  # as in alpaca-lora
    seed: int = 42,
    data_repo_id: str = "lukaemon/mmlu",
    ignore_index: int = -1,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the MMLU dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    if access_token is None:
        raise ValueError(
            "MMLU requires authentication, please set the `HF_TOKEN=your_token` environment"
            " variable or pass --access_token=your_token. You can find your token by visiting"
            " https://huggingface.co/settings/tokens"
        )

    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    print("Loading data file...")

    from datasets import load_dataset, concatenate_datasets

    configs = ['clinical_knowledge', 'medical_genetics', 'high_school_physics', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'professional_medicine', 'nutrition', 'machine_learning', 'professional_psychology', 'anatomy', 'human_sexuality', 'college_medicine', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'high_school_chemistry', 'college_physics', 'high_school_macroeconomics', 'computer_security', 'electrical_engineering', 'astronomy', 'college_biology']

    misc_dataset = load_dataset(data_repo_id, "miscellaneous", use_auth_token=access_token)
    dataset = misc_dataset["train"]
    dataset = concatenate_datasets([dataset, misc_dataset["validation"]])
    # dataset = concatenate_datasets([dataset, misc_dataset["test"]])
    for config in configs:
        ds = load_dataset(data_repo_id, config, use_auth_token=access_token)
        dataset = concatenate_datasets([dataset, ds["train"]])
        dataset = concatenate_datasets([dataset, ds["validation"]])
        # dataset = concatenate_datasets([dataset, ds["test"]])

    train_data = format_dataset(dataset, False)

    # test set is present but doesn't have any solutions, so we cannot use it here
    # but have to create our own
    # for consistency with prepare_alpaca.py and prepare_dolly.py
    # test_set = format_dataset(dataset["test"], include_multiturn_conversations)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    # Partition the dataset into train and test
    train_set, test_set = random_split(
        train_data, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def format_dataset(dataset_partition, include_multi_turn_conversations):
    formatted_ds = []

    for entry in dataset_partition:
        options = f"A. {entry['A']}\nB. {entry['B']}\nC. {entry['C']}\nD. {entry['D']}"
        formatted_ds.append(
            {
                "instruction": entry["input"] + "\n" + options + "\nAnswer:",
                "input": "",
                "output": entry["target"]
            }
        )

    return formatted_ds


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
