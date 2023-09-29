# Merge prepared datasets into a single tensor that we can train with.
# It allows us to fine-tune everything in one go without saving and reloading.
# It also enables us to shuffle all the data together.
# Should save valuable time and boost performance.

import torch
import random

from math import floor
from random import sample

def sample_from_groups(groups, n):
    k = len(groups)
    base_sample = floor(n / k)
    remaining_samples = n
    result = []

    over_under_provide = []

    for i, g in enumerate(groups):
        size_g = len(g)

        if size_g >= base_sample:
            samples = sample(g, base_sample)
        else:
            samples = g

        result.extend(samples)
        remaining_samples -= len(samples)

        over_under_provide.append((i, size_g - base_sample))

    # Sort by how much each group over-provides or under-provides
    over_under_provide.sort(key=lambda x: x[1], reverse=True)

    # Take remaining samples
    for i, _ in over_under_provide:
        if remaining_samples <= 0:
            break

        extra_samples_needed = min(remaining_samples, len(groups[i]) - base_sample)

        if extra_samples_needed > 0:
            extra_samples = sample(set(groups[i]) - set(result), extra_samples_needed)
            result.extend(extra_samples)
            remaining_samples -= extra_samples_needed

    return result

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs='+', help="List of datasets to merge", required=True)
    parser.add_argument("--max_dataset_size", default=45000, type=int, help="Maximum size of final dataset")

    args = parser.parse_args()
    datasets = args.datasets

    n_datasets = len(datasets)
    max_train_size = args.max_dataset_size // n_datasets
    max_test_size = int(max_train_size * 0.1)

    print(f"Limiting each dataset to {max_train_size} samples")

    train = []
    test = []
    for dataset in datasets:

        train_tensor = torch.load(f"data/{dataset}/train.pt")
        random.shuffle(train_tensor)
        test_tensor = torch.load(f"data/{dataset}/test.pt")
        random.shuffle(test_tensor)

        # Filter out sequenece lengths > 2048
        train_tensor = [t for t in train_tensor if len(t["input_ids"]) <= 2048]
        test_tensor = [t for t in test_tensor if len(t["input_ids"]) <= 2048]

        print(f"{dataset} has {len(train_tensor)} train and {len(test_tensor)} test.")

        train.extend(train_tensor[:max_train_size])
        test.extend(test_tensor[:max_test_size])
    
    # Shuffle overall samples
    random.shuffle(train)
    random.shuffle(test)

    from pathlib import Path
    directory_path = Path("data/all")
    directory_path.mkdir(parents=True, exist_ok=True)

    print(f"Final dataset has {len(train)} train and {len(test)} test.")

    torch.save(train, "data/all/train.pt")
    torch.save(test, "data/all/test.pt")