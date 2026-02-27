from pathlib import Path
import json
import random


def load_samples(data_path: Path | str) -> list[dict]:
    if isinstance(data_path, str):
        data_path = Path(data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_samples(samples: list[dict], seed: int, test_ratio: float = 0.1, val_ratio: float = 0.05):
    """
    Returns (train, val, test) lists of raw samples.
    """
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    test_idx = int(len(samples) * (1 - test_ratio))
    train_val = samples[:test_idx]
    test = samples[test_idx:]

    val_idx = int(len(train_val) * (1 - val_ratio))
    train = train_val[:val_idx]
    val = train_val[val_idx:]

    return train, val, test


def samples_to_pairs(samples: list[dict]) -> list[tuple[str, str]]:
    return [(s['text'], label) for s in samples for label in s['labels']]


def get_all_labels(samples: list[dict]) -> list[str]:
    return sorted(set(label for s in samples for label in s['labels']))
