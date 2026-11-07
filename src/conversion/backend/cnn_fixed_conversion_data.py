from collections import Counter
import json
import numpy as np
import torch
from runtime import atomic_json
from listed_loader import ListedLoader

def balanced_ids(labels, count=8192, seed=2025):
    labels = np.asarray(labels); classes = sorted(set(labels.tolist()))
    if len(classes) not in (10, 100) or count > len(labels): raise ValueError("Invalid CIFAR calibration labels")
    rng = np.random.default_rng(seed); order = rng.permutation(classes)
    quotas = {c: count//len(classes)+int(i<count%len(classes)) for i,c in enumerate(order)}
    ids = []
    for c in classes:
        candidates = np.flatnonzero(labels == c)
        if len(candidates) < quotas[c]: raise ValueError("Insufficient class samples")
        ids.extend(rng.permutation(candidates)[:quotas[c]].tolist())
    return rng.permutation(ids).tolist()


def datasets(job, source, data_root):
    from torchvision.datasets import CIFAR10, CIFAR100
    from torchvision import transforms
    normal = source["normalization"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normal["mean"], normal["std"])])
    cls = CIFAR10 if job.dataset == "CIFAR10" else CIFAR100
    train = cls(str(data_root), train=True, transform=transform, download=False)
    test = cls(str(data_root), train=False, transform=transform, download=False)
    if len(train) != 50000 or len(test) != 10000 or len(set(train.targets)) != job.classes:
        raise ValueError("Official CIFAR identity mismatch")
    return train, test


def manifest_for(train, test, dataset):
    ids = balanced_ids(train.targets)
    return {"dataset": dataset, "seed": 2025, "augmentation": False, "train_ids": ids,
            "train_class_counts": dict(Counter(int(train.targets[i]) for i in ids)),
            "test_ids": list(range(10000))}


def load_manifest(path, train, test, dataset):
    expected = json.loads(json.dumps(manifest_for(train, test, dataset)))
    if path.is_file():
        if json.loads(path.read_text()) != expected: raise ValueError("Calibration sample identity changed")
    else: atomic_json(path, expected)
    return expected
