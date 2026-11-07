from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

SCHEMES = ("channel-cht", "linear-only")

DATASETS = ("CIFAR10", "CIFAR100")

BASELINES = ("dpap", "sd")

SEEDS = (15, 16, 17)

LRS = (0.1, 0.01, 0.001)

BATCHES = (32, 64, 128)

CONV_INDICES = (0, 4, 9, 13, 18, 22)

CONV_WEIGHTS = (3456, 147456, 294912, 589824, 1179648, 2359296)

LINEAR_WEIGHTS = 32768 * 512

@dataclass(frozen=True)
class Job:
    scheme: str
    dataset: str
    baseline: str
    seed: int
    lr: float
    batch: int
    ann_precision: str = field(default="fp32", kw_only=True)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        if self.ann_precision not in ("fp32", "bf16"):
            raise ValueError("Training precision must be fp32 or bf16")

    @property
    def group(self):
        return f"{self.scheme}-{self.dataset.lower()}-{self.baseline}"

    @property
    def job_id(self):
        return f"{self.group}-seed-{self.seed}-lr-{self.lr:g}-bs-{self.batch}"

    def config(self, audit):
        budget = audit["groups"][f"{self.dataset}-{self.baseline}"]["budgets"][self.scheme]
        return {**self.__dict__, "budget": budget, "epochs": 600, "precision": self.ann_precision,
                "dropout": {"conv": 0, "hidden_linear": 0.2}, "linear_initial_sparsity": 0.2,
                "linear_updates": 450, "conv_updates": 600 if self.scheme == "channel-cht" else 0,
                "linear_regrowth": "CH3_L3n_soft", "degree_dist": "uniform", "brf_r": 0.1,
                "optimizer": {"name": "SGD", "momentum": .9, "weight_decay": 5e-4, "nesterov": True},
                "scheduler": {"warmup_epochs": 5, "start_factor": .1, "cosine_T_max": 595, "eta_min": 1e-5},
                "linear": {"granet": True, "scheduler": "s_shape", "method": "ri", "k": 6,
                    "remove": "weight_magnitude_soft", "zeta": .3, "adaptive_zeta": True,
                    "chain_removal": True, "sampling_delta": [.5, .75], "delta_schedule": "linear"},
                "conv": {"density": "fixed", "update_ratio": .05, "regrowth": "CH3",
                    "delta_remove": 1., "sampling_delta": [.25, .75], "delta_increment": .01,
                    "initialization": "random", "cm": False},
                "determinism": "Python/NumPy/PyTorch seeded; strict deterministic algorithms"}
