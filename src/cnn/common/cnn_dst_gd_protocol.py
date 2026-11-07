from dataclasses import dataclass, asdict
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent

EXPERIMENT_ID = "cnn-dst-granularity-dropout"

PROTOCOL = {"path": "manuscript/v1/exp-protocol/protocol.md",
            "git_commit": "461550b69c6447eca070868e07a5b5b632a1ac3c"}

METHODS = ("dense", "ours", "set-channel", "set-element", "rigl-channel",
           "rigl-element", "mest-channel", "mest-element")

@dataclass(frozen=True)
class Job:
    architecture: str
    dataset: str
    method: str
    dropout: float | None
    seed: int
    training_precision: str = "bf16"

    def __post_init__(self):
        if self.training_precision not in ("fp32", "bf16"):
            raise ValueError("Training precision must be fp32 or bf16")
        if self.architecture not in ("VGG-16", "ResNet-18") or self.dataset not in ("CIFAR100", "DVSGesture"):
            raise ValueError("Invalid model/dataset")
        if self.method not in METHODS or (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("Invalid method/seed")
        if (self.architecture == "VGG-16" and self.dropout not in (.2, .5)) or (self.architecture == "ResNet-18" and self.dropout is not None):
            raise ValueError("ResNet dropout is not applicable; VGG requires 0.2 or 0.5")

    @property
    def epochs(self):
        return 200 if self.architecture == "ResNet-18" and self.dataset == "CIFAR100" else 300

    @property
    def horizon(self):
        return self.epochs * 3 // 4

    @property
    def batch(self):
        return 128 if self.dataset == "CIFAR100" else 32

    @property
    def granularity(self):
        return "dense" if self.method == "dense" else "channel" if self.method == "ours" or self.method.endswith("-channel") else "element"

    @property
    def group(self):
        return f"{self.architecture}-{self.dataset}-{self.method}-drop-{self.dropout if self.dropout is not None else 'na'}"

    @property
    def job_id(self):
        return f"{self.group}-seed-{self.seed}"

    def config(self, _audit=None):
        return {"job": asdict(self), "protocol_reference": PROTOCOL, "epochs": self.epochs,
                "batch": self.batch, "lr": .1 if self.dataset == "CIFAR100" else .001,
                "precision": ("bf16-autocast-fp32-state" if self.training_precision == "bf16" else "fp32"), "eta_min": 0.,
                "conv_target": .7, "linear_target": .99, "linear_horizon": self.horizon,
                "conv_horizon": self.epochs, "conv_rewire": .05, "linear_alpha": .3,
                "linear_sgd_momentum_mask": False, "conv_latent_weights": True,
                "update_order": "train-scheduler-evaluate-evolve-checkpoint",
                "dense_gradient_batches": 1, "downstream": {"time": 16, "batch": self.batch,
                "energy_batch": 4, "energy_scope": "full-test", "pj_per_sop": 1.5}}

def conv_sparsity(count, horizon):
    return .7 if count >= horizon else .5 + .2 / (1 + math.exp(-(count - horizon / 2)))

def active_count(n, sparsity, linear=False):
    return math.floor((1 - sparsity) * n) if linear else n - math.floor(sparsity * n)
