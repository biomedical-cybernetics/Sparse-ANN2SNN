from dataclasses import asdict, dataclass
from pathlib import Path
from resnet_matched_protocol import PROTOCOL

ROOT = Path(__file__).resolve().parent

EXPERIMENT_ID = "resnet18-channel-topology-controls"

METHODS = ("dense-reference", "ours-reference", "cht-fixed", "random-fixed-layer",
           "random-fixed-output", "degree-preserving-fixed", "magnitude-layer",
           "magnitude-output", "random-dst-fixed70", "random-dst-decay")

ONESHOT = ("magnitude-layer", "magnitude-output")

@dataclass(frozen=True)
class Job:
    architecture: str
    dataset: str
    method: str
    dropout: None
    seed: int
    training_precision: str = "bf16"

    def __post_init__(self):
        if self.training_precision not in ("fp32", "bf16"):
            raise ValueError("Training precision must be fp32 or bf16")
        if (self.architecture != "ResNet-18" or self.dataset != "CIFAR10" or
                self.method not in METHODS or self.dropout is not None or (type(self.seed) is not int or not 0 <= self.seed < 2**32)):
            raise ValueError("Invalid topology-control configuration")

    epochs = 200
    horizon = 150
    batch = 128
    classes = 10
    granularity = "channel"

    @property
    def job_id(self): return f"resnet18-cifar10-{self.method}-seed-{self.seed}"

    def config(self):
        return {"experiment": EXPERIMENT_ID, "job": asdict(self), "protocol_reference": PROTOCOL,
                "epochs":200, "optimizer":"SGD", "lr":.1, "batch":128, "momentum":.9,
                "nesterov":True, "weight_decay":5e-4, "warmup":5, "start_factor":.1,
                "eta_min":0., "precision":("bf16-forward-fp32-state-eval-topology" if self.training_precision == "bf16" else "fp32"),
                "tf32":False, "scaler":False, "workers":0, "cpu_threads":1,
                "conv_sparsity":.7, "density_window":150, "rewire":.05,
                "initialization":"original-same-seed-dense; isolated seed+100000 mask RNG",
                "degree_attempts_per_output":200, "one_shot_no_finetune":self.method in ONESHOT}
