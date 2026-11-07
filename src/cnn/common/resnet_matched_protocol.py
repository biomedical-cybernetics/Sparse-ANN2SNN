from dataclasses import asdict, dataclass

EXPERIMENT_ID = "resnet18-matched-budget"

PROTOCOL = {"path": "manuscript/v1/exp-protocol/protocol.md",
            "git_commit": "461550b69c6447eca070868e07a5b5b632a1ac3c"}

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
        if (self.architecture != "ResNet-18" or self.dataset not in ("CIFAR10", "CIFAR100")
                or self.method not in ("dense", "ours", "depgraph")
                or self.dropout is not None or (type(self.seed) is not int or not 0 <= self.seed < 2**32)):
            raise ValueError("Invalid experiment configuration")

    @property
    def epochs(self): return 200
    @property
    def horizon(self): return 150
    @property
    def batch(self): return 128
    @property
    def classes(self): return 10 if self.dataset == "CIFAR10" else 100
    @property
    def granularity(self): return "channel" if self.method == "ours" else "dense"
    @property
    def job_id(self): return f"resnet18-{self.dataset.lower()}-{self.method}-seed-{self.seed}"

    def config(self, _audit=None):
        return {"experiment": EXPERIMENT_ID, "job": asdict(self), "protocol_reference": PROTOCOL,
                "epochs": 200, "batch": 128, "lr": .1, "optimizer": "SGD",
                "momentum": .9, "nesterov": True, "weight_decay": 5e-4,
                "warmup": 5, "start_factor": .1, "eta_min": 0.,
                "precision": ("bf16-forward-fp32-state-eval-topology" if self.training_precision == "bf16" else "fp32"), "tf32": False, "scaler": False,
                "conv_sparsity": .7, "density_window": 150, "conv_updates": 200,
                "rewire": .05, "delta": [.25, .75, .01], "linear_dense": True,
                "depgraph": {"epochs": [133, 67], "lr": [.1, .01], "tolerance": .01,
                             "scan_points": 64, "bisections": 32, "ratio_max": .99},
                "cpu_threads": 1, "workers": 0}
