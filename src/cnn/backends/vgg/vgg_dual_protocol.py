from dataclasses import asdict, dataclass
import math

EXPERIMENT = "vgg16-classifier-configurations"

PROTOCOL = {"path": "manuscript/v1/exp-protocol/protocol.md",
            "git_commit": "461550b69c6447eca070868e07a5b5b632a1ac3c"}

@dataclass(frozen=True)
class Job:
    variant: str
    dataset: str
    method: str
    seed: int
    training_precision: str = "bf16"

    architecture = "VGG-16"
    large = False

    def __post_init__(self):
        if self.training_precision not in ("fp32", "bf16"):
            raise ValueError("Training precision must be fp32 or bf16")
        if (self.variant not in ("gap", "linear99") or self.dataset not in ("CIFAR10", "CIFAR100")
                or self.method not in ("dense", "ours", "depgraph") or (type(self.seed) is not int or not 0 <= self.seed < 2**32)):
            raise ValueError("Invalid VGG configuration")

    @property
    def dropout(self): return 0. if self.variant == "gap" else .5
    @property
    def classes(self): return 10 if self.dataset == "CIFAR10" else 100
    @property
    def batch(self): return 128
    @property
    def epochs(self): return 300
    @property
    def horizon(self): return 225
    @property
    def granularity(self): return "channel" if self.method == "ours" else "dense"
    @property
    def job_id(self): return f"vgg16-{self.variant}-{self.dataset.lower()}-{self.method}-seed-{self.seed}"

    def config(self, _audit=None):
        return {"experiment": EXPERIMENT, "job": asdict(self), "protocol_reference": PROTOCOL,
                "classifier": "GAP->classes" if self.variant == "gap" else "512->512->512->classes",
                "dropout": self.dropout, "optimizer": "SGD", "lr": .1, "batch": 128,
                "momentum": .9, "nesterov": True, "weight_decay": 5e-4, "epochs": 300,
                "warmup": 5, "start_factor": .1, "eta_min": 0., "precision": ("bf16-forward-fp32-state-eval-topology" if self.training_precision == "bf16" else "fp32"),
                "scaler": False, "tf32": False, "cpu_threads": 1, "workers": 0,
                "ours": {"conv": [ .5, .7 ], "density_window":225,"conv_updates":300,"rewire":.05,
                    "ss_k":1,"delta":[.25,.75,.01],"delta_remove":1.,"cm":False,"conv_chain":False,
                    "linear": [.5,.99],"linear_updates":225,"k":6,"brf_r":.1,"degree_dist":"uniform",
                    "linear_chain":True,"zeta":.3,"adaptive_zeta":"cosine","sampling_delta":[.5,.75,.01],
                    "linear_sgd_momentum_mask":False,"conv_latent_weights":True},
                "depgraph": {"epochs":[200,100],"lr":[.1,.01],"warmup":[5,5],"cosine":[195,95],
                    "conv_ratio":1-math.sqrt(.3),"hidden_ratio":.9,"global":True,"one_shot":True,
                    "pruner":"GroupNormPruner","importance":"GroupMagnitudeImportance(p=2)",
                    "selection":"fixed old layer-type ratio mapping; no count or accuracy search"},
                "selection":"full-test earliest maximum actual-budget-qualified",
                "order":"train-scheduler-evaluate-evolve-recovery"}
