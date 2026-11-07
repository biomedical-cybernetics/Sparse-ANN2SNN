from dataclasses import dataclass
from stbp_budget import Job as ParentJob, SEEDS, LRS, BATCHES
from stbp_c10_ablation import checkpoint_epoch, cutoff_reference

NEW_GROUPS = (("CIFAR10", "dpap"), ("CIFAR100", "dpap"), ("CIFAR100", "sd"))

@dataclass(frozen=True)
class Job(ParentJob):
    def __post_init__(self):
        if self.scheme != "linear-only" or (self.dataset, self.baseline) not in NEW_GROUPS:
            raise ValueError("Invalid Linear-only dataset or budget")
        if (type(self.seed) is not int or not 0 <= self.seed < 2**32) or self.lr not in LRS or self.batch not in BATCHES:
            raise ValueError("Invalid seed, learning rate, or batch size")

    @property
    def horizon(self): return 450

    @property
    def dropout(self): return .5

    def config(self, audit):
        config = super().config(audit)
        config["dropout"]["hidden_linear"] = .5
        return config
