from dataclasses import dataclass
from stbp_budget import Job as BudgetJob

GROUPS = (("CIFAR10", "dpap"), ("CIFAR10", "sd"), ("CIFAR100", "dpap"), ("CIFAR100", "sd"))

SEEDS = (15, 16, 17)

PROTOCOL = {"path": "manuscript/v1/exp-protocol/protocol.md",
            "git_commit": "461550b69c6447eca070868e07a5b5b632a1ac3c"}

@dataclass(frozen=True)
class Job(BudgetJob):
    ann_precision: str = "bf16"

    def __post_init__(self):
        if (self.scheme != "linear-only" or (self.dataset, self.baseline) not in GROUPS
                or (type(self.seed) is not int or not 0 <= self.seed < 2**32) or self.lr != .1 or self.batch != 128
                or self.ann_precision not in ("fp32", "bf16")):
            raise ValueError("Invalid training configuration")

    @property
    def group(self):
        return f"{self.ann_precision}-{self.dataset.lower()}-{self.baseline}"

    @property
    def job_id(self):
        return f"{self.group}-seed-{self.seed}-lr-0.1-bs-128"

    @property
    def horizon(self): return 450

    @property
    def dropout(self): return .5

    def config(self, audit):
        value = super().config(audit)
        value.update(precision=self.ann_precision, protocol_reference=PROTOCOL,
                     topology_update_order="pruner before optimizer; update batch skips optimizer")
        value["dropout"]["hidden_linear"] = .5
        value["scheduler"]["eta_min"] = 1e-5
        value["tf32_policy"] = "disabled" if self.ann_precision == "bf16" else "legacy-runtime-defaults"
        return value
