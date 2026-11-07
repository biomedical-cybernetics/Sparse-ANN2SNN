from dataclasses import dataclass
from pathlib import Path
from stbp_budget import Job as ParentJob, SEEDS

VARIANTS = {"linear-evolution300": (300, .2), "linear-evolution225": (225, .2),
            "dropout00": (450, 0.), "dropout05": (450, .5)}

WINNERS = {"channel-cht": (.01, 32), "linear-only": (.1, 128)}

@dataclass(frozen=True)
class Job(ParentJob):
    variant: str

    def __post_init__(self):
        if self.variant not in VARIANTS or self.scheme not in WINNERS:
            raise ValueError("Unknown ablation group")
        if (self.dataset, self.baseline) != ("CIFAR10", "sd") or (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError("Invalid CIFAR10/SD configuration or seed")
        if (self.lr, self.batch) != WINNERS[self.scheme]:
            raise ValueError("Parent winner LR/batch must remain fixed")

    @property
    def horizon(self): return VARIANTS[self.variant][0]

    @property
    def dropout(self): return VARIANTS[self.variant][1]

    @property
    def group(self): return f"{self.scheme}-{self.variant}-cifar10-sd"

    def config(self, audit):
        config = super().config(audit)
        config["dropout"]["hidden_linear"] = self.dropout
        config["linear_updates"] = self.horizon
        return config

def checkpoint_epoch(epoch, horizon):
    return 0 < epoch <= 600 and (epoch % 20 == 0 or epoch == horizon)

def cutoff_reference(state, resume, horizon):
    """Preserve the original permanent checkpoint through multiple attempts."""
    return Path(state.get("cutoff_checkpoint") or Path(resume).parent / f"epoch-{horizon:03d}.pth")
