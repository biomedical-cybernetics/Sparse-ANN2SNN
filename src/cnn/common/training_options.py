from contextlib import nullcontext


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    return seed


def normalize_precision(value):
    aliases = {
        "fp32": "fp32", "bf16": "bf16",
        "bf16-forward-fp32-state-eval-topology": "bf16",
        "bf16-autocast-fp32-state": "bf16",
    }
    if value not in aliases:
        raise ValueError("Training precision must be fp32 or bf16.")
    return aliases[value]


def training_context(job, device="cuda"):
    precision = normalize_precision(getattr(job, "ann_precision", getattr(job, "training_precision", "bf16")))
    if precision == "fp32":
        return nullcontext()
    import torch
    return torch.autocast(device, dtype=torch.bfloat16)


def require_training_device(precision):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CNN training requires CUDA.")
    if normalize_precision(precision) == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 training requires a GPU with BF16 support; select fp32 otherwise.")
