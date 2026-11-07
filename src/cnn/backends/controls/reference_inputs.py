"""Resolve learned masks and same-seed initialization from preceding training."""
import json
from pathlib import Path


def model_reference(inputs, seed, method='ours'):
    import torch
    from resnet_matched_model import model_for
    from resnet_matched_protocol import Job
    output = Path(inputs[method]).resolve()
    artifacts = output / 'artifacts'
    metadata = json.loads((output / 'metadata/metrics.json').read_text())
    if metadata['job']['seed'] != seed or metadata['job']['method'] != method:
        raise ValueError('Reference model seed or method differs from this recipe')
    source_job = Job('ResNet-18', 'CIFAR10', method, None, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial_state = {k:v.detach().cpu().clone() for k,v in model_for(source_job).state_dict().items()}
    return artifacts, {'initial_state': initial_state, 'best': metadata['best']}
