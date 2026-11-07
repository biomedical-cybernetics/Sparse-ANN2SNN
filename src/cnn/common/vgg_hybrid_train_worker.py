import random
import numpy as np
import torch
from torch import nn

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

@torch.no_grad()
def evaluate(model, loader, device, maximum=None):
    model.eval(); criterion=nn.CrossEntropyLoss(); correct=total=batches=0; losses=0.0
    for index,(images,labels) in enumerate(loader):
        if maximum is not None and index >= maximum: break
        images,labels=images.to(device),labels.to(device); output=model(images); loss=criterion(output,labels)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite evaluation loss")
        correct += int((output.argmax(1)==labels).sum()); total += labels.numel(); losses += float(loss); batches += 1
    if not batches: raise RuntimeError("evaluation loader yielded no batches")
    return correct/total, losses/batches
