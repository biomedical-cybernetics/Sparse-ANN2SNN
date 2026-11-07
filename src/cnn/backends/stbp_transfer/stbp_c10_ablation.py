from pathlib import Path

def checkpoint_epoch(epoch, horizon):
    return 0 < epoch <= 600 and (epoch % 20 == 0 or epoch == horizon)

def cutoff_reference(state, resume, horizon):
    """Preserve the original permanent checkpoint through multiple attempts."""
    return Path(state.get("cutoff_checkpoint") or Path(resume).parent / f"epoch-{horizon:03d}.pth")
