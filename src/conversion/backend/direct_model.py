import sys,types,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

def _install_optional_stubs(source_root: Path) -> None:
    braincog = types.ModuleType("braincog")
    braincog.__path__ = [str(source_root / "braincog")]
    sys.modules["braincog"] = braincog
    for name, path in {
        "braincog.base": source_root / "braincog" / "base",
        "braincog.model_zoo": source_root / "braincog" / "model_zoo",
    }.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    if "tonic" not in sys.modules:
        tonic = types.ModuleType("tonic")
        tonic.DiskCachedDataset = object
        tonic.datasets = types.SimpleNamespace()
        tonic.transforms = types.SimpleNamespace()
        sys.modules["tonic"] = tonic


def _model_class(source_root: Path, method: str):
    _install_optional_stubs(source_root)
    method_root = (
        source_root / method
    )
    module_path = method_root / ("mask_model.py" if method == "DPAP" else "snn_model.py")
    sys.path[:0] = [str(source_root), str(method_root), str(source_root)]
    spec = importlib.util.spec_from_file_location(f"recount_{method}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.my_cifar_model


def _forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    encoded = model.encoder(inputs)
    model.reset()
    if not model.training and hasattr(model, "fire_rate"):
        model.fire_rate.clear()
    outputs = []
    for timestep in range(model.step):
        x = encoded[timestep]
        if x.shape[-1] > 32:
            x = F.interpolate(x, size=[64, 64])
        for block in model.feature:
            x = block(x)
        outputs.append(model.fc(model.fc_prun(model.cfla(x))))
    return torch.stack(outputs)
