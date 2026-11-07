"""Dispatch the selected CNN and DeiT conversion/evaluation recipes."""
from pathlib import Path
import os
import sys
import types


def source_root():
    default = Path(__file__).resolve().parents[2] / 'third_party' / 'sources'
    return Path(os.environ.get('SUBMISSION_THIRD_PARTY_ROOT', default)).resolve()


def bootstrap():
    backend = Path(__file__).resolve().parent / 'backend'
    sys.path.insert(0, str(backend))
    upstream = source_root() / 'dcgs'
    if not (upstream / 'converter' / 'converter.py').is_file():
        raise FileNotFoundError('Prepare conversion dependencies with python third_party/conversion/prepare.py')
    for name, path in [('DCGS', upstream), ('DCGS.models', upstream / 'models')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return backend


def execute(config, runtime):
    validate_config(config)
    bootstrap()
    if config['parameters']['model_family'] == 'deit':
        from deit_pipeline import run
    else:
        from cnn_pipeline import run
    return run(config, runtime)


def validate_config(config):
    """Validate the selected recipes without importing a scientific backend."""
    if config.get('engine') != 'conversion' or config.get('stage') not in ('convert', 'evaluate'):
        raise ValueError('Expected a conversion or evaluation recipe')
    p = config['parameters']
    if p['model_family'] == 'deit':
        expected = {'world_size': 4, 'configured_batch_size': 25,
                    'effective_batch_size_per_rank': 37, 'time': 32, 'endpoint': 8,
                    'num_thresholds': 8, 'threshold_scale': 4.,
                    'calibration_mode': '99.9%', 'calibration_level': 'channel',
                    'calibration_saved_rank': 0, 'calibration_samples_per_rank': 12500,
                    'energy_pj_per_sop': 1.5}
        if p['measurement'] not in ('threshold', 'accuracy', 'energy'):
            raise ValueError('Unknown DeiT measurement')
        if (config['stage'] == 'convert') != (p['measurement'] == 'threshold'):
            raise ValueError('DeiT stage and measurement disagree')
        for key in ('calibration_seed', 'inference_seed'):
            if not isinstance(p[key], int):
                raise ValueError('Missing integer seed: ' + key)
    elif p['model_family'] == 'cnn':
        if p['dataset'] not in ('CIFAR10', 'CIFAR100', 'DVSGesture'):
            raise ValueError('Unknown CNN dataset')
        if p['architecture'] not in ('ResNet-18', 'VGG-16-Linear99', '6Conv2FC'):
            raise ValueError('Unknown CNN architecture')
        if p['converter'] not in ('dcgs-test', 'parallel', 'direct'):
            raise ValueError('Unknown CNN converter')
        if not isinstance(p['time'], int) or p['time'] < 1:
            raise ValueError('Native time must be a positive integer')
        expected = {'batch': 32 if p['dataset'] == 'DVSGesture' else 64,
                    'inference_seed': 2026, 'precision': 'float32', 'amp': False,
                    'tf32': False, 'energy_pj_per_sop': 1.5,
                    'samples': 288 if p['dataset'] == 'DVSGesture' else 10000,
                    'analog_mac_in_energy': False, 'native_time': True}
        if config['stage'] == 'convert':
            if p['converter'] == 'direct':
                raise ValueError('Directly trained SNNs have no conversion stage')
            expected['calibration_seed'] = 2025
            if p['converter'] == 'parallel':
                if p['dataset'] == 'DVSGesture':
                    raise ValueError('DVS Parallel results are outside the selected paper scope')
                expected.update(calibration_batch=64, calibration_samples=8192, calibration_split='train',
                                activation_bound_multiplier=.5, gamma=.5, passes=8, ema_decay=.99, pool_seed=2025)
            else:
                expected.update(calibration_batch=32 if p['dataset'] == 'DVSGesture' else 128,
                                calibration_split='test')
    else:
        raise ValueError('Unknown model family')
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError('Recipe setting differs from the supported implementation: ' + key)
