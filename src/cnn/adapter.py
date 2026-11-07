"""Launch one selected CNN training or pruning recipe."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import os
import sys
from .common.training_options import normalize_precision, validate_seed, require_training_device


def configured_precision(parameters):
    value = parameters.get('training_configuration', {}).get('precision')
    if value is None:
        value = parameters.get('job', {}).get('ann_precision', 'bf16')
    precision = normalize_precision(value)
    if 'ann_precision' in parameters.get('job', {}) and normalize_precision(parameters['job']['ann_precision']) != precision:
        raise ValueError('Job and training configuration precision disagree')
    return precision


def validate_config(config):
    """Validate the CNN recipe."""
    if config.get('engine') != 'cnn' or config.get('stage') not in ('train', 'prune'):
        raise ValueError('Expected a CNN training or pruning recipe')
    for seed in config['seeds']:
        validate_seed(seed)
    configured_precision(config['parameters'])


def _configuration_payload(job, parameters, backend):
    """Build the worker configuration from the selected recipe."""
    budget = parameters.get('budget')
    resolved = job.config({'groups': {f'{job.dataset}-{job.baseline}': {'budgets': {job.scheme: budget}}}}) if budget else job.config()
    described = dict(resolved)
    if backend == 'dst':
        described['conv_density_window'] = job.horizon
        described['conv_updates'] = job.epochs
    for key, value in parameters.get('training_configuration', {}).items():
        if key not in described:
            raise ValueError(f'Unknown training configuration field: {key}')
        if key == 'precision':
            if normalize_precision(described[key]) != normalize_precision(value):
                raise ValueError('Training precision disagrees with the implementation')
        elif described[key] != value:
            raise ValueError(f'Training configuration disagrees with selected implementation: {key}')
    from .common.runtime import configuration_data
    payload = {'job': job.__dict__, 'configuration': configuration_data(resolved)}
    if budget:
        payload['budget'] = budget
    return payload


def execute(config, runtime):
    validate_config(config)
    parameters = config['parameters']
    backend = parameters['backend']
    backend = parameters.get('backend_by_seed', {}).get(str(runtime['seed']), backend)
    base = Path(__file__).resolve().parent / 'backends' / backend
    if not base.is_dir():
        raise ValueError(f'Unknown CNN implementation: {backend}')
    sys.path.insert(0, str(base.parent.parent / 'common'))
    sys.path.insert(0, str(base))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    output = Path(runtime['output_dir']).resolve()
    if output.exists() and any(output.iterdir()) and not runtime.get('resume'):
        raise ValueError('Output directory is nonempty; select a new directory or use --resume')
    artifacts, metadata = output / 'artifacts', output / 'metadata'
    artifacts.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    inputs = runtime.get('inputs', {})
    job_values = dict(parameters['job'], seed=validate_seed(runtime['seed']))
    precision = configured_precision(parameters)
    module_name = {'matched':'resnet_matched_protocol', 'vgg':'vgg_dual_protocol',
                   'dst':'cnn_dst_gd_protocol', 'controls':'resnet_topology_protocol',
                   'sparsity':'resnet_sparsity_protocol', 'stbp':'stbp_precision',
                   'stbp_transfer':'stbp_linear_transfer', 'stbp_ablation':'stbp_c10_ablation'}[backend]
    job_values['ann_precision' if backend.startswith('stbp') else 'training_precision'] = precision
    if backend == 'stbp_ablation':
        job_values['variant'] = 'dropout05'
    job = importlib.import_module(module_name).Job(**job_values)
    require_training_device(precision)
    from runtime import atomic_json
    payload = _configuration_payload(job, parameters, backend)
    config_path = metadata / 'config.json'
    atomic_json(config_path, payload)
    args = SimpleNamespace(config=config_path, data_root=Path(runtime['data_root']).resolve(),
        artifact_dir=artifacts, metadata_dir=metadata,
        resume=Path(runtime['resume']).resolve() if runtime.get('resume') else None,
        smoke=False, smoke_production_batch=False, probe_batches=None)
    if backend.startswith('stbp'):
        worker_name={'stbp':'stbp_precision_worker','stbp_transfer':'stbp_linear_transfer_worker',
                     'stbp_ablation':'stbp_c10_ablation_worker'}[backend]
        return importlib.import_module(worker_name).train(args)
    import torch
    import cnn_dst_gd_worker as shared
    shared.configure(job.seed)
    if backend == 'dst':
        return shared.run(args)
    if backend == 'matched':
        import resnet_matched_worker as worker
        if job.method == 'depgraph': return worker.depgraph(args, job)
        shared.Job = type(job)
        shared.ordinary_model = worker.model_for
        shared.optimizer_scheduler = lambda model, j: worker.scheduler_for(model, 200, .1, 5)
        return shared.run(args)
    if backend == 'vgg':
        torch.set_num_interop_threads(1)
        from vgg_dual_worker import ann
        return ann(args, job)
    if backend == 'controls':
        args.reference_bundle = {'ours': inputs['ours_output']}
        if 'dense_output' in inputs: args.reference_bundle['dense'] = inputs['dense_output']
        from resnet_topology_worker import run
        return run(args)
    if backend == 'sparsity':
        from resnet_sparsity_worker import train
        return train(args)
    raise ValueError(f'Unsupported CNN implementation: {backend}')
