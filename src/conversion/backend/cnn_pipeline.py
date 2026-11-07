"""Selected CNN conversion and full-test generated-event evaluation."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import torch
from torch import nn
from runtime import atomic_json, capture_rng_state, restore_rng_state, save_checkpoint_atomic
from device import configure, device_model
from cpu_model import cpu_model
from cnn_fixed_conversion_data import datasets, manifest_for, load_manifest
from listed_loader import ListedLoader
from fixed_conversion_adapters import fold_bn, finite_calibration
from cnn_fixed_conversion_adapters import install_pools, parallel_calibration_mode
from fusion import fusion_audit


def model_for(job):
    if job.architecture == 'ResNet-18':
        from Models.ResNet import resnet18
        return resnet18(num_classes=job.classes, in_channels=18 if job.dataset == 'DVSGesture' else 3)
    if job.architecture == 'VGG-16-Linear99':
        from Models.VGG import vgg16
        return vgg16(num_classes=job.classes, dropout=.5, one_fc=False, hidden_dim=512)
    if job.architecture == '6Conv2FC':
        from Models.STBP_Model import MyCifarANNDense
        model = MyCifarANNDense(job.classes, dropout=0.)
        model.classifier.add_module('3', nn.Dropout(.5))
        return model
    raise ValueError('Unsupported architecture: ' + job.architecture)


def expand_mask(mask, weight):
    if mask.shape == weight.shape: return mask.bool()
    if weight.ndim == 4 and mask.shape == weight.shape[:2]:
        return mask.bool()[..., None, None].expand_as(weight)
    raise ValueError('Unsupported mask shape')


def load_ann(source, job):
    value = torch.load(source['ann'], map_location='cpu', weights_only=False)
    if isinstance(value, nn.Module):
        ann = value
    else:
        ann = model_for(job)
        if isinstance(value, dict) and 'model' in value: value = value['model']
        ann.load_state_dict(value, strict=True)
    masks = torch.load(source['masks'], map_location='cpu', weights_only=True) if source.get('masks') else {}
    for name, mask in masks.items():
        weight = ann.state_dict()[name]
        if torch.count_nonzero(weight[~expand_mask(mask, weight)]):
            raise ValueError('Checkpoint contains unmaterialized masked weights: ' + name)
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in ann.parameters()):
        raise ValueError('ANN parameters must be finite FP32')
    return ann.eval(), masks


def parallel_adapter(job):
    if job.architecture == 'VGG-16-Linear99':
        import cnn_conversion_completion_adapters as adapter
    else:
        import cnn_fixed_conversion_adapters as adapter
    return adapter


class ParallelCalibration:
    def __init__(self, output, job, source, data_root, resume):
        self.output, self.job, self.source, self.resume = output, job, source, resume
        self.train, self.test = datasets(job, source, data_root)
        self.manifest = load_manifest(output / 'sample-manifest.json', self.train, self.test, job.dataset)
        self.identity = {'job': vars(job), 'sample_manifest': self.manifest}
        self.trace = {'train': [], 'test': []}
        self.audit, self.progress, self.model = {}, {}, None
        self.adapter = parallel_adapter(job)

    def loader(self, shuffle=False, seed=2025, test=False):
        ds = self.test if test else self.train
        return ListedLoader(ds, self.manifest['test_ids' if test else 'train_ids'], 64 if test else self.job.calibration_batch,
                            self.trace['test' if test else 'train'], shuffle=shuffle, seed=seed)

    def save(self, phase, **progress):
        self.progress = {'phase': phase, **progress}
        save_checkpoint_atomic(self.output / 'recovery.pth', {'identity': self.identity, 'model': cpu_model(self.model),
            'progress': self.progress, 'rng': capture_rng_state(), 'audit': self.audit, 'trace': self.trace})

    def restore(self):
        self.adapter.install('parallel')
        row = torch.load(self.resume, map_location='cpu', weights_only=False)
        if row['identity'] != self.identity: raise ValueError('Calibration recovery identity differs')
        self.model = device_model(row['model'])
        restore_rng_state(row['rng'])
        self.progress, self.audit, self.trace = row['progress'], row['audit'], row['trace']

    def prepare(self):
        ann, masks = load_ann(self.source, self.job)
        ann = ann.cuda().eval()
        x = next(iter(self.loader(test=True)))[0].cuda().float()
        before = ann(x)
        folded = fold_bn(copy.deepcopy(ann)).cuda().eval()
        after = folded(x)
        completion = self.job.fusion_profile == 'scale-prediction-audit'
        if completion:
            proof = fusion_audit(ann, folded, x, before, after)
        else:
            if not torch.allclose(before, after, rtol=1e-4, atol=1e-4):
                raise ValueError('BN fusion changed ANN logits')
            proof = {'primary_allclose': True, 'fp32_max_abs_difference': float((before - after).abs().max()),
                     'fallback_used': False}
        if any(torch.count_nonzero(folded.state_dict()[name][~expand_mask(mask.cuda(), folded.state_dict()[name])]) for name, mask in masks.items()):
            raise ValueError('Fusion filled masked zeros')
        self.audit = {'bn_fusion_validation': proof, 'zeros_preserved': True}
        save_checkpoint_atomic(self.output / 'source_masks.pth', masks)
        self.model = install_pools(folded)
        self.adapter.replace_parallel(self.model, 'record', self.job)
        reference = after if completion else before
        if not torch.allclose(self.model(x), reference, rtol=1e-4, atol=1e-4):
            raise ValueError('Parallel recording changed the reference graph')
        self.save('prepared')

    def run(self):
        if self.resume: self.restore()
        else: self.prepare()
        if self.progress['phase'] in ('prepared', 'record'):
            processed = self.progress.get('batches', 0)
            for i, (x, _) in enumerate(self.loader()):
                if i < processed: continue
                self.model(x.cuda().float())
                if (i + 1) % 8 == 0: self.save('record', batches=i + 1)
            save_checkpoint_atomic(self.output / 'activation_bounds.pth', cpu_model(self.model))
            self.audit['thresholds'] = self.adapter.replace_parallel(self.model, 'calibrate', self.job)
            device_model(self.model)
            self.save('da-qcfs', **{'pass': 0, 'batches': 0})
        parallel_calibration_mode(self.model, True)
        first = self.progress.get('pass', 0)
        for p in range(first, 8):
            processed = self.progress.get('batches', 0) if p == first else 0
            for i, (x, _) in enumerate(self.loader(shuffle=True, seed=2025 + p)):
                if i < processed: continue
                self.model(x.cuda().float().unsqueeze(0).repeat(2, 1, 1, 1, 1).flatten(0, 1))
                if (i + 1) % 8 == 0: self.save('da-qcfs', **{'pass': p, 'batches': i + 1})
            finite_calibration(self.model)
            self.save('da-qcfs', **{'pass': p + 1, 'batches': 0})
        save_checkpoint_atomic(self.output / 'da_qcfs.pth', cpu_model(self.model))
        parallel_calibration_mode(self.model, False)
        self.adapter.replace_parallel(self.model, 'spike', self.job)
        device_model(self.model)
        save_checkpoint_atomic(self.output / 'converted.pth', cpu_model(self.model))
        atomic_json(self.output / 'conversion.json', {'identity': self.identity, 'configuration': vars(self.job),
            'audit': self.audit, 'sample_ids': self.trace})


def normalization(dataset, direct=False):
    if direct: return {'mean': [.4914, .4822, .4465], 'std': [.2023, .1994, .2010]}
    if dataset == 'CIFAR10': return {'mean': [.4914, .4822, .4465], 'std': [.247, .2435, .2616]}
    if dataset == 'CIFAR100': return {'mean': [.5071, .4866, .4409], 'std': [.2673, .2564, .2762]}
    return None


def run(config, runtime):
    parameters = config['parameters']
    job = SimpleNamespace(**parameters)
    job.seed = int(runtime['seed'])
    job.classes = 11 if job.dataset == 'DVSGesture' else 10 if job.dataset == 'CIFAR10' else 100
    job.is_dcgs = job.converter == 'dcgs-test'
    output = Path(runtime['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(runtime.get('checkpoint') or runtime['inputs']['checkpoint']).resolve()
    source = {'normalization': normalization(job.dataset, job.converter == 'direct')}
    field = 'checkpoint' if job.converter == 'direct' else 'converted' if config['stage'] == 'evaluate' and job.converter == 'parallel' else 'ann'
    source[field] = str(checkpoint)
    if runtime.get('inputs', {}).get('masks'):
        source['masks'] = runtime['inputs']['masks']
    if config['stage'] == 'convert':
        if job.converter == 'parallel':
            configure(2025)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                return ParallelCalibration(output, job, source, runtime['data_root'], runtime.get('resume')).run()
        from DCGS.utils import seed_all
        from DCGS.converter import Converter, Threshold_Getter
        seed_all(2025)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from cnn_evaluation import dataset
        entry = {'job': vars(job), 'kind': 'cnn', 'source': source}
        ds = dataset(entry, runtime['data_root'])
        loader = torch.utils.data.DataLoader(ds, batch_size=job.calibration_batch, shuffle=False, num_workers=0, drop_last=False)
        ann, _ = load_ann(source, job)
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            model = Converter.change_maxpool_before_relu(ann.cuda().eval())
            model = Threshold_Getter(loader, mode='var', level='channel', device='cuda', momentum=.1, output_fx=False)(model)
        save_checkpoint_atomic(output / 'thre.pth', cpu_model(model) if job.method == 'depgraph' else model.cpu().state_dict())
        atomic_json(output / 'calibration.json', {'configuration': parameters, 'samples': len(ds), 'sample_ids': list(range(len(ds)))})
        return
    if job.is_dcgs:
        threshold = runtime.get('inputs', {}).get('threshold')
        if not threshold: raise ValueError('DCGS evaluation requires --input threshold=PATH')
        source['threshold'] = threshold
    if job.converter == 'direct':
        source['braincog_root'] = str(Path(__file__).resolve().parents[2] / 'direct')
    entry = {'id': config['id'] + '-seed-' + str(job.seed), 'job': vars(job), 'source': source,
        'kind': 'direct' if job.converter == 'direct' else 'cnn', 'baseline': parameters.get('baseline'),
        'samples': 288 if job.dataset == 'DVSGesture' else 10000}
    from cnn_evaluation import evaluate
    return evaluate(entry, job.batch, runtime['data_root'], output, False)
