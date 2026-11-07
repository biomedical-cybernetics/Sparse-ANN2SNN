import argparse
import json
import time
import os
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import sys
from pathlib import Path

import logging
import torch
import numpy as np
from scipy.io import savemat
from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *

from prun_and_generation import *
from snn_model import *
from utils import *
from orchestration.runtime import (
    capture_rng_state, load_checkpoint_checked, restore_rng_state, save_checkpoint_atomic
)

seed_parser = argparse.ArgumentParser(description='Seeded SD-SNN runner')
seed_parser.add_argument('--dataset', required=True, choices=['CIFAR10', 'CIFAR100'])
seed_parser.add_argument('--data-root', required=True)
seed_parser.add_argument('--output-dir', required=True)
seed_parser.add_argument('--seed', required=True, type=int)
seed_parser.add_argument('--device', default=0, type=int)
seed_parser.add_argument('--epochs', default=600, type=int)
seed_parser.add_argument('--batch-size', default=50, type=int)
seed_parser.add_argument('--lr', default=5e-3, type=float)
seed_parser.add_argument('--timesteps', default=8, type=int)
seed_parser.add_argument('--checkpoint-interval', default=20, type=int)
seed_parser.add_argument('--resume')
seed_parser.add_argument('--probe-batches', type=int)
seed_args = seed_parser.parse_args()

_logger = logging.getLogger('train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = seed_args.dataset.lower()
num_classes = 10 if dataset == 'cifar10' else 100
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), dataset])
output_dir = get_outdir(seed_args.output_dir, 'train', exp_name)
setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))
_logger.info(exp_name)

config_parser = cfg = argparse.ArgumentParser(description='Training Config', add_help=False)
model='cifar_convnet'
step=seed_args.timesteps
encode='direct'
node_type='PLIFNode'
thresh=0.5
tau=2.0
torch.backends.cudnn.benchmark = True
devicee=seed_args.device
seed=seed_args.seed

channels = 2
lr=seed_args.lr
batch_size=seed_args.batch_size
epochs=seed_args.epochs
linear_scaled_lr = lr * batch_size/ 1024.0
cfg.opt='adamw'
cfg.lr=linear_scaled_lr
cfg.weight_decay=0.01
cfg.momentum=0.9
cfg.epochs=epochs
cfg.sched='cosine'
cfg.min_lr=1e-5
cfg.warmup_lr=1e-6
cfg.warmup_epochs=5
cfg.cooldown_epochs=10
cfg.decay_rate=0.1

epoch_prune = 1
eval_metric='top1'
best_test = 0
best_testepoch = 0
best_testprun = 0
best_testepochprun = 0
spines_num=18

results_dir = seed_args.output_dir
os.makedirs(results_dir, exist_ok=True)
results_mat_path = os.path.join(results_dir, 'res.mat')
best_model_path = os.path.join(results_dir, 'best_model.pth')
epoch_train_times = []
test_top1_history = []
compression_history = []
layer_lengths = None
layer_labels = []
best_top1 = -float('inf')
global_sparsity_history = []
milestone_result_paths = {
    300: os.path.join(results_dir, 'res_epoch300.mat'),
    600: os.path.join(results_dir, 'res_epoch600.mat'),
}


torch.cuda.set_device('cuda:%d' % devicee)
torch.manual_seed(seed)

model = my_cifar_model(step=step,encode_type=encode,node_type=node_type,num_classes=num_classes)
model = model.cuda()
print(model)
optimizer = create_optimizer(cfg, model)
lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % dataset)(
    batch_size=batch_size, step=step, root=seed_args.data_root)

train_loss_fn = UnilateralMse(1.)
validate_loss_fn = UnilateralMse(1.)

m = Mask(model,spines_num)


def persist_results(target_path=None):
    """Persist collected metrics to a MATLAB .mat file for later analysis."""
    if not epoch_train_times:
        return
    mat_payload = {
        'train_time': np.array(epoch_train_times, dtype=np.float32),
        'test_top1': np.array(test_top1_history, dtype=np.float32),
        'global_sparsity': np.array(global_sparsity_history, dtype=np.float32),
    }
    if compression_history:
        mat_payload['compression'] = np.array(compression_history, dtype=np.float32)
    if layer_labels:
        mat_payload['layer_labels'] = np.array(layer_labels, dtype='U64')
    if layer_lengths is not None:
        mat_payload['layer_lengths'] = np.array(layer_lengths, dtype=np.int64)
    savemat(target_path or results_mat_path, mat_payload)

def train_epoch(
        epoch, model, loader, optimizer, loss_fn,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(loader):
        if seed_args.probe_batches is not None and batch_idx >= seed_args.probe_batches:
            break
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        output = model(inputs)

        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx %100 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print("Train: epoch:",epoch,batch_idx,"/",len(loader),"loss:",losses_m.avg,"acc1:", top1_m.avg,"lr:",lr)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            if last_batch or batch_idx %100 == 0:
                print("Test: loss:",losses_m.avg,"acc1:", top1_m.avg)

            batch_time_m.update(time.time() - end)
            end = time.time()


    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


start_epoch = 0
if seed_args.resume:
    checkpoint = load_checkpoint_checked(
        seed_args.resume,
        run_id=os.environ['ANN2SNN_RUN_ID'],
        configuration=json.loads(os.environ['ANN2SNN_CONFIGURATION']),
        seed=seed,
        world_size=int(os.environ.get('ANN2SNN_WORLD_SIZE', '1')),
    )
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng_state(checkpoint['rng_state'])
    start_epoch = int(checkpoint['epoch'])
    best_top1 = float(checkpoint['best_top1'])
    if checkpoint.get('mask') is not None:
        m = checkpoint['mask']
        def restore_to_device(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {key: restore_to_device(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore_to_device(item) for item in value]
            if isinstance(value, tuple):
                return tuple(restore_to_device(item) for item in value)
            return value

        for attribute, value in tuple(m.__dict__.items()):
            if attribute not in ('model', 'feature', 'fc1'):
                setattr(m, attribute, restore_to_device(value))
        m.model = model
        m.feature = model.feature
        m.fc1 = model.fc_prun[0]
    histories = checkpoint.get('histories', {})
    epoch_train_times[:] = histories.get('train_time', [])
    test_top1_history[:] = histories.get('test_top1', [])
    compression_history[:] = histories.get('compression', [])
    global_sparsity_history[:] = histories.get('global_sparsity', [])

for epoch in range(start_epoch, epochs):

    epoch_start_time = time.time()
    train_metrics= train_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn,
        lr_scheduler=lr_scheduler)
    epoch_train_times.append(time.time() - epoch_start_time)
    if seed_args.probe_batches is not None:
        print(f'Probe completed after {seed_args.probe_batches} batches.')
        break

    if epoch==0:
        m.init_length()
        if layer_lengths is None:
            layer_lengths = [m.model_length[idx] for idx in m.convlayer]
            layer_labels = [f'layer_{layer_idx}_idx_{idx}' for layer_idx, idx in enumerate(m.convlayer)]
    if epoch>0:
        m.model = model
        m.init_mask_dsd()
        if epoch>spines_num:
            matt=m.do_mask_dsd()
        if epoch>2*spines_num:
            m.do_growth_ww(epoch)
            matt=m.do_pruning_dsd(epoch)
        model = m.model
    cc=m.if_zero()
    if layer_lengths is None:
        layer_lengths = [m.model_length[idx] for idx in m.convlayer]
        layer_labels = [f'layer_{layer_idx}_idx_{idx}' for layer_idx, idx in enumerate(m.convlayer)]
    layer_sparsity = [zero_cnt / total if total else 0. for zero_cnt, total in zip(cc, layer_lengths)]
    compression_history.append(layer_sparsity)
    total_weights = float(sum(layer_lengths)) if layer_lengths is not None else 0.
    global_zero = float(sum(cc))
    global_sparsity = global_zero / total_weights if total_weights else 0.
    global_sparsity_history.append(global_sparsity)

    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    test_top1_history.append(top1)
    print('Epoch {0}: test acc {1:.4f}, global sparsity {2:.4f}'.format(epoch, top1, global_sparsity))
    if top1 > best_top1:
        best_top1 = top1
        torch.save(model.state_dict(), best_model_path)
        print('New best top1 {:.4f} at epoch {}'.format(best_top1, epoch))
    if top1 > best_testprun:
        best_testprun = top1
        best_testepochprun =epoch
    if epoch%40==0:
        print('best acc:',best_testprun,'best epoch:',best_testepochprun)
    if epoch>4:
        _logger.info('*** epoch: {0} (pruning rate {1},acc:{2})'.format(epoch, cc,top1))

    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    persist_results()
    current_epoch = epoch + 1
    if current_epoch in milestone_result_paths:
        persist_results(target_path=milestone_result_paths[current_epoch])
    if current_epoch % seed_args.checkpoint_interval == 0 or current_epoch == epochs:
        payload = {
            'run_id': os.environ['ANN2SNN_RUN_ID'],
            'configuration': json.loads(os.environ['ANN2SNN_CONFIGURATION']),
            'seed': seed,
            'world_size': int(os.environ.get('ANN2SNN_WORLD_SIZE', '1')),
            'epoch': current_epoch,
            'best_top1': best_top1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'mask': m,
            'rng_state': capture_rng_state(),
            'histories': {
                'train_time': epoch_train_times,
                'test_top1': test_top1_history,
                'compression': compression_history,
                'global_sparsity': global_sparsity_history,
            },
        }
        if os.environ.get('ANN2SNN_RETAIN_PERIODIC') == '1':
            save_checkpoint_atomic(
                os.path.join(results_dir, 'checkpoints', f'epoch-{current_epoch:04d}.pth'),
                payload,
            )
        save_checkpoint_atomic(os.path.join(results_dir, 'last_checkpoint.pth'), payload)

if seed_args.probe_batches is None:
    torch.save(model.state_dict(), os.path.join(results_dir, 'final_model.pth'))
