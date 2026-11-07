# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
import os

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator


import utils
from opts import cht_opts
from dst_scheduler import DSTScheduler
from dst_util import recur_split_atten
from runtime import capture_rng_state, restore_rng_state, save_checkpoint_atomic
from precision import build_loss_scaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def timing_start(args):
    if not args.timing_components:
        return None
    torch.cuda.synchronize()
    if args.distributed:
        torch.distributed.barrier()
    return time.perf_counter()


def timing_stop(args, started):
    if started is None:
        return None
    torch.cuda.synchronize()
    if args.distributed:
        torch.distributed.barrier()
    return time.perf_counter() - started


def timing_memory_snapshot(args, device):
    """Return per-rank CUDA allocator peaks for the immediately preceding phase."""
    if not args.timing_components or device.type != 'cuda':
        return []
    torch.cuda.synchronize(device)
    local = {
        'rank': utils.get_rank(),
        'allocated_gib': round(torch.cuda.memory_allocated(device) / 1024**3, 4),
        'reserved_gib': round(torch.cuda.memory_reserved(device) / 1024**3, 4),
        'max_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
        'max_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 1024**3, 4),
    }
    if not args.distributed:
        return [local]
    gathered = [None for _ in range(utils.get_world_size())]
    torch.distributed.all_gather_object(gathered, local)
    return gathered


def get_args_parser():
    #basic parameters
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--amp-dtype', choices=('fp16', 'bf16'), default='fp16')
    parser.add_argument(
        '--timing-stop-after-epochs', default=None, type=int,
        help='Stop a timing probe early while retaining the original schedule horizon.',
    )
    parser.add_argument(
        '--timing-force-first-dst-update', action='store_true',
        help='Move only the DST counter so a one-epoch timing probe performs one update.',
    )
    parser.add_argument(
        '--timing-cht-six-epoch-plan', action='store_true',
        help='',
    )
    parser.add_argument(
        '--timing-force-interval-checkpoint', action='store_true',
        help='Timing only: write one interval checkpoint on the final short-probe epoch.',
    )
    parser.add_argument(
        '--timing-components', action='store_true',
        help='Record synchronized train, evaluation, checkpoint, and topology timings.',
    )
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    #cht parameters
    cht_opts(parser)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true') #3augment

    parser.add_argument('--src', action='store_true') #simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")#NOTE
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument('--cosub', action='store_true')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--probe-batches', default=None, type=int,
                        help='Smoke only: cap the number of training batches.')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Enable bounded, non-formal training/evaluation smoke behavior.')
    parser.add_argument('--smoke-eval-batches', default=1, type=int,
                        help='Smoke only: cap the number of evaluation batches.')
    parser.add_argument('--checkpoint-interval', default=20, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def set_device(args):
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))

    if getattr(args, "device", None):
        # 用户显式指定 --device，优先使用
        device = torch.device(args.device)
    else:
        # 否则按 local_rank 分配逻辑 GPU（被 CUDA_VISIBLE_DEVICES 重映射过的）
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)   # 把当前进程指向该 CUDA 设备
        else:
            device = torch.device("cpu")

    # 打印映射信息，便于调试
    print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("LOCAL_RANK =", local_rank)
    print("Using device =", device)
    if device.type == "cuda":
        # 打印逻辑 0 的名字和物理 id（如果你想看物理 id）
        print("torch.cuda.device_count() =", torch.cuda.device_count())
        print("torch.cuda.get_device_name(torch.cuda.current_device()) =", torch.cuda.get_device_name(torch.cuda.current_device()))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            phys = visible.split(",")
            if local_rank < len(phys):
                print(f"local_rank {local_rank} -> physical GPU id {phys[local_rank]}")
    return device, local_rank

def main(args):
    if args.smoke_test != (args.probe_batches is not None):
        raise ValueError('--smoke-test and --probe-batches must be used together')
    if args.smoke_test and (args.probe_batches < 1 or args.smoke_eval_batches < 1):
        raise ValueError('smoke-test batch limits must be positive')
    if args.timing_cht_six_epoch_plan:
        if not args.timing_components or args.sparsity <= 0:
            raise ValueError('--timing-cht-six-epoch-plan requires --timing-components and sparse CHT')
        if args.timing_stop_after_epochs != 6 or args.epochs != 300:
            raise ValueError('--timing-cht-six-epoch-plan requires --epochs 300 and --timing-stop-after-epochs 6')
        if args.timing_force_first_dst_update:
            raise ValueError('the CHT six-epoch timing plan cannot combine with --timing-force-first-dst-update')
    if args.timing_force_interval_checkpoint and not args.timing_components:
        raise ValueError('--timing-force-interval-checkpoint requires --timing-components')
    #==============================preset==============================#
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device, local_rank = set_device(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    # random.seed(seed)

    cudnn.benchmark = True

    #==============================dataset==============================#
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    #==============================model==============================#
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )
    if (not args.eval) and args.distillation_type!='none':
        model.set_distilled_training(True)

    #==============================split attention to do chain removal==============================#
    if args.sparsity>0.0:
        recur_split_atten(model)

    #==============================Load state dict==============================#
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    #=========================preparation, e.g. distribution + optimizer scheduler=========================#
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = build_loss_scaler(args.amp_dtype)

    #==============================pruner==============================#
    if args.sparsity>0.0:
        if args.adaptive_zeta and args.EM_S:
            raise ValueError("Adaptive zeta and EM_S cannot be used together")


        pdevice = next(model_without_ddp.parameters()).device
        if args.mask:
            backward_masks = [x.to(pdevice) for x in backward_masks]
        else:
            backward_masks = None

        if args.pruner_statedict:
            pstatedict = torch.load(args.pruner_statedict, map_location=pdevice)
            if 'ANN2SNN_METHOD' in os.environ:
                expected_pruner_identity = {
                    'run_id': os.environ.get(
                        'ANN2SNN_RESUME_RUN_ID', os.environ['ANN2SNN_RUN_ID']
                    ),
                    'configuration': json.loads(os.environ['ANN2SNN_CONFIGURATION']),
                    'seed': args.seed,
                    'world_size': utils.get_world_size(),
                    'method': os.environ['ANN2SNN_METHOD'],
                }
                mismatches = {
                    key: (pstatedict.get('checkpoint_identity', {}).get(key), value)
                    for key, value in expected_pruner_identity.items()
                    if pstatedict.get('checkpoint_identity', {}).get(key) != value
                }
                if mismatches:
                    raise ValueError(f"resume pruner identity mismatch: {mismatches}")
        else:
            pstatedict = None

        T_end = args.epochs * args.dst_end_epoch_fraction * len(data_loader_train)

        pruner = DSTScheduler(
            model,
            optimizer,
            alpha = args.zeta,
            delta = args.update_interval * len(data_loader_train),
            static_topo = False,
            T_end = T_end,
            ignore_linear_layers = False,
            sparsity_distribution = args.sparsity_distribution,
            grad_accumulation_n = 1,
            args = args,
            state_dict = pstatedict,
            backward_masks = backward_masks #自己加的，可忽略
        )

        if args.timing_force_first_dst_update:
            if args.update_interval != 5:
                raise ValueError(
                    'timing-force-first-dst-update requires update_interval=5'
                )
            forced_calls = args.probe_batches if args.smoke_test else len(data_loader_train)
            pruner.step = pruner.delta_T - forced_calls
            print(
                'Timing-only DST counter initialized at '
                f'{pruner.step}/{pruner.delta_T}; one update is expected.',
                flush=True,
            )

        print(pruner)
    else:
        pruner = None

    lr_scheduler, _ = create_scheduler(args, optimizer)

    #==============================teacher -> distillation loss==============================#
    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    #==============================if resume, can be ignore==============================#
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            expected_identity = {
                'run_id': os.environ.get('ANN2SNN_RESUME_RUN_ID', os.environ['ANN2SNN_RUN_ID']),
                'configuration': json.loads(os.environ['ANN2SNN_CONFIGURATION']),
                'seed': args.seed,
                'world_size': utils.get_world_size(),
            }
            if 'ANN2SNN_METHOD' in os.environ:
                expected_identity['method'] = os.environ['ANN2SNN_METHOD']
            mismatches = {
                key: (checkpoint.get(key), value)
                for key, value in expected_identity.items()
                if checkpoint.get(key) != value
            }
            if mismatches:
                raise ValueError(f"resume checkpoint identity mismatch: {mismatches}")
            rank_states = checkpoint.get('rng_states_by_rank')
            if rank_states is None or len(rank_states) != utils.get_world_size():
                raise ValueError('resume checkpoint lacks per-rank RNG states')
            restore_rng_state(rank_states[utils.get_rank()])
        lr_scheduler.step(args.start_epoch)

    #==============================evaluation and then exit==============================#
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.amp_dtype)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    #==============================training==============================#
    print(f"Start training for {args.epochs} epochs")
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step(epoch)
        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.timing_components and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        timing_stage = 'ordinary'
        if args.timing_cht_six_epoch_plan:
            # The six-epoch timing probe covers warmup, pre-DST, active-DST,
            # two topology updates, and post-DST states.
            if epoch < 2:
                dst_start_pruner = None
                timing_stage = 'warmup' if epoch == 0 else 'pre_dst'
            elif epoch == 2:
                dst_start_pruner = pruner
                timing_stage = 'active_dst'
            elif epoch in (3, 4):
                dst_start_pruner = pruner
                updates_before_epoch = epoch - 2
                pruner.step = updates_before_epoch * pruner.delta_T - len(data_loader_train)
                timing_stage = 'topology_update'
            else:
                dst_start_pruner = pruner
                pruner.step = pruner.T_end
                timing_stage = 'post_dst'
        else:
            dst_start_pruner = pruner if epoch >= args.dst_start_epoch else None

        train_started = timing_start(args)
        dst_steps_before = pruner.dst_steps if dst_start_pruner is not None else 0
        train_stats = train_one_epoch(
            model, dst_start_pruner, criterion, data_loader_train,
            optimizer, lr_scheduler, args.iterative_warmup_steps, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )
        train_seconds = timing_stop(args, train_started)
        train_memory = timing_memory_snapshot(args, device)
        if args.timing_components and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        if args.timing_force_first_dst_update:
            updates = pruner.dst_steps - dst_steps_before
            print(f'Timing-only DST updates in epoch {epoch + 1}: {updates}', flush=True)
            if updates != 1:
                raise RuntimeError(
                    f'forced timing probe expected one DST update; observed {updates}'
                )
        local_rng_state = capture_rng_state()
        if args.distributed:
            rng_states_by_rank = [None for _ in range(utils.get_world_size())]
            torch.distributed.all_gather_object(rng_states_by_rank, local_rng_state)
        else:
            rng_states_by_rank = [local_rng_state]
        checkpoint_identity = {
            'run_id': os.environ['ANN2SNN_RUN_ID'],
            'configuration': json.loads(os.environ['ANN2SNN_CONFIGURATION']),
            'seed': args.seed,
            'world_size': utils.get_world_size(),
            'rng_states_by_rank': rng_states_by_rank,
        }
        if 'ANN2SNN_METHOD' in os.environ:
            checkpoint_identity['method'] = os.environ['ANN2SNN_METHOD']

        checkpoint_seconds = 0.0
        rolling_checkpoint_seconds = 0.0
        interval_checkpoint_seconds = 0.0
        best_checkpoint_seconds = 0.0
        if args.output_dir:
            checkpoint_started = timing_start(args)
            checkpoint_path = output_dir / 'checkpoint.pth'
            checkpoint_payload = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
                **checkpoint_identity,
            }
            utils.save_on_master(checkpoint_payload, checkpoint_path)
            current_epoch = epoch + 1
            if pruner:
                pruner_path = output_dir / 'pruner.pth'
                pruner_payload = pruner.state_dict()
                pruner_payload.update({
                    'checkpoint_identity': checkpoint_identity,
                    'epoch': epoch,
                })
                utils.save_on_master(pruner_payload, pruner_path)
            measured = timing_stop(args, checkpoint_started)
            if measured is not None:
                rolling_checkpoint_seconds += measured
            if (
                current_epoch % args.checkpoint_interval == 0
                or current_epoch == args.epochs
                or (args.timing_force_interval_checkpoint and current_epoch == args.timing_stop_after_epochs)
            ):
                interval_started = timing_start(args)
                if utils.is_main_process():
                    save_checkpoint_atomic(
                        output_dir / "checkpoints" / f"epoch-{current_epoch:04d}.pth",
                        checkpoint_payload,
                    )
                measured = timing_stop(args, interval_started)
                if measured is not None:
                    interval_checkpoint_seconds += measured
            checkpoint_seconds = rolling_checkpoint_seconds + interval_checkpoint_seconds
        checkpoint_memory = timing_memory_snapshot(args, device)
        if args.timing_components and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        evaluation_started = timing_start(args)
        test_stats = evaluate(
            data_loader_val,
            model,
            device,
            args.amp_dtype,
            max_batches=args.smoke_eval_batches if args.smoke_test else None,
        )
        evaluation_seconds = timing_stop(args, evaluation_started)
        evaluation_memory = timing_memory_snapshot(args, device)
        print(f"Epoch {epoch+1} Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%.")

        #==============================save model, etc==============================#
        best_checkpoint_written = False
        if max_accuracy < test_stats["acc1"] or args.smoke_test:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                best_checkpoint_written = True
                best_checkpoint_started = timing_start(args)
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        **checkpoint_identity,
                    }, checkpoint_path)
                if pruner:
                    best_pruner_path = output_dir / 'best_pruner.pth'
                    best_pruner_payload = pruner.state_dict()
                    best_pruner_payload.update({
                        'checkpoint_identity': checkpoint_identity,
                        'epoch': epoch,
                    })
                    utils.save_on_master(best_pruner_payload, best_pruner_path)
                measured = timing_stop(args, best_checkpoint_started)
                if measured is not None:
                    best_checkpoint_seconds += measured
                    checkpoint_seconds += measured

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print(f'Epoch {epoch+1} Training time {epoch_total_time_str}' , flush=True)

        if args.timing_components and args.output_dir and utils.is_main_process():
            topology_seconds = float(train_stats.get('topology_update_seconds', 0.0))
            timing_record = {
                'schema_version': 1,
                'epoch': epoch,
                'amp_dtype': args.amp_dtype,
                'epoch_total_seconds': epoch_total_time,
                'train_loop_seconds': train_seconds,
                'train_excluding_topology_seconds': train_seconds - topology_seconds,
                'evaluation_seconds': evaluation_seconds,
                'checkpoint_seconds': checkpoint_seconds,
                'rolling_checkpoint_seconds': rolling_checkpoint_seconds,
                'interval_checkpoint_seconds': interval_checkpoint_seconds,
                'best_checkpoint_seconds': best_checkpoint_seconds,
                'best_checkpoint_written': best_checkpoint_written,
                'topology_update_seconds': topology_seconds,
                'topology_update_count': int(train_stats.get('topology_update_count', 0)),
                'timing_stage': timing_stage,
                'train_memory': train_memory,
                'topology_memory': train_stats.get('topology_memory', []),
                'checkpoint_memory': checkpoint_memory,
                'evaluation_memory': evaluation_memory,
            }
            with (output_dir / 'timing.jsonl').open('a') as handle:
                handle.write(json.dumps(timing_record, sort_keys=True) + '\n')

        if args.timing_stop_after_epochs is not None and epoch + 1 >= args.timing_stop_after_epochs:
            print(f'Timing-only stop after {epoch + 1} completed epochs.', flush=True)
            break
        if args.smoke_test:
            print(
                f'Smoke test completed after {args.probe_batches} training batches '
                f'and {args.smoke_eval_batches} evaluation batches.',
                flush=True,
            )
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import faulthandler
    from torch.distributed.elastic.multiprocessing.errors import record

    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    record(main)(args)
