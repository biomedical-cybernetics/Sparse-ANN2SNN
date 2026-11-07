import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import sys
from pathlib import Path

def load_data_imagenet(args, data_root):
    # Temp add deit-main to sys.path to import its modules

    from image_dataset import build_dataset

    # Ensure args has necessary fields for build_dataset if not provided
    if not hasattr(args, 'data_path'): args.data_path = data_root
    if not hasattr(args, 'data_set'): args.data_set = 'IMNET'
    if not hasattr(args, 'input_size'): args.input_size = 224
    if not hasattr(args, 'eval_crop_ratio'): args.eval_crop_ratio = 0.875

    dataset_train, args.nb_classes = None, 1000
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        #        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #    )
        #        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #    )
        sampler_train = None
        if getattr(args, 'dist_eval', False):
            if len(dataset_val) % num_tasks != 0:
                raise ValueError('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_train = None
        sampler_val = SequentialSampler(dataset_val)

    #    dataset_train, sampler=sampler_train,
    data_loader_train = None

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.bs),
        num_workers=getattr(args, 'num_workers', 16),
        pin_memory=getattr(args, 'pin_mem', True),
        drop_last=False
    )

    return data_loader_train, data_loader_val, args.nb_classes
