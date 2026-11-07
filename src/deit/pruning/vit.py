
import argparse
import logging
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dense"))

import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

# Import prune functions - assuming prune_vit.py is in lib/
from lib.prune_vit import prune_ria_vit, prune_sparsegpt_vit, prune_magnitude_vit

import utils
from datasets import build_dataset
from engine import evaluate
from dst_scheduler import DSTScheduler
from dst_util import recur_split_atten, get_W
import opts

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser('ViT Pruning script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--ckpt', default='', help='resume from checkpoint')
    parser.add_argument('--save_dir', type=str)

    # Pruning parameters
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    # Use same choices as RIA main.py for compatibility, but map to implemented functions
    parser.add_argument("--prune_method", type=str, default="ria",
                        choices=["magnitude", "ri", "wanda", "sparsegpt", "ria"],
                        help="Pruning method")
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--calib_dataset', type=str, default="./data/test", help='Calibration dataset path')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--probe-batches', type=int, default=None,
                        help='Probe only: limit the evaluation dataset to this many batches.')
    parser.add_argument("--per_outneuron", action="store_true", help="pruning per outneuron. Wanda's tactic.")
    parser.add_argument("--reconstruction", action="store_true", help="remaining weight reconstruction based on sparsegpt")

    # RIA specific (from RIA/main.py)
    parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")

    # Dataset parameters (for evaluation)
    parser.add_argument('--data-path', default='./data/', type=str, help='dataset path')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')

    # Distributed training parameters (required by deit utils)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Others required by build_dataset or evaluate
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    #args required but not involved in logic
    parser.add_argument('--gptq', action="store_true")
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

    return parser

def set_device(args):
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0)))

    if getattr(args, "device", None) and args.device != 'cuda':
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
    print(f"[Process {local_rank}] ENV CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Process {local_rank}] Using PyTorch device object = {device}")

    if device.type == "cuda":
        # 获取当前设备属性
        props = torch.cuda.get_device_properties(device)
        print(f"[Process {local_rank}] Active GPU: {props.name}, Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"[Process {local_rank}] torch.cuda.device_count() = {torch.cuda.device_count()}")

    return device, local_rank

def main(args):
    utils.init_distributed_mode(args)

    device, local_rank = set_device(args)

    # fixes the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # 1. Load Model
    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )


    if args.ckpt:
        logger.info(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=True)

    recur_split_atten(model)

    model.to(device)

    # 2. Pruning
    if args.sparsity_ratio > 0:
        # Only prune on the main process to avoid redundant computation and ensuring consistency
        if utils.is_main_process():
            logger.info(f"Starting pruning with method: {args.prune_method}")
            logger.info(f"Sparsity ratio: {args.sparsity_ratio}")

            # Pruning functions usually handle 'model' in-place.
            # Ensure model is in eval mode
            model.eval()

            if args.prune_method in ["wanda", "ria"]:
                prune_ria_vit(args, model, device=device)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt_vit(args, model, device=device)
            elif args.prune_method in ["magnitude", "ri"]:
                prune_magnitude_vit(args, model, device=device)
            else:
                raise NotImplementedError(f"Pruning method {args.prune_method} not implemented for ViT script yet.")

            logger.info("Pruning finished.")

        # Wait for Rank 0 to finish pruning
        if args.distributed:
            torch.distributed.barrier()

    # 3. Distributed Data Parallel
    model_without_ddp = model
    if args.distributed:
        # DDP will broadcast Rank 0's model parameters (which are now pruned) to all other ranks
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 4. Evaluation
    logger.info("Creating dataset for evaluation")
    dataset_val, _ = build_dataset(is_train=False, args=args)
    if args.probe_batches is not None:
        probe_items = max(1, int(1.5 * args.batch_size) * args.probe_batches)
        dataset_val = torch.utils.data.Subset(dataset_val, range(min(len(dataset_val), probe_items)))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    logger.info("Starting evaluation...")
    test_stats = evaluate(data_loader_val, model, device)
    logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    # 5. Save Pruned Model & DSTScheduler
    if utils.is_main_process():
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to {save_dir}")
        torch.save({
            'model': model_without_ddp.state_dict(),
            'epoch': 300,
            'args': args,
        }, save_dir / "ckpt.pth")


        # save DSTScheduler
        W_refs, _, _ = get_W(model_without_ddp, 0)

        backward_masks = []
        for w in W_refs:
            mask = (w != 0).float().detach() # Create mask from pruned weights
            backward_masks.append(mask)

        torch.save(backward_masks, save_dir / "mask.pth")

        logger.info(f"Finish.")

if __name__ == '__main__':
    import numpy as np

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
