# Sparse ANN-to-SNN conversion

Code for **Near-Dense Accuracy and Energy-Efficient Sparse SNNs via One-Shot ANN-to-SNN Conversion**.

## Installation

```bash
conda env create -f environment.yml
conda activate sparse-ann-snn
python third_party/conversion/prepare.py
python third_party/deit/setup_ria.py
python run.py --list
```

## Data

Set `--data-root` to:

- **CIFAR-10/100:** the directory containing the torchvision CIFAR archives or
  extracted `cifar-10-batches-py` and `cifar-100-python` directories.
- **DVS-Gesture:** the parent of `dvsgesture/`, prepared in the SpikingJelly layout.
- **ImageNet-1K:** a directory containing `train/<class>/...` and `test/<class>/...`;
  `test` contains the ImageNet validation split.

```bash
python run.py train --config configs/cnn/resnet18-cifar10-ours.json --check-config
```

## CNN training

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train \
  --config configs/cnn/resnet18-cifar10-ours.json \
  --data-root /path/to/cifar --output-dir outputs/resnet18-ours-s15 --seed 15
```

Choose the configuration for each experiment below. Filenames are relative to
`configs/cnn/`; braces list alternatives. Repeat the command for seeds 15, 16 and 17.

| Experiment                 | Configuration                                                                   |
| -------------------------- | ------------------------------------------------------------------------------- |
| ResNet-18, CIFAR           | `resnet18-{cifar10,cifar100}-{dense,ours,depgraph}.json`                      |
| VGG-16, CIFAR              | `vgg16-{cifar10,cifar100}-{dense,ours,depgraph}.json`                         |
| ResNet-18, DVS-Gesture     | `resnet18-dvsgesture-{dense,ours}.json`                                       |
| DST comparisons            | `resnet18-{cifar100,dvsgesture}-{set-channel,rigl-channel,mest-channel}.json` |
| Sparsity ablation          | `resnet18-{cifar10,cifar100}-ours-{s60,s80}.json`                             |
| Matched direct-SNN budgets | `6conv2fc-{cifar10,cifar100}-ours-{dpap,sd}-budget.json`                      |

### Topology controls

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train \
  --config configs/cnn/resnet18-cifar10-cht-fixed.json \
  --input ours_output=outputs/resnet18-ours-s15 \
  --data-root /path/to/cifar --output-dir outputs/cht-fixed-s15 --seed 15

CUDA_VISIBLE_DEVICES=0 python run.py prune \
  --config configs/cnn/resnet18-cifar10-magnitude-layer.json \
  --input ours_output=outputs/resnet18-ours-s15 \
  --input dense_output=outputs/resnet18-dense-s15 \
  --data-root /path/to/cifar --output-dir outputs/magnitude-layer-s15 --seed 15
```

| Control                 | Configuration in `configs/cnn/`                 | Stage     |
| ----------------------- | ------------------------------------------------- | --------- |
| CHT fixed               | `resnet18-cifar10-cht-fixed.json`               | `train` |
| Random fixed            | `resnet18-cifar10-random-fixed-layer.json`      | `train` |
| Degree-preserving fixed | `resnet18-cifar10-degree-preserving-fixed.json` | `train` |
| Random DST              | `resnet18-cifar10-random-dst-fixed70.json`      | `train` |
| One-shot magnitude      | `resnet18-cifar10-magnitude-layer.json`         | `prune` |

Use preceding training outputs from the same seed.

## Direct SNN training

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train \
  --config configs/direct/dpap-cifar10.json \
  --data-root /path/to/cifar --output-dir outputs/dpap-cifar10-s42 --seed 42
```

| Method | Configuration in `configs/direct/` | Seeds      |
| ------ | ------------------------------------ | ---------- |
| DPAP   | `dpap-{cifar10,cifar100}.json`     | 42, 43, 44 |
| SD-SNN | `sd-snn-{cifar10,cifar100}.json`   | 36, 37, 38 |

## DeiT training and pruning

### Training

```bash
torchrun --standalone --nproc_per_node=8 run.py train \
  --config configs/deit/deit-imagenet-dense.json \
  --data-root /path/to/imagenet --output-dir outputs/deit-dense-s0 --seed 0

torchrun --standalone --nproc_per_node=8 run.py train \
  --config configs/deit/deit-imagenet-ours.json \
  --data-root /path/to/imagenet --output-dir outputs/deit-ours-s0 --seed 0
```

Repeat for seeds 0, 1 and 2.

### Pruning

```bash
torchrun --standalone --nproc_per_node=4 run.py prune \
  --config configs/deit/deit-imagenet-sparsegpt.json \
  --checkpoint outputs/deit-dense-s0/best_checkpoint.pth \
  --data-root /path/to/imagenet --output-dir outputs/deit-sparsegpt-s0 --seed 0
```

| Method                 | Configuration in `configs/deit/`          |
| ---------------------- | ------------------------------------------- |
| SparseGPT              | `deit-imagenet-sparsegpt.json`            |
| RIA                    | `deit-imagenet-ria.json`                  |
| RIA + reconstruction   | `deit-imagenet-ria-reconstruction.json`   |
| Wanda                  | `deit-imagenet-wanda.json`                |
| Wanda + reconstruction | `deit-imagenet-wanda-reconstruction.json` |
| RI                     | `deit-imagenet-ri.json`                   |
| Magnitude              | `deit-imagenet-magnitude.json`            |

### SparseGPT fine-tuning

```bash
torchrun --standalone --nproc_per_node=8 run.py train \
  --config configs/deit/deit-imagenet-sparsegpt-finetuned.json \
  --checkpoint outputs/deit-sparsegpt-s0/ckpt.pth \
  --input mask=outputs/deit-sparsegpt-s0/mask.pth \
  --data-root /path/to/imagenet --output-dir outputs/deit-sparsegpt-ft-s0 --seed 0
```

For training recovery, add `--resume PATH`. DeiT CHT also needs its matching
pruner state; use `--input pruner_checkpoint=PATH` if it is not beside the checkpoint.

## CNN conversion and evaluation

### DCGS

```bash
CUDA_VISIBLE_DEVICES=0 python run.py convert \
  --config configs/conversion/resnet-18-cifar10-ours-dcgs-test-convert.json \
  --checkpoint outputs/resnet18-ours-s15/artifacts/best_model.pth \
  --input masks=outputs/resnet18-ours-s15/artifacts/best_masks.pth \
  --data-root /path/to/cifar --output-dir outputs/resnet18-ours-s15-dcgs --seed 15

CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --config configs/conversion/resnet-18-cifar10-ours-dcgs-test-t8-evaluate.json \
  --checkpoint outputs/resnet18-ours-s15/artifacts/best_model.pth \
  --input threshold=outputs/resnet18-ours-s15-dcgs/thre.pth \
  --data-root /path/to/cifar --output-dir outputs/resnet18-ours-s15-dcgs-t8 --seed 15
```

### Parallel Conversion

```bash
CUDA_VISIBLE_DEVICES=0 python run.py convert \
  --config configs/conversion/resnet-18-cifar10-ours-parallel-16-convert.json \
  --checkpoint outputs/resnet18-ours-s15/artifacts/best_model.pth \
  --input masks=outputs/resnet18-ours-s15/artifacts/best_masks.pth \
  --data-root /path/to/cifar --output-dir outputs/resnet18-ours-s15-parallel16 --seed 15

CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --config configs/conversion/resnet-18-cifar10-ours-parallel-t16-evaluate.json \
  --checkpoint outputs/resnet18-ours-s15-parallel16/converted.pth \
  --data-root /path/to/cifar --output-dir outputs/resnet18-ours-s15-parallel16-eval --seed 15
```

Select the matching model, dataset, method and time configuration from
`configs/conversion/`:

| Evaluation                     | Configuration suffix              |
| ------------------------------ | --------------------------------- |
| DCGS calibration               | `-dcgs-test-convert.json`       |
| DCGS accuracy and Energy       | `-dcgs-test-t<T>-evaluate.json` |
| Parallel calibration           | `-parallel-<T>-convert.json`    |
| Parallel accuracy and Energy   | `-parallel-t<T>-evaluate.json`  |
| Direct SNN accuracy and Energy | `-direct-t8-evaluate.json`      |

### Direct SNN evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --config configs/conversion/6conv2fc-cifar10-dpap-dpap-direct-t8-evaluate.json \
  --checkpoint outputs/dpap-cifar10-s42/best_model.pth \
  --data-root /path/to/cifar --output-dir outputs/dpap-cifar10-s42-eval --seed 42
```

For SD-SNN, use `6conv2fc-cifar10-sd-snn-sd-direct-t8-evaluate.json` and its trained
checkpoint; replace `cifar10` with `cifar100` for CIFAR-100.

## DeiT conversion and evaluation

```bash
torchrun --standalone --nproc_per_node=4 run.py convert \
  --config configs/conversion/deit-dcgs-threshold.json \
  --checkpoint outputs/deit-ours-s0/best_checkpoint.pth \
  --data-root /path/to/imagenet --output-dir outputs/deit-ours-s0-dcgs --seed 0

torchrun --standalone --nproc_per_node=4 run.py evaluate \
  --config configs/conversion/deit-dcgs-accuracy.json \
  --checkpoint outputs/deit-ours-s0-dcgs/thre.pth \
  --data-root /path/to/imagenet --output-dir outputs/deit-ours-s0-accuracy --seed 0

torchrun --standalone --nproc_per_node=4 run.py evaluate \
  --config configs/conversion/deit-dcgs-energy.json \
  --checkpoint outputs/deit-ours-s0-dcgs/thre.pth \
  --data-root /path/to/imagenet --output-dir outputs/deit-ours-s0-energy --seed 0
```

Use the same commands with each method's checkpoint and a separate output directory.

| Models                          | Calibration                                 | Accuracy                                   | Energy                                   |
| ------------------------------- | ------------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| Dense, Ours and pruning methods | `deit-dcgs-threshold.json`                | `deit-dcgs-accuracy.json`                | `deit-dcgs-energy.json`                |
| SparseGPT fine-tuned            | `deit-sparsegpt-ft21-dcgs-threshold.json` | `deit-sparsegpt-ft21-dcgs-accuracy.json` | `deit-sparsegpt-ft21-dcgs-energy.json` |

Configuration filenames above are relative to `configs/conversion/`.

## Training-cost benchmark

```bash
python run.py benchmark --config configs/benchmark/deit-cost.json \
  --gpus 0,1,2,3,4,5,6,7 --seed 0 \
  --data-root /path/to/imagenet --output-dir outputs/deit-cost-s0
```

Repeat for seeds 0, 1 and 2.
