import numpy as np
import os, random
from os.path import join
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from .augment import EventAugment
DVS_DATASETS = {'dvsgesture': {'subdir':'dvsgesture','in_channels':18,'outdim':11}}

def _resolve_dataloader_workers(default=8):
    env_workers = os.getenv("SNN_DVS_NUM_WORKERS")
    if env_workers is not None:
        try:
            return max(0, int(env_workers))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, min(default, cpu_count // 2))

def _worker_init_fn(_worker_id):
    # Avoid CPU oversubscription from intra-op threading in each worker process.
    torch.set_num_threads(1)

def DVSGesture(root, resolution=(128, 128), num_classes=11, one_hot_label=True):
    train_set = DVS128Gesture(root, train = True, data_type="event")
    test_set = DVS128Gesture(root, train = False, data_type="event")

    return SpikingjellyDataset(train_set, True, resolution=resolution, num_classes=num_classes, one_hot_label=one_hot_label), SpikingjellyDataset(test_set, False,
                                                                                            resolution=resolution, num_classes=num_classes, one_hot_label=one_hot_label)

class SpikingjellyDataset:

    def __init__(self, dataset, train, resolution, num_classes=None, one_hot_label=False):
        self.dataset = dataset
        self.num_classes = num_classes
        self.one_hot_label = one_hot_label
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None
        self.quantization_layer = QuantizationLayerVoxGrid((9, 128, 128))
        self.crop_dimension = (224, 224)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dict_events, label = self.dataset[idx]

        x = dict_events['x'].astype(np.float32)
        y = dict_events['y'].astype(np.float32)
        t = dict_events['t'].astype(np.float32)
        p = dict_events['p'].astype(np.float32)

        # Keep 4-column events for augmentation compatibility, then append batch index.
        events_np = np.empty((x.shape[0], 4), dtype=np.float32)
        events_np[:, 0] = x
        events_np[:, 1] = y
        events_np[:, 2] = t
        events_np[:, 3] = p
        events = torch.from_numpy(events_np)
        # print(events.shape)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events = torch.cat([events, torch.zeros((len(events), 1), dtype=events.dtype)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = vox
        events = events.squeeze(0)

        if self.one_hot_label:
            if self.num_classes is None:
                raise ValueError("num_classes must be provided when one_hot_label=True")
            label = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).to(torch.float32)

        return events, label

    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y

class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3
        B = int(1 + events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T
        # normalizing timestamps
        # tqdm.write("-------------bi shape----------------")
        for bi in range(B):
            # tqdm.write(str(t[events[:, -1] == bi].shape))
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        for i_bin in range(C):
            values = torch.zeros_like(t)
            values[(t > i_bin / C) & (t <= (i_bin + 1) / C)] = 1

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)  # (B, 2, H, W)
        return vox

def build_neuromorphic_dataset(dataset, batch_size, data_root=None, one_hot_label=False):
    dataset_key = dataset.lower()
    if dataset_key not in DVS_DATASETS:
        raise NotImplementedError(f"Unsupported neuromorphic dataset: {dataset}")

    dataset_root = join(data_root, DVS_DATASETS[dataset_key]["subdir"])

    if dataset_key == "cifar10dvs":
        train_set, test_set = Cifar10DVS(
            dataset_root,
            num_classes=DVS_DATASETS[dataset_key]["outdim"],
            one_hot_label=one_hot_label,
        )
    elif dataset_key == "dvsgesture":
        train_set, test_set = DVSGesture(
            dataset_root,
            num_classes=DVS_DATASETS[dataset_key]["outdim"],
            one_hot_label=one_hot_label,
        )
    elif dataset_key == "ncaltech":
        train_set = NCaltech101(dataset_root, True, (224, 224))
        test_set = NCaltech101(dataset_root, False, (224, 224))
    else:
        raise NotImplementedError(f"Unsupported neuromorphic dataset: {dataset}")

    num_workers = _resolve_dataloader_workers()
    prefetch_factor = max(1, int(os.getenv("SNN_DVS_PREFETCH_FACTOR", "2")))
    pin_memory = True
    persistent_workers = num_workers > 0

    train_loader_kwargs = {
        "dataset": train_set,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    test_loader_kwargs = {
        "dataset": test_set,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }

    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        train_loader_kwargs["worker_init_fn"] = _worker_init_fn
        test_loader_kwargs["prefetch_factor"] = prefetch_factor
        test_loader_kwargs["worker_init_fn"] = _worker_init_fn

    train_loader = torch.utils.data.DataLoader(**train_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(**test_loader_kwargs)

    in_channels = DVS_DATASETS[dataset_key]["in_channels"]
    outdim = DVS_DATASETS[dataset_key]["outdim"]

    return train_loader, test_loader, in_channels, outdim
