import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import os, sys
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_data_mlp(dataset,batch_size,dim_factor=2,data_root=None):
    # Define standard transforms for new RGB datasets
    imnet_mean = [0.485, 0.456, 0.406]
    imnet_std = [0.229, 0.224, 0.225]
    transform_train_rgb = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imnet_mean, imnet_std)
    ])
    transform_test_rgb = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(imnet_mean, imnet_std)
    ])
    if dataset == 'EMNIST':
        # EMNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root=data_root,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ]),
            download=True,
            split='balanced'),
            batch_size=batch_size,
            shuffle=True)

        # EMNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.EMNIST(
            root=data_root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1751,), (0.3332,))
            ]),
            download=True,
            split='balanced'),
            batch_size=batch_size,
            shuffle=False)

        indim = 784
        outdim = 47


    elif dataset == "MNIST":
        # MNIST train dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True)
        # MNIST test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 784
        outdim = 10


    elif dataset == "Fashion_MNIST":
        # Fashion MNIST train dataset
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root=data_root,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=batch_size,
            shuffle=True)

        # Fashion MNIST test dataset
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
            root=data_root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ]),
            download=True),
            batch_size=batch_size,
            shuffle=False)
        indim = 784
        outdim = 10

    elif dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_root, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])),
            batch_size=batch_size, shuffle=False)
        indim = 3072
        outdim = 10


    elif dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_root, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762])])),
            batch_size=batch_size, shuffle=False)
        indim = 3072
        outdim = 100

    elif dataset == "FER2013":
        train_loader = torch.utils.data.DataLoader(
            datasets.FER2013(
            root=data_root,
            split='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5077,), (0.2550,))
            ])),
        batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FER2013(
            root=data_root,
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5077,), (0.2550,))
            ]),
        ),
        batch_size=batch_size, shuffle=False)
        indim = 48*48
        outdim = 7

    elif dataset == "SVHN":
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
            root=os.path.join(data_root,'svhn'),
            split='train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4377, 0.4438, 0.4728],
                                    [0.198, 0.201, 0.197])
            ]),
            download=True
        ),
        batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
            root=os.path.join(data_root,'svhn'),
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4377, 0.4438, 0.4728],
                                    [0.198, 0.201, 0.197])
            ]),
            download=True
        ),
        batch_size=batch_size, shuffle=False)
        indim = 32*32*3
        outdim = 10

    elif dataset == 'tiny-imagenet-ori':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2296, 0.2263, 0.2255]
        train_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=True, transform=transforms.Compose([
                # transforms.RandomResizedCrop(64),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 64*64*3
        outdim = 200

    elif dataset == 'tiny-imagenet-crop':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2296, 0.2263, 0.2255]
        train_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=True, transform=transforms.Compose([
                transforms.Resize(64),
                transforms.RandomCrop(48),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=False, transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(48),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 48*48*3
        outdim = 200

    elif dataset == 'tiny-imagenet-resize':
        mean = [0.4802, 0.4481, 0.3976]
        std = [0.2175, 0.2139, 0.2133]
        train_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=True, transform=transforms.Compose([
                transforms.Resize(48, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TinyImageNet_load(root=os.path.join(data_root,'tiny-imagenet-200'), train=False, transform=transforms.Compose([
                transforms.Resize(48, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])),
            batch_size=batch_size, shuffle=False)
        indim = 48*48*3
        outdim = 200
    # ========== NEW DATASETS ADDITION START ========== #
    elif dataset == 'OxfordFlowers':
        # Oxford 102 Flowers Dataset
        from torchvision.datasets import Flowers102

        # Combine train and validation sets
        train_val = Flowers102(
            root=data_root,
            split='train',
            download=True,
            transform=None
        )
        val_set = Flowers102(
            root=data_root,
            split='val',
            download=True,
            transform=None
        )
        # Create combined training set (train + validation)
        train_set = torch.utils.data.ConcatDataset([train_val, val_set])
        test_set = Flowers102(
            root=data_root,
            split='test',
            download=True,
            transform=transform_test_rgb
        )

        # Apply transforms to training subset
        train_set = TransformedSubset(train_set, transform=transform_train_rgb)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        indim = 64 * 64 * 3
        outdim = 102  # 102 flower categories

    elif dataset == 'Caltech256':
        # Caltech-256 Dataset
        from torchvision.datasets import Caltech256
        def ensure_rgb(image):
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        # Load full dataset without initial transform
        full_set = Caltech256(
            root=data_root,
            download=False,
            transform=None
        )
        transform_train_rgb_caltech = transforms.Compose([
            transforms.Lambda(ensure_rgb),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imnet_mean, imnet_std)
        ])
        transform_test_rgb_caltech = transforms.Compose([
            transforms.Lambda(ensure_rgb),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(imnet_mean, imnet_std)
        ])
        # Create 80/20 train-test split
        train_size = int(0.8 * len(full_set))
        test_size = len(full_set) - train_size
        train_sub, test_sub = random_split(
            full_set,
            [train_size, test_size]
        )

        # Apply transforms to subsets
        train_set = TransformedSubset(train_sub, transform=transform_train_rgb_caltech)
        test_set = TransformedSubset(test_sub, transform=transform_test_rgb_caltech)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        indim = 64 * 64 * 3
        outdim = 257  # 256 categories + 1 clutter class

    elif dataset == "OxfordIIITPet":

        # 加载数据集（自动下载）
        train_val_set = datasets.OxfordIIITPet(
            root=data_root,
            split='trainval',
            download=True,
            transform=transform_train_rgb
        )

        test_set = datasets.OxfordIIITPet(
            root=data_root,
            split='test',
            download=True,
            transform=transform_test_rgb
        )

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        indim = 64 * 64 * 3  # 输入维度（64x64 RGB图像）
        outdim = 37  # 37个宠物类别（12种猫+25种狗）:cite[4]:cite[8]

    elif dataset == 'INaturalist':
        # iNaturalist 2021 Dataset (Mini version)
        from torchvision.datasets import INaturalist

        train_loader = torch.utils.data.DataLoader(
            INaturalist(
                root=os.path.join(data_root,'INaturalist'),
                version='2021_train_mini',  # Manageable subset
                download=True,
                transform=transform_train_rgb
            ),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            INaturalist(
                root=os.path.join(data_root,'INaturalist'),
                version='2021_valid',  # Full validation set
                download=True,
                transform=transform_test_rgb
            ),
            batch_size=batch_size,
            shuffle=False
        )

        indim = 64 * 64 * 3
        outdim = 10000  # 10,000 species categories
    # ========== NEW DATASETS ADDITION END ========== #

    dimension = indim * dim_factor
    hiddim = [dimension, dimension, dimension]

    return train_loader, test_loader, indim, outdim, hiddim
