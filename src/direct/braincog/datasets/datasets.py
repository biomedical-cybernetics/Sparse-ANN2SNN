import torch
import torchvision.datasets as datasets
from torchvision import transforms
from timm.data import create_transform

DATA_DIR = "./data"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)

def build_transform(is_train, img_size):
    """
    构建数据增强, 适用于static data
    :param is_train: 是否训练集
    :param img_size: 输出的图像尺寸
    :return: 数据增强策略
    """
    resize_im = img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * img_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    if img_size > 32:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    return transforms.Compose(t)

def build_dataset(is_train, img_size, dataset, path, same_da=False):
    """
    构建带有增强策略的数据集
    :param is_train: 是否训练集
    :param img_size: 输出图像尺寸
    :param dataset: 数据集名称
    :param path: 数据集路径
    :param same_da: 为训练集使用测试集的增广方法
    :return: 增强后的数据集
    """
    transform = build_transform(False, img_size) if same_da else build_transform(is_train, img_size)

    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    else:
        raise NotImplementedError

    return dataset, nb_classes

def get_cifar10_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, **kwargs):
    """
    获取CIFAR10数据
     https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR10', root, same_da)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR10', root, same_da)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False,
        num_workers=num_workers
    )
    return train_loader, test_loader, None, None

def get_cifar100_data(batch_size, num_workers=8, same_data=False,root=DATA_DIR, *args, **kwargs):
    """
    获取CIFAR100数据
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR100', root, same_data)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR100', root, same_data)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, False, None
