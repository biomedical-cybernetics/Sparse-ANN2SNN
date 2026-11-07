from .ResNet import resnet18, resnet50
from .VGG import vgg16
from .STBP_Model import MyCifarANNDense
from .load_data import load_data_mlp
import torch.nn as nn
DVS_DATASETS = {"dvsgesture": {"subdir": "dvsgesture", "in_channels": 18, "outdim": 11}}


def prepare_model_and_loader(args, data_root):
    in_channels = 3
    if args.dataset.lower() == 'imagenet':
        train_loader, test_loader, outdim = load_data_imagenet(args, data_root)
        indim, hiddim = None, None # Not used/needed for ViT
    elif args.dataset.lower() in DVS_DATASETS:
        from .load_data_dvs import build_neuromorphic_dataset

        train_loader, test_loader, in_channels, outdim = build_neuromorphic_dataset(
            args.dataset,
            args.bs,
            data_root
        )
        indim, hiddim = None, None
    else:
        train_loader,test_loader,indim,outdim,hiddim=load_data_mlp(args.dataset, args.bs, args.dim, data_root = data_root)

    if args.conv_sparsity > 0.0:
        pass

    model_name=args.architecture.lower()

    if 'vit' in model_name or 'deit' in model_name:
        # Import timm only when needed.
        from timm.models import create_model

        model = create_model(
            args.architecture,
            pretrained=False,
            num_classes=outdim,
            drop_rate=getattr(args, 'drop', 0.0),
            drop_path_rate=getattr(args, 'drop_path', 0.0),
            img_size=getattr(args, 'input_size', 224)
        )
        # Approximate activation count or handle dynamically
        num_activations = len(list(model.modules()))
    elif model_name == 'resnet-50':
        if args.conv_sparsity==0.0 or args.pruning:
            model = resnet50(num_classes=outdim, in_channels=in_channels)
        else:
            model = ResNet50(CHT.hanming_config, num_classes=outdim, init_channels=in_channels)
    elif model_name == 'resnet-18':
        if args.conv_sparsity==0.0 or args.pruning:
            model = resnet18(num_classes=outdim, in_channels=in_channels)
        else:
            model = ResNet18(CHT.hanming_config, num_classes=outdim, init_channels=in_channels)
    elif 'vgg-16' in model_name:
        hidden_dim = 4096 if 'large' in model_name else 512
        if args.conv_sparsity == 0.0 or args.pruning:
            model = vgg16(num_classes=outdim, dropout=args.dropout, one_fc=args.one_fc, hidden_dim=hidden_dim, in_channels=in_channels)
        else:
            model = VGG16_CIFAR_BN(CHT.hanming_config, outdim, dropout=args.dropout, one_fc=args.one_fc, hidden_dim=hidden_dim, init_channels=in_channels)
    elif 'stbpann' in model_name:
        if args.conv_sparsity==0.0:
            model = MyCifarANNDense(outdim)
        else:
            model = MyCifarANN_CHT(CHT.hanming_config, outdim)
    elif model_name == 'mlp':
        mlp_hidden_dims = args.mlp_hidden_dims if getattr(args, 'mlp_hidden_dims', None) is not None else hiddim
        model = MLP(indim, mlp_hidden_dims, outdim, dropout=0)
    else:
        raise NotImplementedError

    num_activations = count_actiavtions(model)

    return model, num_activations, train_loader,test_loader

def count_actiavtions(model : nn.Module):
    count = 0
    for mod in model.modules():
        if isinstance(mod, (nn.GELU, nn.ReLU)):
            count += 1
    return count
