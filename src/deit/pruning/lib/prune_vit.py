from tqdm import tqdm
import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT
from .sparsegpt import SparseGPT
from .data import get_imagenet
from .prune import find_layers

def prune_ria_vit(args, model, device=torch.device("cuda:0")):
    '''
    RIA pruning method, ViT structure, ImageNet, unstructured, no quantization
    '''

    # 1. Load Calibration Data
    print("Loading ImageNet calibration data...")
    # Assume args.calib_dataset contains the path to ImageNet
    dataloader = get_imagenet(args.calib_dataset, nsamples=args.nsamples, batch_size=args.batch_size, seed=args.seed)
    print("Dataset loading complete")

    # 2. Prepare Inputs (Capture inputs to the first block)
    layers = model.blocks # timm ViT usually stores blocks here

    # Storage for captured inputs
    # Note: ViT inputs are (Batch, N_tokens, Dim).

    inps = []
    outs = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, **kwargs):
            # Capture the input tensor
            inps.append(x.detach()) # x is on device
            raise RuntimeError # Stop inference

    # Replace first layer
    original_layer0 = layers[0]
    layers[0] = Catcher(original_layer0)

    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch.to(device))
            except RuntimeError:
                pass  #stop forward by Error ......

    # Restore model
    layers[0] = original_layer0

    # 3. Layer Iteration

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer, layers=[nn.Linear]) # Only finding Linears inside the Block

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=args.reconstruction)

        def add_batch(name):
            def tmp(_, inp, out):
                # WrappedGPT expects (Batch, Seq, Dim) or (Batch, Dim).
                # ViT Linear layers inputs are (Batch, Tokens, Dim).
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # 3.1 Register Hooks to collect Hessian / Activation stats
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 3.2 Pass calibration data through current layerd
        outs = []
        for curr_input in inps:
            with torch.no_grad():
                # timm blocks forward: x = block(x)d
                out = layer(curr_input)
                outs.append(out)

        # Remove hooks
        for h in handles:
            h.remove()

        # 3.3 Pruning & Reconstruction
        for name in subset:
            print(f"Pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()

            # --- Metric Calculation ---
            if args.prune_method == "wanda":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "ria":
                # --- RIA Metric Calculation ---
                W_abs = torch.abs(W)
                # Add eps to avoid div by zero if dead neurons
                W_metric = (W_abs / (torch.sum(W_abs, dim=0) + 1e-10) +
                            W_abs / (torch.sum(W_abs, dim=1).reshape(-1, 1) + 1e-10))

                scaler = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                W_metric = W_metric * (scaler ** args.a)

            W_mask = (torch.zeros_like(W_metric) == 1) # All False

            non_zero_elements = W.numel()
            n_prune = int(non_zero_elements * args.sparsity_ratio)

            if n_prune > 0:
                # Flatten -> Sort -> Threshold
                # args.per_outneuron for row-wise unstructured
                if getattr(args, 'per_outneuron', False):
                    # Prune each row independently (uniform row sparsity)
                    k = int(W.shape[1] * args.sparsity_ratio)
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)
                else:
                    # Global pruning within the layer
                    thresh = torch.sort(W_metric.flatten())[0][n_prune]
                    W_mask = (W_metric <= thresh)

            # --- Apply Pruning ---
            if args.reconstruction:
                # Use Hessian to reconstruct/compensate remaining weights (SparseGPT style update with RIA mask)
                print(f"Reconstructing {name}...")
                wrapped_layers[name].fasterprune(args.sparsity_ratio, mask=W_mask)
            else:
                # Simple RIA pruning
                subset[name].weight.data[W_mask] = 0

            wrapped_layers[name].free() # Free Hessian memory

        # 3.4 Update Inputs for next layer
        # The outputs we collected in step 3.2 were based on UNPRUNED weights (due to hooks running with raw module).
        # To strictly follow layer-wise logic, we should re-run the *pruned* layer to generate inputs for the next layer.

        new_inps = []
        for curr_input in inps:
             with torch.no_grad():
                new_inps.append(layer(curr_input))

        inps = new_inps
        torch.cuda.empty_cache()

def prune_sparsegpt_vit(args, model, device=torch.device("cuda:0")):
    print('Starting SparseGPT for ViT ...')
    dataloader = get_imagenet(args.calib_dataset, nsamples=args.nsamples, batch_size=args.batch_size, seed=args.seed)

    layers = model.blocks

    inps = []
    outs = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x, **kwargs):
            inps.append(x.detach())
            raise ValueError

    original_layer0 = layers[0]
    layers[0] = Catcher(original_layer0)
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch.to(device))
            except ValueError:
                pass
    layers[0] = original_layer0

    outs = []

    print('Ready.')

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer, layers=[nn.Linear])

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                # SparseGPT expects inputs flattened if len(shape) == 3
                # ViT Input: (Batch, Tokens, Dim) -> (Batch*Tokens, Dim) inside SparseGPT.add_batch if not handled
                # But SparseGPT.add_batch usually handles 2D (Features, Samples) or 3D.
                # Looking at original code:
                # if len(inp.shape) == 2: inp = inp.unsqueeze(0)
                # tmp = inp.shape[2]
                #      if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
                # So passing (Batch, Tokens, Dim) is fine.
                # However, our catcher gets list of (Batch, Tokens, Dim).
                # Here we pass inp[0] because hook signature is (module, inp, out) where inp is tuple.
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        outs = []
        for curr_input in inps:
             with torch.no_grad():
                out = layer(curr_input)
                outs.append(out)

        for h in handles:
            h.remove()

        for name in gpts:
            print(f"Pruning layer {i} name {name}")
            # ViT usually doesn't need 'norm=True' logic which is specific to some LLMs.
            # Assuming standard SparseGPT usage.
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=0, prune_m=0, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # Update inputs for next layer
        new_inps = []
        for curr_input in inps:
             with torch.no_grad():
                new_inps.append(layer(curr_input))
        inps = new_inps
        torch.cuda.empty_cache()

def prune_magnitude_vit(args, model, device=torch.device("cuda:0")):
    layers = model.blocks

    # Check if we should do row-wise pruning
    per_outneuron = getattr(args, 'per_outneuron', False)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=[nn.Linear])

        for name in subset:
            print(f"Pruning layer {i} name {name}")
            W = subset[name].weight.data.clone()

            if args.prune_method == "magnitude":
                W_metric = torch.abs(W)
            elif args.prune_method == "ri":
                # Static RI metric (without activation scaler)
                W_abs = torch.abs(W)
                W_metric = W_abs / (torch.sum(W_abs, dim=0) + 1e-10) + \
                           W_abs / (torch.sum(W_abs, dim=1).reshape(-1, 1) + 1e-10)

            W_mask = (torch.zeros_like(W_metric) == 1) # All False

            n_prune = int(W.numel() * args.sparsity_ratio)

            if n_prune > 0:
                if per_outneuron:
                    # Row-wise / per-output-neuron pruning
                    k = int(W.shape[1] * args.sparsity_ratio)
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)
                else:
                    # Global layer-wise pruning
                    thresh = torch.sort(W_metric.flatten())[0][n_prune]
                    W_mask = (W_metric <= thresh)

            subset[name].weight.data[W_mask] = 0
