import torch


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )
LAYER_INDEX=0

def get_weighted_layers_mlp(model, layers=None, linear_layers_mask=None, chain_list=None):
    global LAYER_INDEX

    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []
    if chain_list is None:
        chain_list = []


    items = model._modules.items()

    for layer_name, p in items:
        if layer_name == "last_layer":
            continue
        elif isinstance(p, torch.nn.Linear):

            layers.append([p])
            chain_list.append(LAYER_INDEX)
            linear_layers_mask.append(1)
            LAYER_INDEX += 1
        else:
            _, linear_layers_mask = get_weighted_layers_mlp(p, layers=layers, linear_layers_mask=linear_layers_mask, chain_list=chain_list)

    return layers, linear_layers_mask

def get_W(model, return_linear_layers_mask=False):
    global LAYER_INDEX
    LAYER_INDEX = 0

    chain_list = []

    layers, linear_layers_mask = get_weighted_layers_mlp(model, chain_list = chain_list)
    # exit()
    # exit()
    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        print(layer[idx].weight.shape)
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)
    assert len(W) == LAYER_INDEX

    if return_linear_layers_mask:
        return W, linear_layers_mask, chain_list
    return W, chain_list

@torch.no_grad()
def reset_momentum(optimizer,mask,W):
    for w, mask, in zip(W, mask):
        param_state =optimizer.state[w]
        optimizer_state_list = ["momentum_buffer"]
        for optimizer_state in optimizer_state_list:
            if optimizer_state in param_state:
                buf = param_state[optimizer_state]
                buf *= mask

@torch.no_grad()
def apply_mask_to_W(mask,W):
    for w,m in zip(W,mask):
        w*=m

def wrapp_optimizer(optimizer,backward_masks,W):
    origion_step=optimizer.step
    def new_step():
        origion_step()
        apply_mask_to_W(backward_masks,W)
        reset_momentum(optimizer,backward_masks,W)
    optimizer.step=new_step
