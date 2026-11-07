import copy, math
from types import SimpleNamespace
import torch
from torch import nn
import Models
from Models.VGG import vgg16

def model_for(job):
    return vgg16(job.classes, dropout=job.dropout, one_fc=job.variant=="gap", hidden_dim=512, in_channels=3)

def validate_graph(model, job, quantized=False):
    conv=[m for m in model.modules() if isinstance(m,nn.Conv2d)]
    linear=[m for m in model.modules() if isinstance(m,nn.Linear)]
    drops=[m.p for m in model.modules() if isinstance(m,nn.Dropout)]
    if (len(conv)!=13 or conv[0].in_channels!=3 or len(linear)!=(1 if job.variant=="gap" else 3)
            or linear[-1].out_features!=job.classes or drops!=([] if job.variant=="gap" else [.5,.5])
            or sum(isinstance(m,nn.BatchNorm2d) for m in model.modules())!=13):
        raise ValueError("VGG graph/classifier/dropout contract violated")
    if job.method!="depgraph" and job.variant=="linear99":
        if [tuple(m.weight.shape) for m in linear]!=[(512,512),(512,512),(job.classes,512)]:
            raise ValueError("Not the small 512-wide classifier")
    if job.method=="depgraph":
        with torch.random.fork_rng(devices=[]):baseline=model_for(job)
        shapes={n:tuple(m.weight.shape) for n,m in baseline.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))}
        for n,m in model.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)) and (n not in shapes or any(a>b for a,b in zip(m.weight.shape,shapes[n]))):
                raise ValueError("Pruned graph exceeds measured full-width envelope")

def load_model(path, job):
    value=torch.load(path,map_location="cpu",weights_only=False)
    if isinstance(value,nn.Module): model=value
    else:
        model=model_for(job);model.load_state_dict(value,strict=True)
    if (any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in model.parameters())
            or any(not torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_tensor(v))):
        raise ValueError("Nonfinite or non-FP32 checkpoint")
    validate_graph(model,job,False)
    return model

def loaders(job, root, smoke=False):
    args=SimpleNamespace(architecture="VGG-16",dataset=job.dataset,bs=128,
        conv_sparsity=0.,linear_sparsity=0.,pruning="",one_fc=job.variant=="gap",hidden_dim=512,
        dropout=job.dropout,dim=2,distributed=False)
    _,_,train,test=Models.prepare_model_and_loader(args,str(root))
    if len(train.dataset)!=50000 or len(test.dataset)!=10000: raise ValueError("Wrong official split")
    return tuple(torch.utils.data.DataLoader(l.dataset,batch_size=128,shuffle=shuffle,
        num_workers=0,pin_memory=True,drop_last=False) for l,shuffle in ((train,True),(test,False)))

def fixed_ratio_prune(model, example, job):
    import torch_pruning as tp
    before={n:list(m.weight.shape) for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))}
    output=model.last_layer
    ratios={m:(1-math.sqrt(.3) if isinstance(m,nn.Conv2d) else .9)
            for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear)) and m is not output}
    model.eval()
    pruner=tp.pruner.GroupNormPruner(model,example,importance=tp.importance.GroupMagnitudeImportance(p=2),
        iterative_steps=1,pruning_ratio_dict=ratios,max_pruning_ratio=1.,ignored_layers=[output],global_pruning=True)
    pruner.step();validate_graph(model,job)
    with torch.no_grad():
        if model(example).shape!=(len(example),job.classes):raise ValueError("Invalid pruned graph")
    from resnet_matched_model import weight_count
    macs,_=tp.utils.count_ops_and_params(model,example_inputs=example)
    target=budget(job)["active_weights"]
    return model,{"selection":"fixed old layer-type ratio mapping; no search","conv_ratio":1-math.sqrt(.3),
        "hidden_ratio":.9,"before_shapes":before,"layers":inventory(model),"actual_weights":weight_count(model),
        "ours_target_weights":target,"relative_difference":(weight_count(model)-target)/target,"macs":float(macs)}

def inventory(model,masks=None):
    masks=masks or {};rows=[]
    for n,m in model.named_modules():
        if not isinstance(m,(nn.Conv2d,nn.Linear)):continue
        mask=masks.get(n+".weight");total=m.weight.numel()
        active=total if mask is None else int(mask.sum())*(total//mask.numel())
        rows.append({"layer":n,"shape":list(m.weight.shape),"weights":total,"active_weights":active,
            "mask_zeros":total-active,"mask_sparsity":1-active/total,
            "numerical_zeros":int((m.weight==0).sum()),"granularity":"dense" if mask is None else
                "channel-pair" if isinstance(m,nn.Conv2d) else "element",
            "zero_input_units":0 if mask is None else int((mask.reshape(mask.shape[0],-1).sum(1)==0).sum())})
    return rows


def budget(job):
    with torch.random.fork_rng(devices=[]):model=model_for(job)
    masks={}
    if job.method!="dense":
        for n,m in model.named_modules():
            if isinstance(m,nn.Conv2d):
                shape=m.weight.shape[:2];a=math.prod(shape)-math.floor(.7*math.prod(shape))
            elif isinstance(m,nn.Linear) and m is not model.last_layer:
                shape=m.weight.shape;a=math.floor(.01*m.weight.numel())
            else:continue
            mask=torch.zeros(shape,dtype=torch.bool);mask.flatten()[:a]=True;masks[n+".weight"]=mask
    rows=inventory(model,masks)
    return {"kind":"final-budget simulation, not trained result","layers":rows,
            "weights":sum(r["weights"] for r in rows),"active_weights":sum(r["active_weights"] for r in rows)}
