"""Exact-budget masks, isolated initialization and uniform random dynamics."""
import math
import torch
from torch import nn
from cnn_dst_gd_model import ConvMask, Topology as BaseTopology, pick
from cnn_dst_gd_protocol import active_count, conv_sparsity
from resnet_matched_model import model_for
from resnet_topology_protocol import ONESHOT


def same_state(left, right):
    if left.keys() != right.keys():
        return False
    for name, value in left.items():
        other = right[name]
        if value.dtype != other.dtype or value.shape != other.shape:
            return False
        a = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = other.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            return False
    return True


def mask_audit(mask, original):
    a,b=mask.cpu().bool(),original.cpu().bool()
    intersection=int((a&b).sum()); union=int((a|b).sum())
    return {"active_pairs":int(a.sum()), "row_degree":a.sum(1).tolist(),
            "column_degree":a.sum(0).tolist(), "no_input_output_channels":int((a.sum(1)==0).sum()),
            "changed_pairs":int((a!=b).sum()), "original_retention":intersection/max(1,int(b.sum())),
            "jaccard_overlap":intersection/max(1,union)}


def degree_randomize(original, generator):
    mask=original.cpu().bool().clone(); rows=[set(torch.where(r)[0].tolist()) for r in mask]
    attempts=200*len(rows); successes=0
    for _ in range(attempts):
        if len(rows)<2: continue
        i,j=torch.randperm(len(rows),generator=generator)[:2].tolist()
        left,right=sorted(rows[i]-rows[j]),sorted(rows[j]-rows[i])
        maximum=min(len(left),len(right))
        if not maximum: continue
        n=int(torch.randint(1,maximum+1,(),generator=generator))
        a={left[k] for k in torch.randperm(len(left),generator=generator)[:n].tolist()}
        b={right[k] for k in torch.randperm(len(right),generator=generator)[:n].tolist()}
        rows[i]=(rows[i]-a)|b; rows[j]=(rows[j]-b)|a; successes+=1
    mask.zero_()
    for i,row in enumerate(rows): mask[i,sorted(row)]=True
    if not torch.equal(mask.sum(0),original.cpu().sum(0)) or not torch.equal(mask.sum(1),original.cpu().sum(1)):
        raise ValueError("Degree-preserving swap changed degrees")
    audit=mask_audit(mask,original)
    audit.update(attempts=attempts,effective_exchanges=successes,
                 warning="Limited randomization" if not successes or audit["jaccard_overlap"]>.95 else None)
    return mask,audit


def uniform_mask(original, method, generator):
    mask=torch.zeros_like(original,device="cpu",dtype=torch.bool)
    if method=="random-fixed-output":
        for i,n in enumerate(original.cpu().sum(1).tolist()):
            mask[i,torch.randperm(mask.shape[1],generator=generator)[:n]]=True
    else:
        mask.flatten()[torch.randperm(mask.numel(),generator=generator)[:int(original.sum())]]=True
    return mask


def magnitude_mask(weight, original, output=False):
    score=weight.detach().float().abs().mean((-1,-2)).cpu()
    if not torch.isfinite(score).all(): raise FloatingPointError("Nonfinite magnitude")
    mask=torch.zeros_like(original,device="cpu",dtype=torch.bool)
    # Stable sorting breaks ties by ascending pair index.
    if output:
        for i,n in enumerate(original.cpu().sum(1).tolist()):
            mask[i,torch.argsort(score[i],descending=True,stable=True)[:n]]=True
    else:
        mask.flatten()[torch.argsort(score.flatten(),descending=True,stable=True)[:int(original.sum())]]=True
    return mask


class RandomConv(ConvMask):
    @torch.no_grad()
    def evolve(self):
        with torch.autocast(self.module.weight.device.type,enabled=False):
            old=self.mask.clone()
            sparsity=conv_sparsity(self.count,self.job.horizon) if self.job.method=="random-dst-decay" else .7
            target=active_count(old.numel(),sparsity); grow=math.floor(.05*target)
            keep=pick(None,old,target-grow,random=True)
            self.mask.zero_(); self.mask.flatten()[keep]=True
            new=pick(None,~self.mask,grow,random=True); self.mask.flatten()[new]=True
            self.count+=1
            if int(self.mask.sum())!=target: raise ValueError("Random DST budget error")
            return {"layer":self.name,"update":self.count,"active":target,
                    "density_pruned":int(old.sum())-target,"planned_regrow":grow,
                    "changed":int((old!=self.mask).sum()),"delta_used":None}


class Topology(BaseTopology):
    def __init__(self,model,optimizer,job,reference_masks):
        self.model,self.job,self.conv,self.linear=model,job,[],None
        self.initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; self.diagnostics={}
        generator=torch.Generator(device="cpu").manual_seed(job.seed+100000)
        for name,module in model.named_modules():
            if not isinstance(module,nn.Conv2d) or "shortcut" in name: continue
            key=name+".weight"; original=reference_masks[key].cpu().bool()
            if tuple(original.shape)!=tuple(module.weight.shape[:2]) or int(original.sum())!=active_count(original.numel(),.7):
                raise ValueError("Reference mask scope/budget mismatch")
            # Preserve the source constructor's CUDA randperm ordering for decay initialization.
            layer=RandomConv(module,key,job)
            if job.method=="cht-fixed": mask=original.clone()
            elif job.method=="degree-preserving-fixed":
                mask,audit=degree_randomize(original,generator); self.diagnostics[key]=audit
            elif job.method in ONESHOT: mask=magnitude_mask(module.weight,original,job.method=="magnitude-output")
            else: mask=uniform_mask(original,job.method,generator)
            layer.mask.copy_(mask.to(layer.mask.device)); self.conv.append(layer)
            self.diagnostics.setdefault(key,mask_audit(mask,original))
        if len(self.conv)!=17: raise ValueError("Expected 17 target Conv masks")
        self.initial_masks={k:v.clone() for k,v in self.masks().items()}

    @torch.no_grad()
    def evolve(self,epoch):
        return [layer.evolve() for layer in self.conv] if self.job.method.startswith("random-dst-") else []

    def inventory(self):
        rows=super().inventory()
        for row in rows:
            mask=self.masks().get(row["layer"]+".weight")
            if mask is not None:
                row.update(active_pairs=int(mask.sum()),row_degree=mask.sum(1).tolist(),
                           column_degree=mask.sum(0).tolist(),no_input_output_channels=int((mask.sum(1)==0).sum()))
        return rows


def install(model,optimizer,job,reference_masks):
    if job.method!="random-dst-decay": return Topology(model,optimizer,job,reference_masks)
    # Exact source mask construction, seed guard is supplied by the common worker.
    class SourceIdentity:
        method="ours"; granularity="channel"; architecture="ResNet-18"
    topology=Topology.__new__(Topology)
    topology.model,topology.job,topology.conv,topology.linear=model,job,[],None
    topology.initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; topology.diagnostics={}
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d) and "shortcut" not in name:
            layer=RandomConv(module,name+".weight",SourceIdentity()); layer.job=job
            topology.conv.append(layer)
            topology.diagnostics[layer.name]=mask_audit(layer.mask,reference_masks[layer.name])
    topology.initial_masks={k:v.clone() for k,v in topology.masks().items()}
    return topology
