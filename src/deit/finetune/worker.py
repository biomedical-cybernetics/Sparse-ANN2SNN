"""Eight-rank fixed-mask DeiT training."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time

os.environ.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
os.environ["NVIDIA_TF32_OVERRIDE"]="0"

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, get_worker_info

from runtime import atomic_json
from training_support import lr_at


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state(state["cuda"])


class WorkerStateDataset(Dataset):
    """Collect worker RNG state after each complete drop_last epoch.

    Batches are assigned round-robin. At the epoch boundary, all prefetched
    batches have been consumed and the persistent iterator restarts at worker zero.
    """
    def __init__(self, dataset, batches, batch_size, workers, queue, restored=None):
        self.dataset=dataset; self.batches=batches; self.batch_size=batch_size
        self.workers=workers; self.queue=queue; self.restored=restored
        self.calls=0; self.restored_once=False

    def __len__(self): return len(self.dataset)

    def __getitem__(self,index):
        info=get_worker_info()
        if info is None: raise RuntimeError("Training loader requires worker processes")
        if not self.restored_once:
            if self.restored is not None:
                s=self.restored[info.id]
                random.setstate(s["python"]); np.random.set_state(s["numpy"])
                torch.set_rng_state(s["torch"])
            self.restored_once=True
        result=self.dataset[index]
        self.calls+=1
        count=(self.batches//self.workers+int(info.id<self.batches%self.workers))*self.batch_size
        if self.calls==count:
            state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
            # Serialize tensors now: queue readers must not depend on worker IPC lifetime.
            self.queue.put((info.id,pickle.dumps(state,protocol=5)))
            self.calls=0
        return result


class Shard(Sampler):
    def __init__(self,n,rank,world): self.indices=range(rank,n,world)
    def __len__(self): return len(self.indices)
    def __iter__(self): return iter(self.indices)


def finite(tensors):
    flags=[torch.isfinite(t).all() for t in tensors if torch.is_tensor(t) and t.is_floating_point()]
    by_device={}
    for flag in flags: by_device.setdefault(flag.device,[]).append(flag)
    return all(bool(torch.stack(v).all().item()) for v in by_device.values())


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(value,tmp); os.replace(tmp,path)


def shutdown(loader):
    it=getattr(loader,"_iterator",None)
    if it is not None:
        it._shutdown_workers()
        loader._iterator=None


@torch.no_grad()
def evaluate(model,loader,device,smoke=False):
    model.eval()
    totals=torch.zeros(4,device=device,dtype=torch.float64)
    batches=itertools.islice(loader,1) if smoke else loader
    for x,y in batches:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.float16):
            pred=model(x); loss=torch.nn.functional.cross_entropy(pred,y,reduction="sum")
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite validation loss")
        top=pred.topk(5,dim=1).indices
        totals+=torch.stack(((top[:,0]==y).sum(),(top==y[:,None]).any(1).sum(),
            torch.tensor(len(y),device=device),loss)).double()
    dist.all_reduce(totals)
    c1,c5,n,loss=totals.cpu().tolist()
    if n!=(1536 if smoke else 50000): raise RuntimeError(f"Validation coverage mismatch: {n}")
    return dict(acc1=100*c1/n,acc5=100*c5/n,samples=int(n),loss=loss/n)


def setup_precision(device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 not disabled")
    torch.cuda.set_device(device)


def check_dataset(dataset,reference):
    actual=dict(samples=len(dataset),classes=dataset.class_to_idx)
    if actual!=reference: raise RuntimeError("Dataset sample count or class mapping differs")
    return actual


def train(manifest,job,output,resume=None):
    import main as legacy
    import models  # noqa: F401: model registration
    from datasets import build_dataset
    from samplers import RASampler
    from dst_util import recur_split_atten, get_W
    from dst_scheduler import DSTScheduler
    from fixed_topology import validate_masks, audit_fixed_topology, same_masks
    from timm.models import create_model
    from timm.optim import create_optimizer
    from timm.scheduler import create_scheduler
    from timm.utils import ModelEma, get_state_dict
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy

    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=8: raise RuntimeError("Training requires exactly eight ranks")
    device=torch.device("cuda",int(os.environ["LOCAL_RANK"]))
    seed=job["seed"]; epochs=job["epochs"]; smoke=manifest.get("smoke",False)
    source=manifest["inputs"][f"{job['method']}-s{seed}"]
    started=time.monotonic()
    setup_precision(device); torch.backends.cudnn.benchmark=True
    torch.manual_seed(seed+rank); np.random.seed((seed+rank) % 2**32); random.seed(seed+rank)
    a=legacy.get_args_parser().parse_args(["--epochs",str(epochs),"--batch-size","128",
        "--lr","0.0005","--sparsity","0.7","--fixed-topology","--mask",source["mask"],
        "--data-path",manifest["data_root"],"--data-set","IMNET","--train-mode"])
    a.nb_classes=1000; a.distributed=True; a.rank=rank; a.world_size=world
    train_data,classes=build_dataset(True,a); val_data,_=build_dataset(False,a)
    sampler=RASampler(train_data,num_replicas=8,rank=rank,shuffle=True)
    steps=len(sampler)//128
    if (len(train_data),len(val_data),classes,steps)!=(1281167,50000,1000,1251):
        raise RuntimeError("ImageNet/step identity mismatch")
    dataset_identity=dict(train_samples=len(train_data),validation_samples=len(val_data),
        classes=train_data.class_to_idx)
    check_dataset(train_data,manifest["dataset_reference"]["train"])
    check_dataset(val_data,manifest["dataset_reference"]["test"])
    model=create_model(a.model,pretrained=False,num_classes=1000,drop_rate=0.,drop_path_rate=.1,img_size=224)
    recur_split_atten(model)
    initial=torch.load(source["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(initial["model"],strict=True); del initial
    masks=torch.load(source["mask"],map_location="cpu",weights_only=False)
    if isinstance(masks,dict): masks=next(masks[k] for k in ("backward_masks","masks","mask") if k in masks)
    masks=[m.detach().bool().cpu().contiguous() for m in masks]
    weights,_,_=get_W(model,a); mask_info=validate_masks(weights,masks,.7)
    if mask_info!=source["budget"]: raise RuntimeError("Canonical source masks changed")
    if audit_fixed_topology(weights,masks)["masked_weight_nonzero"]: raise RuntimeError("Source weights outside mask")
    model.to(device); ema=ModelEma(model,decay=.99996,device="",resume="")
    ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[device.index])
    a.lr=a.lr*a.batch_size*world/512.
    optimizer=create_optimizer(a,model)
    scheduler,_=create_scheduler(a,optimizer)
    actual_steps=[0]
    def counted(opt,args,kwargs): actual_steps[0]+=1
    handle=optimizer.register_step_post_hook(counted)
    pruner=DSTScheduler(ddp,optimizer,alpha=a.zeta,delta=a.update_interval*steps,
        static_topo=True,T_end=epochs*a.dst_end_epoch_fraction*steps,
        ignore_linear_layers=False,sparsity_distribution=a.sparsity_distribution,
        grad_accumulation_n=1,args=a,backward_masks=[m.to(device) for m in masks])
    scaler=torch.amp.GradScaler("cuda")
    mixup=Mixup(mixup_alpha=a.mixup,cutmix_alpha=a.cutmix,cutmix_minmax=a.cutmix_minmax,
        prob=a.mixup_prob,switch_prob=a.mixup_switch_prob,mode=a.mixup_mode,
        label_smoothing=a.smoothing,num_classes=1000)
    criterion=SoftTargetCrossEntropy()
    val_loader=DataLoader(val_data,sampler=Shard(50000,rank,8),batch_size=192,
        num_workers=10,pin_memory=True,drop_last=False)
    resume_payload=torch.load(resume,map_location="cpu",weights_only=False) if resume else None
    if resume_payload:
        if (resume_payload["identity"]!=manifest["identity"] or resume_payload["job"]!=job
                or resume_payload["dataset_identity"]!=dataset_identity or resume_payload["mask"]!=mask_info
                or resume_payload["smoke"] or not same_masks(resume_payload.get("masks"),masks)):
            raise RuntimeError("Recovery identity mismatch")
        model.load_state_dict(resume_payload["model"],strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["lr_scheduler"])
        scaler.load_state_dict(resume_payload["scaler"])
        ema.ema.load_state_dict(resume_payload["model_ema"])
        pruner.load_state_dict(resume_payload["pruner"])
        if not same_masks(pruner.backward_masks,masks):
            raise RuntimeError("Recovery masks differ from the fixed masks")
        initial_stats=resume_payload["initial"]
        curves=resume_payload["curves"]; best=resume_payload["best"]
        start_epoch=resume_payload["epoch"]+1
        worker_states=resume_payload["rng_states_by_rank"][rank]["workers"]
        training_seconds=resume_payload["training_loop_seconds"]
        best_path=resume_payload["best_checkpoint"]
        # Refresh report-only save timing from the immutable epoch log: it is
        # measured after serializing the corresponding recovery checkpoint.
        old_log=Path(resume).parent/"curves.jsonl"
        if old_log.exists():
            recorded=[json.loads(line) for line in old_log.read_text().splitlines()]
            if [r["epoch"] for r in recorded[:len(curves)]]==[r["epoch"] for r in curves]:
                curves=recorded[:len(curves)]
    else:
        initial_stats=evaluate(model,val_loader,device,smoke)
        curves=[]; best=dict(acc1=-1.,epoch=None); start_epoch=0; worker_states=None
        training_seconds=0.; best_path=str(output/"best_checkpoint.pth")
    ctx=torch.multiprocessing.get_context("fork"); queue=ctx.Queue()
    wrapped=WorkerStateDataset(train_data,steps,128,10,queue,worker_states)
    loader=DataLoader(wrapped,sampler=sampler,batch_size=128,num_workers=10,
        pin_memory=True,drop_last=True,persistent_workers=True)
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/"initial.json",dict(source=source,mask=mask_info,validation=initial_stats,
            identity=manifest["identity"],job=job,smoke=smoke,args=vars(a),dataset_identity=dataset_identity))
        print("FIXED_TOPOLOGY_INITIAL "+json.dumps(dict(job=job,accuracy=initial_stats,mask=mask_info)),flush=True)
    if resume_payload:
        restore_rng(resume_payload["rng_states_by_rank"][rank]["main"])
    init_seconds=time.monotonic()-started
    try:
        for epoch in range(start_epoch,epochs):
            sampler.set_epoch(epoch); scheduler.step(epoch)
            lr=optimizer.param_groups[0]["lr"]
            if abs(lr-lr_at(epoch,epochs))>1e-12: raise RuntimeError("Scheduler differs from frozen protocol")
            # Persistent loader reset consumes no new base seed. A resumed loader
            # must start fresh workers without consuming an extra main RNG draw.
            before=rng() if resume_payload and epoch==start_epoch else None
            iterator_started=time.monotonic()
            iterator=iter(loader)
            if before: restore_rng(before)
            iterator_seconds=time.monotonic()-iterator_started
            ddp.train(); count=0; loss_sum=0.; scale_before=scaler.get_scale(); actual_before=actual_steps[0]
            torch.cuda.synchronize(); t0=time.monotonic()
            stream=itertools.islice(iterator,2) if smoke else iterator
            scale_events=[]; previous_scale=scale_before
            for images,targets in stream:
                images=images.to(device,non_blocking=True); targets=targets.to(device,non_blocking=True)
                images,targets=mixup(images,targets)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=ddp(images); loss=criterion(logits,targets)
                lv=loss.item()
                if not math.isfinite(lv): raise RuntimeError(f"Nonfinite loss at epoch {epoch+1}, batch {count}")
                optimizer.zero_grad()
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                pruner()
                torch.cuda.synchronize(); ema.update(ddp)
                current_scale=scaler.get_scale()
                if current_scale!=previous_scale:
                    scale_events.append(dict(batch=count+1,before=previous_scale,after=current_scale))
                previous_scale=current_scale
                loss_sum+=lv; count+=1
                if rank==0 and count%100==0: print(f"Epoch {epoch+1}/{epochs} batch {count}/{steps} loss={lv:.4f}",flush=True)
            torch.cuda.synchronize(); elapsed=time.monotonic()-t0
            measured=torch.tensor(elapsed,device=device); dist.all_reduce(measured,op=dist.ReduceOp.MAX)
            elapsed=measured.item(); training_seconds+=elapsed
            if count!=(2 if smoke else steps): raise RuntimeError("Training batch coverage mismatch")
            successful=actual_steps[0]-actual_before
            amp=dict(attempted=count,successful=successful,skipped=count-successful,
                scale_before=scale_before,scale_after=scaler.get_scale(),scale_events=scale_events)
            amp_all=[None]*world; dist.all_gather_object(amp_all,amp)
            if any(x!=amp for x in amp_all): raise RuntimeError("AMP step accounting differs across ranks")
            local_workers={}
            if not smoke:
                for _ in range(10):
                    wid,value=queue.get(timeout=60)
                    if wid in local_workers: raise RuntimeError("Duplicate worker RNG state")
                    local_workers[wid]=pickle.loads(value)
            local_audit=audit_fixed_topology(pruner.W,masks,optimizer)
            if (not same_masks(pruner.backward_masks,masks)
                    or local_audit["masked_weight_nonzero"] or local_audit["masked_moment_nonzero"]
                    or not pruner.static_topo or pruner.dst_steps!=0):
                raise RuntimeError(f"Fixed topology invariant failed: {local_audit}")
            parameters=list(model.parameters())+list(ema.ema.parameters())
            state_tensors=[t for s in optimizer.state.values() for t in s.values()]
            flags=torch.tensor(int(finite(parameters+state_tensors)),device=device)
            dist.all_reduce(flags,op=dist.ReduceOp.MIN)
            if not flags.item(): raise RuntimeError("Nonfinite parameter/EMA/optimizer state")
            if any(p.dtype!=torch.float32 for p in model.parameters()): raise RuntimeError("Non-FP32 parameter")
            if any(p.grad is not None and p.grad.dtype!=torch.float32 for p in model.parameters()):
                raise RuntimeError("Non-FP32 gradient")
            if any(t.dtype!=torch.float32 for t in state_tensors if torch.is_tensor(t) and t.is_floating_point()):
                raise RuntimeError("Non-FP32 optimizer state")
            t=time.monotonic(); validation=evaluate(model,val_loader,device,smoke)
            val_seconds=time.monotonic()-t
            if validation["acc1"]>best["acc1"]:
                best=dict(validation,epoch=epoch+1); improved=True
            else: improved=False
            peaks=torch.tensor([torch.cuda.max_memory_allocated()/1024**3,
                torch.cuda.max_memory_reserved()/1024**3],device=device,dtype=torch.float64)
            dist.all_reduce(peaks,op=dist.ReduceOp.MAX)
            record=dict(epoch=epoch+1,lr=lr,train_loss=loss_sum/count,validation=validation,
                iterator_start_seconds=iterator_seconds,
                training_loop_seconds=elapsed,validation_seconds=val_seconds,amp=amp,mask_audit=local_audit,
                max_allocated_gib=peaks[0].item(),max_reserved_gib=peaks[1].item(),
                memory_scope="maximum-across-all-eight-ranks",tf32=False,training_autocast="fp16",
                parameter_gradient_optimizer_dtype="fp32")
            curves.append(record)
            per_rank=[None]*world
            dist.all_gather_object(per_rank,dict(main=rng(),workers=local_workers))
            payload=dict(identity=manifest["identity"],job=job,model=model.state_dict(),
                optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                model_ema=get_state_dict(ema),pruner=pruner.state_dict(),masks=masks,mask=mask_info,
                dataset_identity=dataset_identity,
                rng_states_by_rank=per_rank,epoch=epoch,curves=curves,initial=initial_stats,
                best=best,best_checkpoint=best_path,training_loop_seconds=training_seconds,smoke=smoke,args=vars(a))
            t=time.monotonic()
            if rank==0:
                if improved: atomic_torch(output/"best_checkpoint.pth",payload); best_path=str(output/"best_checkpoint.pth")
                payload["best_checkpoint"]=best_path
                atomic_torch(output/"recovery.pth",payload)
                if epoch==epochs-1 or smoke: atomic_torch(output/"final_checkpoint.pth",payload)
                # Validate serialized full state without altering training RNG.
                if smoke:
                    loaded=torch.load(output/"recovery.pth",map_location="cpu",weights_only=False)
                    if loaded["identity"]!=payload["identity"] or loaded["mask"]!=mask_info:
                        raise RuntimeError("Checkpoint round-trip identity failed")
                    for name,value in model.state_dict().items():
                        if not torch.equal(value.cpu(),loaded["model"][name]):
                            raise RuntimeError("Checkpoint model round-trip failed")
            dist.barrier()
            record["checkpoint_seconds"]=time.monotonic()-t
            if rank==0:
                atomic_json(output/"progress.json",dict(identity=manifest["identity"],job=job,
                    epoch=epoch+1,best=best,latest=record,checkpoint=str(output/"recovery.pth"),smoke=smoke))
                with (output/"curves.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
            dist.barrier()
            if rank==0: print("EPOCH_COMPLETE "+json.dumps(dict(record,checkpoint_seconds=time.monotonic()-t)),flush=True)
            if smoke: break
        if rank==0:
            summary=dict(identity=manifest["identity"],job=job,source=source,mask=mask_info,
                initial=initial_stats,best=best,final=dict(validation,epoch=epoch+1),
                best_checkpoint=best_path,final_checkpoint=str(output/"final_checkpoint.pth"),
                training_loop_seconds=training_seconds,initialization_seconds=init_seconds,
                wall_seconds=time.monotonic()-started,epochs=curves,smoke=smoke,
                recovery_worker_rng_complete=not smoke)
            atomic_json(output/"summary.json",summary)
    finally:
        shutdown(loader); handle.remove(); queue.close()
