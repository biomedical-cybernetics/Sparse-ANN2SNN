import argparse, copy, json, time
from pathlib import Path
import torch
from training_options import training_context
from torch import nn
from runtime import atomic_json, configuration_data, capture_rng_state, restore_rng_state, save_checkpoint_atomic
import cnn_dst_gd_worker as shared
from resnet_matched_protocol import Job
from resnet_matched_model import model_for, scheduler_for, target_budget, calibrate_budget, weight_count
from mask_utils import apply_masks

def load_model(path, job):
    content = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(content, nn.Module): return content.cuda()
    model = model_for(job).cuda(); model.load_state_dict(content, strict=True); return model

def finite_state(model, optimizer):
    values = list(model.parameters()) + [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    if any(not torch.isfinite(v).all() for v in values): raise FloatingPointError("Nonfinite model/state")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in values): raise ValueError("Parameters/state must be FP32")

def persist_model(path, model):
    save_checkpoint_atomic(path, copy.deepcopy(model).cpu())

def train_phase(args, job, model, train, test, phase, epochs, lr, warmup, *, masks=None, resume=None, context=None):
    masks = masks or {}; context = context or {}
    optimizer, scheduler = scheduler_for(model, epochs, lr, warmup)
    history = []; best = None; start = steps = 0
    identity = {"job": job.__dict__, "configuration": configuration_data(job.config()), "phase": phase}
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if any(payload.get(k) != v for k,v in identity.items()) or payload.get("probe"):
            raise ValueError("Resume phase/config identity mismatch")
        model.load_state_dict(payload["model"].state_dict()); optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"]); history=payload["history"]; best=payload["best"]
        start=payload["epoch"]; steps=payload["optimizer_steps"]; context=payload["context"]
        for name, value in payload["best_artifacts"].items(): save_checkpoint_atomic(args.artifact_dir/name, value)
        restore_rng_state(payload["rng"])
    best_path = args.artifact_dir/f"{phase}_best_model.pth"
    final_path = args.artifact_dir/f"{phase}_final_model.pth"
    for epoch in range(start+1, (start+1 if args.smoke else epochs)+1):
        model.train(); used_lr=optimizer.param_groups[0]["lr"]; total=0; loss_sum=0.
        for i,(x,y) in enumerate(train):
            if args.smoke and i>=2: break
            x,y=x.cuda(),y.cuda(); optimizer.zero_grad(set_to_none=True)
            with training_context(job):
                output=model(x); loss=nn.functional.cross_entropy(output,y)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
            loss.backward(); optimizer.step(); apply_masks(model,masks); steps+=1
            loss_sum+=float(loss.detach())*len(y); total+=len(y)
        scheduler.step()
        acc,loss=shared.evaluate(model,test,1 if args.smoke else None)
        row={"epoch":epoch,"lr":used_lr,"test_top1":acc,"test_loss":loss,"train_loss":loss_sum/total,"budget_qualified":True}
        history.append(row)
        if best is None or acc>best["top1"]:
            best={"top1":acc,"loss":loss,"epoch":epoch}; persist_model(best_path,model)
        print(json.dumps({"phase":phase,**row}),flush=True)
        if epoch%20==0 or epoch==epochs or args.smoke:
            finite_state(model,optimizer)
            saved={p.name:torch.load(p,map_location="cpu",weights_only=False) for p in args.artifact_dir.glob("*best_model.pth")}
            if (args.artifact_dir/"pruned_model.pth").is_file():
                saved["pruned_model.pth"]=torch.load(args.artifact_dir/"pruned_model.pth",map_location="cpu",weights_only=False)
            payload={**identity,"model":copy.deepcopy(model).cpu(),"optimizer":shared.cpu_copy(optimizer.state_dict()),
                     "scheduler":scheduler.state_dict(),"rng":capture_rng_state(),"history":history,"best":best,
                     "epoch":epoch,"optimizer_steps":steps,"best_artifacts":saved,"context":context,"probe":args.smoke,
                     "masks":shared.cpu_copy(masks)}
            directory=args.artifact_dir/"checkpoints"; directory.mkdir(exist_ok=True)
            save_checkpoint_atomic(directory/f"{phase}-epoch-{epoch:04d}.pth",payload)
            save_checkpoint_atomic(directory/"last_checkpoint.pth",payload)
    persist_model(final_path,model); finite_state(model,optimizer)
    save_checkpoint_atomic(args.artifact_dir/f"{phase}_rng.pth",capture_rng_state())
    result={**identity,"best":best,"history":history,"optimizer_steps":steps,"probe":args.smoke,
            "scaler_enabled":False,"state_dtype":"float32","all_parameters_finite":True,"context":context,
            "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}}
    return result

def depgraph(args,job):
    train,test=shared.loaders(job,args.data_root,args.smoke); shared.configure(job.seed)
    model=model_for(job).cuda(); payload=None; pre=None
    if args.resume:
        payload=torch.load(args.resume,map_location="cpu",weights_only=False)
        if payload.get("job")!=job.__dict__ or payload.get("configuration")!=configuration_data(job.config()) or payload.get("probe"):
            raise ValueError("DepGraph resume identity mismatch")
        model=payload["model"].cuda()
        pre=payload.get("context",{}).get("pretrain")
    if payload is None or payload["phase"]=="pretrain":
        pre=train_phase(args,job,model,train,test,"pretrain",133,.1,5,resume=args.resume)
        model=load_model(args.artifact_dir/"pretrain_best_model.pth",job)
        target=target_budget(job)["active_weights"]
        example=next(iter(test))[0][:1].cuda()
        try: model,audit=calibrate_budget(model,example,target)
        except Exception as error:
            atomic_json(args.metadata_dir/"budget_failure.json",{"target":target,"error":repr(error),"audit":getattr(error,"audit",None)}); raise
        import torch_pruning as tp
        macs,_=tp.utils.count_ops_and_params(model,example_inputs=example)
        audit["macs"]=float(macs); audit["layers"]=[{"layer":n,"shape":list(m.weight.shape)} for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
        if not (args.artifact_dir/"pruned_model.pth").exists(): persist_model(args.artifact_dir/"pruned_model.pth",model)
        atomic_json(args.metadata_dir/"budget_audit.json",audit)
        context={"pretrain":pre,"budget":audit}
    else:
        if payload["phase"]!="finetune": raise ValueError("Unknown DepGraph resume phase")
        context=payload["context"]; atomic_json(args.metadata_dir/"budget_audit.json",context["budget"])
        for name,value in payload["best_artifacts"].items(): save_checkpoint_atomic(args.artifact_dir/name,value)
        persist_model(args.artifact_dir/"pruned_model.pth",model)
    fine=train_phase(args,job,model,train,test,"finetune",67,.01,5,
        resume=args.resume if payload and payload["phase"]=="finetune" else None,context=context)
    best=load_model(args.artifact_dir/"finetune_best_model.pth",job)
    persist_model(args.artifact_dir/"best_model.pth",best); persist_model(args.artifact_dir/"final_model.pth",model)
    persist_model(args.artifact_dir/"unrestricted_best_model.pth",best)
    save_checkpoint_atomic(args.artifact_dir/"best_masks.pth",{})
    save_checkpoint_atomic(args.artifact_dir/"best_pruner.pth",{})
    result={**fine,"phase":"ann","pretrain":pre,"budget_audit":context["budget"]}
    atomic_json(args.metadata_dir/"metrics.json",result)
