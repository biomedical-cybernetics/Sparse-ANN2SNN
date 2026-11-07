import json, os
from pathlib import Path
import torch
import cnn_dst_gd_worker as shared
from runtime import atomic_json, configuration_data, capture_rng_state, restore_rng_state, save_checkpoint_atomic, preserve_checkpoint_link
from resnet_matched_model import model_for, scheduler_for
from resnet_topology_protocol import Job, ONESHOT
from resnet_topology_model import install, same_state
from reference_inputs import model_reference

def run(args):
    cfg=json.loads(args.config.read_text()); job=Job(**cfg["job"])
    if cfg["configuration"]!=configuration_data(job.config()): raise ValueError("Wrong configuration")
    reference,info=model_reference(args.reference_bundle,job.seed)
    masks=torch.load(reference/"best_masks.pth",map_location="cpu",weights_only=True)
    args.metadata_dir.mkdir(parents=True,exist_ok=True); args.artifact_dir.mkdir(parents=True,exist_ok=True)
    shared.Job=Job
    def ordinary(j):
        model=model_for(j)
        def precision_boundary(module,inputs,output):
            expected=torch.bfloat16 if module.training and j.method not in ONESHOT and j.training_precision == "bf16" else torch.float32
            if output.dtype!=expected: raise ValueError("Training/evaluation precision boundary violated")
        model.last_layer.register_forward_hook(precision_boundary)
        return model
    shared.ordinary_model=ordinary
    shared.optimizer_scheduler=lambda model,j:scheduler_for(model,j.epochs,.1)
    def topology(model,optimizer,j):
        result=install(model,optimizer,j,masks)
        expected_initial = info.pop("initial_state")
        if j.method not in ONESHOT and not same_state(result.initial_state, expected_initial):
            raise ValueError("Original same-seed initial weights/buffers differ")
        del result.initial_state, expected_initial
        save_checkpoint_atomic(args.artifact_dir/"initial_masks.pth",shared.cpu_copy(result.initial_masks))
        atomic_json(args.metadata_dir/"initialization.json",{"mask_seed":j.seed+100000,
            "mask_diagnostics":result.diagnostics,"source_best":info["best"],
            "reference_artifacts":str(reference),"trained_weights_inherited":j.method in ONESHOT})
        return result
    shared.Topology=topology
    if job.method not in ONESHOT:
        shared.run(args)
        if not args.smoke:
            # Retain the complete final recovery state outside replaceable intervals.
            # A hard link avoids a second inventory/copy of this large checkpoint.
            preserve_checkpoint_link(args.artifact_dir/"checkpoints/epoch-0200.pth",args.artifact_dir/"final_state.pth")
        return
    if args.resume: raise ValueError("One-shot has no optimizer or resume training")
    shared.configure(job.seed); _,test=shared.loaders(job,args.data_root)
    shared.configure(job.seed); model=ordinary(job).cuda()
    dense,dense_info=model_reference(args.reference_bundle,job.seed,"dense")
    del dense_info
    model.load_state_dict(torch.load(dense/"best_model.pth",map_location="cpu",weights_only=True))
    rng=capture_rng_state(); result=topology(model,None,job); restore_rng_state(rng)
    before=shared.cpu_copy(model.state_dict())
    acc,loss=shared.evaluate(model,test,1 if args.smoke else None)
    if any(not torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items()):
        raise ValueError("One-shot evaluation changed weights or BN buffers")
    x,_=next(iter(test)); audit=shared.export_audit(job,result,x[:1].cuda())
    shared.save_export("best",result,args.artifact_dir)
    atomic_json(args.metadata_dir/"best_export_audit.json",audit)
    atomic_json(args.metadata_dir/"metrics.json",{"job":job.__dict__,"configuration":configuration_data(job.config()),
        "probe":args.smoke,"best":{"top1":acc,"loss":loss,"epoch":None},"no_fine_tune":True,
        "optimizer_steps":0,"history":[],
        "export_audit":audit,"conv_updates":[0]*17,"training_precision":"not-applicable"})
    atomic_json(args.metadata_dir/("smoke-complete.json" if args.smoke else ".complete.json"),{"probe":args.smoke})
