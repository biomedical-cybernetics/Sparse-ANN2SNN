from unittest.mock import patch
import torch
from torch import nn
from runtime import atomic_json, configuration_data, save_checkpoint_atomic
from vgg_dual_protocol import Job
from vgg_dual_model import model_for, load_model, loaders, fixed_ratio_prune, inventory
import cnn_dst_gd_worker as shared

def ann(args, job):
    from resnet_matched_model import scheduler_for
    if job.method!="depgraph":
        from cnn_dst_gd_model import Topology as ParentTopology, LinearPruner, ConvMask
        class Topology(ParentTopology):
            def __init__(self,model,optimizer,j):
                self.job,self.model,self.conv,self.linear=j,model,[],None
                if j.method!="dense":
                    self.conv=[ConvMask(m,n+".weight",j) for n,m in model.named_modules() if isinstance(m,nn.Conv2d)]
                    if j.variant=="linear99":self.linear=LinearPruner(model,optimizer,j)
        with patch.multiple(shared,Job=Job,ordinary_model=model_for,loaders=loaders,Topology=Topology,
                            optimizer_scheduler=lambda model,j:scheduler_for(model,300,.1,5)):
            shared.run(args)
        if args.smoke:
            for suffix in ("model","masks","pruner"):
                save_checkpoint_atomic(args.artifact_dir/("best_"+suffix+".pth"),torch.load(
                    args.artifact_dir/("unrestricted_best_"+suffix+".pth"),map_location="cpu",weights_only=False))
        state=torch.load(args.artifact_dir/"checkpoints/last_checkpoint.pth",map_location="cpu",weights_only=False)
        save_checkpoint_atomic(args.artifact_dir/"final_recovery.pth",state)
        return
    import resnet_matched_worker as trainer
    train,test=loaders(job,args.data_root);shared.configure(job.seed)
    model=model_for(job).cuda();state=None
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=False)
        if state.get("job")!=job.__dict__ or state.get("configuration")!=configuration_data(job.config()) or state.get("probe"):
            raise ValueError("Wrong DepGraph resume identity")
        model=state["model"].cuda()
    if state is None or state["phase"]=="pretrain":
        pre=trainer.train_phase(args,job,model,train,test,"pretrain",200,.1,5,resume=args.resume)
        model=load_model(args.artifact_dir/"pretrain_best_model.pth",job).cuda()
        model,audit=fixed_ratio_prune(model,next(iter(test))[0][:1].cuda(),job)
        trainer.persist_model(args.artifact_dir/"pruned_model.pth",model)
        context={"pretrain":pre,"budget":audit}
    else:
        if state["phase"]!="finetune":raise ValueError("Unknown DepGraph resume phase")
        context=state["context"]
        for n,v in state["best_artifacts"].items():save_checkpoint_atomic(args.artifact_dir/n,v)
    atomic_json(args.metadata_dir/"budget_audit.json",context["budget"])
    fine=trainer.train_phase(args,job,model,train,test,"finetune",100,.01,5,
        resume=args.resume if state and state["phase"]=="finetune" else None,context=context)
    best=load_model(args.artifact_dir/"finetune_best_model.pth",job)
    for prefix,value in (("best",best),("unrestricted_best",best),("final",model.cpu())):
        trainer.persist_model(args.artifact_dir/(prefix+"_model.pth"),value)
        save_checkpoint_atomic(args.artifact_dir/(prefix+"_masks.pth"),{})
        save_checkpoint_atomic(args.artifact_dir/(prefix+"_pruner.pth"),{})
    state=torch.load(args.artifact_dir/"checkpoints/last_checkpoint.pth",map_location="cpu",weights_only=False)
    save_checkpoint_atomic(args.artifact_dir/"final_recovery.pth",state)
    atomic_json(args.metadata_dir/"metrics.json",{**fine,"pretrain":context["pretrain"],"budget_audit":context["budget"],
        "phase":"ann","unrestricted":fine["best"],"conv_updates":[],"linear_updates":0})
    atomic_json(args.metadata_dir/"best_export_audit.json",{
        "layers":inventory(best),"mask_based_macs":context["budget"]["macs"],"basis":"physical graph"})
