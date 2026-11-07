import json, math, os
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from runtime import atomic_json, save_checkpoint_atomic, preserve_checkpoint_link
import cnn_dst_gd_model as models
import cnn_dst_gd_worker as training
from resnet_matched_model import model_for, scheduler_for
from resnet_sparsity_protocol import Job

def schedule(count,horizon,target):return .5+(target-.5)/(1+math.exp(-(count-horizon/2))) if count<horizon else target

class ConvMask(models.ConvMask):
    def evolve(self):
        with patch.object(models,'conv_sparsity',lambda count,horizon:schedule(count,horizon,self.job.sparsity)):
            return super().evolve()

class Topology(models.Topology):
    def __init__(self,model,opt,job):
        with patch.object(models,'ConvMask',ConvMask):super().__init__(model,opt,job)
    def inventory(self):
        rows=super().inventory(); masks=self.masks()
        for row in rows:
            mask=masks.get(row['layer']+'.weight')
            if mask is not None:
                row['qualified']=int(mask.sum())==mask.numel()-int(self.job.sparsity*mask.numel())
                row['no_input_channels']=int((mask.sum(1)==0).sum())
                row['target_sparsity']=self.job.sparsity
        return rows

def train(args):
    training.Job=Job;training.ordinary_model=model_for
    training.optimizer_scheduler=lambda m,j:scheduler_for(m,200,.1)
    def install(m,o,j):
        result=Topology(m,o,j)
        atomic_json(args.metadata_dir/'initialization.json',{'mask_seed':j.seed+100000,'trained_weights_inherited':False})
        save_checkpoint_atomic(args.artifact_dir/'initial_masks.pth',training.cpu_copy(result.masks()))
        # A small heartbeat makes a short launch check independent of epoch length.
        batches=[0]
        def heartbeat(module,inputs,output):
            if module.training:
                batches[0]+=1
                if batches[0]==1 or batches[0]%25==0:
                    atomic_json(args.metadata_dir/'heartbeat.json',{'training_batches':batches[0],'dtype':str(output.dtype),'time':__import__('time').time()})
        m.last_layer.register_forward_hook(heartbeat)
        return result
    training.Topology=install
    args.metadata_dir.mkdir(parents=True,exist_ok=True);args.artifact_dir.mkdir(parents=True,exist_ok=True)
    training.run(args)
    if not args.smoke:preserve_checkpoint_link(args.artifact_dir/'checkpoints/epoch-0200.pth',args.artifact_dir/'final_state.pth')
    atomic_json(args.metadata_dir/'memory.json',{'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'production_batch':128,'hardware':torch.cuda.get_device_name(0),'probe':args.smoke})
