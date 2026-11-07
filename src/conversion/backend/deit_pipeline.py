"""DeiT DCGS threshold calibration and selected accuracy/SOP evaluation."""
from pathlib import Path
from types import SimpleNamespace
import os
import torch
from torch import nn
import torch.distributed as dist
import numpy as np
from scipy.io import savemat
from runtime import atomic_json,save_checkpoint_atomic
from DCGS.models.ViT import vit_base_patch16_224
from DCGS.converter import Converter,Threshold_Getter
from DCGS.forwards import forward_replace
from DCGS.utils import seed_all,reset
from Models import load_data_imagenet,count_actiavtions
from merge_qkv import merge_qkv_in_state_dict
from deit_evaluator import make_dcgs_evaluator


def checkpoint_state(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if isinstance(state,nn.Module):return state
    for key in ('model','module'):
        if key in state:state=state[key];break
    return merge_qkv_in_state_dict(state)


def run(config,runtime):
    p=config['parameters'];output=Path(runtime['output_dir']);output.mkdir(parents=True,exist_ok=True)
    if int(os.environ.get('WORLD_SIZE','1'))!=p['world_size']:
        raise ValueError('Use torchrun with '+str(p['world_size'])+' processes for this recipe')
    local_rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
    if not dist.is_initialized():dist.init_process_group(backend='nccl',init_method='env://')
    converting=config['stage']=='convert'
    if p.get('disable_tf32', False):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    args=SimpleNamespace(architecture='vit_base',model_name='vit_base',task='classification',dataset='imagenet',
        mode='get_threshold' if converting else 'test_snn',threshold_mode='99.9%' if converting else 'var',
        threshold_level='channel' if converting else p['inference_threshold_level'],neuron_name='IF' if converting else 'MTH_diff_line',
        coding_type='rate' if converting else 'diff_rate',step_mode='m',num_thresholds=8,threshold_scale=4.,
        time=1 if converting else 32,distributed=True,dist_eval=True,bs=p['configured_batch_size'],fx=False,
        fuse=False,tau=.98,pruning='',num_workers=16,pin_mem=True)
    seed_all(p['calibration_seed'] if converting else p['inference_seed'])
    model=vit_base_patch16_224(num_classes=1000)
    _,loader,_=load_data_imagenet(args,runtime['data_root'])
    if len(loader.dataset)!=50000:raise ValueError('Expected all 50000 ImageNet validation images')
    num_activations=count_actiavtions(model)
    checkpoint=runtime.get('checkpoint') or runtime['inputs']['checkpoint']
    state=checkpoint_state(checkpoint)
    if converting:
        if isinstance(state,nn.Module):model=state
        else:model.load_state_dict(state,strict=True)
        model.to(device).eval();model=Converter.change_maxpool_before_relu(model)
        with torch.no_grad():
            model=Threshold_Getter(loader,mode=args.threshold_mode,level=args.threshold_level,device=device,momentum=.1,output_fx=False)(model)
        if dist.get_rank()==0:
            save_checkpoint_atomic(output/'thre.pth',model.state_dict())
            atomic_json(output/'calibration.json',{'configuration':p,'samples_per_rank':len(loader.sampler),
                'saved_rank':0,'sample_ids':list(iter(loader.sampler))})
        dist.barrier();return
    if isinstance(state,nn.Module):
        model=Converter.replace_by_maxpool_neuron(state,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
    else:
        model=Converter.change_maxpool_before_relu(model)
        model=Converter.replace_by_maxpool_neuron(model,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
        model=Threshold_Getter.replace_nonlinear_by_hook(model,momentum=.1,mode=args.threshold_mode,level=args.threshold_level)
        model.load_state_dict(state,strict=True)
    model=Threshold_Getter.get_scale_from_var(model,T=args.time*(2**args.num_thresholds))
    model=Converter(args.neuron_name,args,T=args.time,step_mode=args.step_mode,fuse_flag=False)(model)
    model=forward_replace(args,model).to(device)
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank]).eval()
    mac_layers=[name for name,layer in model.named_modules() if isinstance(layer,nn.Conv2d) and layer.in_channels==3]
    energy=p['measurement']=='energy'
    from statics import SparseSOPMonitor
    capture={}
    evaluator=make_dcgs_evaluator(monitor_type=SparseSOPMonitor,reset=reset,capture=capture,
        max_batches=10 if energy else None,collect_operations=energy)
    with torch.no_grad():accuracy,activity,_=evaluator(model,loader,device,args,num_activations=num_activations,MACLayers=mac_layers)
    if dist.get_rank()==0:
        expected=1480 if energy else 50000
        if int(capture['samples'])!=expected:raise ValueError('Evaluation sample count differs from recipe')
        payload={'snn_acc':accuracy,'LASFR':activity,**(capture if energy else {'samples':capture['samples']})}
        if energy:payload['energy_uj']=capture['SOPOps']*1.5e-6
        savemat(output/'result.mat',payload)
        result={'configuration':p,'samples':int(capture['samples']),'trajectory_percent':(accuracy*100).tolist(),
                'top1':float(accuracy[7]*100)}
        if energy:
            result.update(sop=capture['SOPOps'].tolist(),mac=capture['MACOps'].tolist(),
                energy_uj=(capture['SOPOps']*1.5e-6).tolist(),layer_names=capture['LayerNames'].tolist(),
                layer_sop=capture['LayerSOPOps'].tolist(),layer_mac=capture['LayerMACOps'].tolist())
        atomic_json(output/'result.json',result)
    dist.barrier()
