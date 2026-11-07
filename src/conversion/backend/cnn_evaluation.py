import json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from runtime import atomic_json,capture_rng_state,restore_rng_state
from device import configure,device_model
from inference import forward_logits,reset_inference_state
from unified_sop_counter import StrictCounter,attach_transport,validate_result

def load_model(entry):
    from cnn_pipeline import model_for,parallel_adapter
    job=SimpleNamespace(**entry['job'])
    source=entry['source']
    if entry['kind']=='direct':
        from direct_model import _model_class
        model=_model_class(Path(source['braincog_root']),entry['baseline'])(step=job.time,encode_type='direct',num_classes=job.classes)
        state=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        if 'state_dict' in state:state=state['state_dict']
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        return model.cuda().eval(),job
    if job.converter=='parallel':
        adapter=parallel_adapter(job);adapter.install('parallel')
        model=torch.load(source['converted'],map_location='cpu',weights_only=False)
        if isinstance(model,dict):model=model['model']
        model=adapter.ensure_summaries(model,job)
        return device_model(model).eval(),job
    ann=model_for(job)
    value=torch.load(source['ann'],map_location='cpu',weights_only=False)
    if isinstance(value,nn.Module):ann=value
    else:
        if isinstance(value,dict) and 'model' in value:value=value['model']
        ann.load_state_dict(value,strict=True)
    import cnn_fixed_conversion_adapters as adapter
    original=adapter.dcgs_args
    def args(j):
        a=original(j)
        if j.architecture=='6Conv2FC':a.architecture='STBPANN';a.model_name='stbpann';a.dim=1
        elif j.architecture=='VGG-16-Linear99':a.architecture='VGG-16';a.model_name='vgg16'
        return a
    adapter.dcgs_args=args
    return adapter.dcgs_model(ann,source['threshold'],job).eval(),job

def forward(model,x,job,direct):
    if direct:
        from direct_model import _forward
        output=_forward(model,x)
        return [output[:t+1].mean(0) for t in range(job.time)]
    return forward_logits(model,x,job)

def reset(model,job,direct):
    if direct:model.reset()
    else:reset_inference_state(model,job)


def dataset(entry,root):
    j=entry['job']
    if j['dataset']=='DVSGesture':
        from Models.load_data_dvs import DVSGesture
        return DVSGesture(str(Path(root)/'dvsgesture'),one_hot_label=False)[1]
    from torchvision.datasets import CIFAR10,CIFAR100
    from torchvision import transforms
    norm=entry['source'].get('normalization')
    if norm is None:
        if entry['kind']=='direct':norm={'mean':[.4914,.4822,.4465],'std':[.2023,.1994,.2010]}
        else:norm={'mean':[.4914,.4822,.4465],'std':[.247,.2435,.2616]} if j['dataset']=='CIFAR10' else {'mean':[.5071,.4866,.4409],'std':[.2673,.2564,.2762]}
    cls=CIFAR10 if j['dataset']=='CIFAR10' else CIFAR100
    return cls(root,train=False,download=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])]))

@torch.no_grad()
def evaluate(e,batch,data_root,output,probe):
    started=time.monotonic();configure(2026);torch.cuda.reset_peak_memory_stats()
    model,j=load_model(e);direct=e['kind']=='direct';ds=dataset(e,data_root)
    assert len(ds)==e['samples']
    loader=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,drop_last=False)
    first=next(iter(loader))[0].cuda().float()
    rng=capture_rng_state();forward(model,first,j,direct);reset(model,j,direct);restore_rng_state(rng)
    # Check intrinsic annotations against the original converted forward too.
    changed=[]
    if j.is_dcgs:
        from cnn_fixed_conversion_adapters import TrackedMTHDiff
        from DCGS.neurons import MTH_diff
        for module in model.modules():
            if isinstance(module,TrackedMTHDiff):changed.append((module,type(module)));module.__class__=MTH_diff
    try:base=forward(model,first,j,direct)
    finally:
        for module,cls in changed:module.__class__=cls
    handles=attach_transport(model,j.time+1 if j.is_dcgs else j.time,direct)
    repeated=forward(model,first,j,direct)
    counter=StrictCounter(model,j.time,j.is_dcgs,direct)
    observed=forward(model,first,j,direct);counter.close()
    proof={'repeat_equal':all(torch.equal(a,b) for a,b in zip(base,repeated)),
        'monitored_equal':all(torch.equal(a,b) for a,b in zip(repeated,observed)), 'batch':len(first)}
    atomic_json(output/'monitor-proof.json',proof)
    if not proof['repeat_equal'] or not proof['monitored_equal']:raise ValueError('Monitor changed outputs')
    del base,repeated,observed,first
    counter=StrictCounter(model,j.time,j.is_dcgs,direct);n=0;correct=np.zeros(j.time);pred=[];batch_sizes=[]
    timed=time.monotonic()
    try:
        for bi,(x,y) in enumerate(loader):
            values=forward(model,x.cuda().float(),j,direct)
            if not all(torch.isfinite(v).all() for v in values):raise ValueError('Nonfinite logits')
            for t,v in enumerate(values):correct[t]+=int((v.argmax(1).cpu()==y).sum())
            out=values[-1].cpu();pred+=out.argmax(1).tolist()
            n+=len(y);batch_sizes.append(len(y));del values,out
            atomic_json(output/'progress.json',{'samples':n,'expected':e['samples'],'elapsed':time.monotonic()-timed})
            if probe and bi==1:break
        result=counter.result(n)
    finally:counter.close()
    for h in handles:h.remove()
    result.update(id=e['id'],job=e['job'],samples=n,sample_ids=list(range(n)),batch=batch,batch_sizes=batch_sizes,predictions=pred,
        top1=float(correct[-1]*100/n),trajectory=(correct*100/n).tolist(),monitor_parity=proof,source=e['source'],probe=probe,
        elapsed_seconds=time.monotonic()-started,inference_seconds=time.monotonic()-timed,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        environment={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'tf32':False,'amp':False},
        warmup={'samples':min(batch,len(ds)),'excluded':True,'state_reset':True,'rng_restored':True})
    validate_result(result,n,j.time)
    atomic_json(output/'result.json',result)
    atomic_json(output/'complete.json',{'id':e['id'],'probe':probe})
    print(json.dumps({'id':e['id'],'top1':result['top1'],'energy':result['energy_uj'][-1],'samples':n,'seconds':result['elapsed_seconds']}),flush=True)
