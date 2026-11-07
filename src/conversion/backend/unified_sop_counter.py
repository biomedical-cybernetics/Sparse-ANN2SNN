import numpy as np
import torch
from torch import nn
from cnn_fixed_conversion_energy import EventCounter, fanout

class StrictCounter(EventCounter):
    def __init__(self, model, time, dcgs=False, direct=False):
        self.direct=direct
        self.allowed_analog={next(n for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear)) or hasattr(m,'fwd_func'))}
        self.pooled_output=None
        super().__init__(model,time,not direct,dcgs)
        self.visits={n:0 for n in self.rows}
        if direct:
            for name,module in model.named_modules():
                if type(module).__name__ in ('LIFNode','PLIFNode'):
                    self.events[name]=np.zeros(time)
                    visits=[0]
                    def event_hook(m,inputs,out,n=name,v=visits):
                        self.events[n][v[0]%time]+=float(out._conversion_cpu_totals.sum())
                        v[0]+=1
                    self.handles.append(module.register_forward_hook(event_hook))
    def pool_hook(self,module,inputs,output):
        super().pool_hook(module,inputs,output)
        # Keep the source alive so a later allocation cannot reuse its pointer.
        # ResNet's lossless post-GAP view drops Python tensor attributes.
        self.pooled_output=output
    def prehook(self,name):
        def hook(m,inputs):
            x=inputs[0];counts=getattr(x,'_conversion_cpu_totals',None)
            analog=getattr(x,'_conversion_summary_analog',False)
            baseline=self.dcgs
            if (counts is None and x.ndim==2 and self.pooled_events is not None
                and self.pooled_output is not None and x.data_ptr()==self.pooled_output.data_ptr()
                and x.numel()==self.pooled_output.numel()):
                counts=self.pooled_events
                analog=getattr(self.pooled_output,'_conversion_summary_analog',False)
            if counts is None:
                if name not in self.allowed_analog:raise ValueError('Missing generated events at '+name)
                counts=getattr(m,'_conversion_root_analog_totals',None)
                if counts is not None:baseline=False
                elif self.direct:counts=(x.detach().cpu()!=0).sum(0).double().unsqueeze(0)
                else:raise ValueError('Missing root analog summary')
                analog=True
            if baseline:counts=counts[1:]
            kwargs=None if m.weight.ndim==2 else {k:getattr(m,k) for k in ('stride','padding','dilation','groups')} if isinstance(m,nn.Conv2d) else m.fwd_kwargs
            key=(name,tuple(counts.shape[1:]))
            if key not in self.fanouts:self.fanouts[key]=fanout(m.weight,counts.shape[1:],kwargs).double()
            values=(counts.double().reshape(len(counts),-1)@self.fanouts[key].reshape(-1)).numpy()
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid operation counts')
            kind='mac' if analog else 'sop'
            if self.direct:
                slot=self.visits[name]%self.time
                self.rows[name][kind][slot]+=values[0]
            else:self.rows[name][kind]+=values
            self.visits[name]+=1
        return hook

def attach_transport(model,slots,direct=False):
    """Propagate event-count metadata without changing tensor values."""
    handles=[]
    def gap(m,ins,out):
        counts=getattr(ins[0],'_conversion_cpu_totals',None)
        if counts is None:raise ValueError('GAP lost event provenance')
        if tuple(out.shape[-2:])!=(1,1):raise ValueError('Only global 1x1 average pooling event transport is defined')
        out._conversion_cpu_totals=counts.sum((-2,-1),keepdim=True)
        out._conversion_summary_analog=getattr(ins[0],'_conversion_summary_analog',False)
    def flatten(m,ins,out):
        counts=getattr(ins[0],'_conversion_cpu_totals',None)
        if counts is None:raise ValueError('Flatten lost event provenance')
        out._conversion_cpu_totals=counts.flatten(1)
        out._conversion_summary_analog=getattr(ins[0],'_conversion_summary_analog',False)
    def binary(m,ins,out):
        values=out.detach().cpu()
        if not torch.all((values==0)|(values==1)):raise ValueError('Direct neuron/pool output is not binary')
        out._conversion_cpu_totals=values.double().sum(0).unsqueeze(0)
        out._conversion_summary_analog=False
    for m in model.modules():
        if isinstance(m,nn.AdaptiveAvgPool2d):handles.append(m.register_forward_hook(gap))
        elif isinstance(m,nn.Flatten):handles.append(m.register_forward_hook(flatten))
        elif direct and (type(m).__name__ in ('LIFNode','PLIFNode') or isinstance(m,nn.MaxPool2d)):
            handles.append(m.register_forward_hook(binary))
    return handles

def validate_result(r,samples,time):
    assert r['samples']==samples and r['sample_ids']==list(range(samples))
    assert len(r['predictions'])==samples
    for kind in ('sop','mac'):
        layers=np.asarray(r['layer_'+kind]);total=np.asarray(r[kind])
        assert layers.shape==(len(r['layer_names']),time)
        assert np.isfinite(layers).all() and (layers>=0).all()
        assert (np.diff(layers,axis=1)>=-1e-8).all()
        np.testing.assert_allclose(layers.sum(0),total,rtol=1e-12,atol=1e-6)
    np.testing.assert_allclose(np.asarray(r['sop'])*1.5e-6,r['energy_uj'],rtol=1e-12)
