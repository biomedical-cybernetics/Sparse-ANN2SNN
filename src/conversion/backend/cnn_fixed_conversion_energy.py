"""Generated-event SOP counting across source CNN graph and temporal layouts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fixed_conversion_energy import EventCounter as BaseCounter


def fanout(weight,input_shape,kwargs=None):
    """Exact input-position connectivity, computed off the inference GPU."""
    mask=(weight.detach().cpu()!=0).float()
    if kwargs is None:return mask.sum(0)
    def pair(v):return (v,v) if isinstance(v,int) else tuple(v)
    stride,padding,dilation=[pair(kwargs[k]) for k in ("stride","padding","dilation")]
    spatial=input_shape[-2:];kernel=weight.shape[-2:]
    output=tuple((spatial[i]+2*padding[i]-dilation[i]*(kernel[i]-1)-1)//stride[i]+1 for i in (0,1))
    extra=tuple(spatial[i]-((output[i]-1)*stride[i]-2*padding[i]+dilation[i]*(kernel[i]-1)+1) for i in (0,1))
    value=F.conv_transpose2d(torch.ones(1,len(mask),*output),mask,
        stride=stride,padding=padding,output_padding=extra,dilation=dilation,groups=kwargs["groups"])[0]
    if tuple(value.shape)!=tuple(input_shape):raise ValueError("Connectivity fan-out shape mismatch")
    return value


class EventCounter(BaseCounter):
    def __init__(self,model,time,parallel=False,dcgs=False):
        self.dcgs=dcgs
        self.fanouts={}
        super().__init__(model,time,parallel or dcgs)
        for name,module in model.named_modules():
            if getattr(module,"event_source",False):
                self.events[name]=np.zeros(time)
                self.handles.append(module.register_forward_hook(self.event_hook(name)))

    def temporal(self,events):
        slots=self.time+1 if self.dcgs else self.time
        if len(events)%slots:raise ValueError("Generated event temporal axis mismatch")
        result=events.reshape((slots,len(events)//slots)+events.shape[1:])
        return result[1:] if self.dcgs else result

    def event_hook(self,name):
        def hook(module,inputs,output):
            events=getattr(output,"_conversion_events",None)
            if events is None:raise ValueError("Missing generated event tensor")
            values=getattr(output,"_conversion_cpu_totals",None)
            if values is None:raise ValueError("Missing intrinsic CPU event summary")
            if self.dcgs:values=values[1:]
            counts=values.reshape(self.time if self.parallel else 1,-1).sum(1).numpy()
            if self.parallel:self.events[name]+=counts
            else:self.events[name][self.step]+=counts[0]
        return hook

    def pool_hook(self,module,inputs,output):
        values=getattr(inputs[0],"_conversion_cpu_totals",None)
        self.pooled_events=None if values is None else values.sum((-2,-1))

    def prehook(self,name):
        def hook(module,inputs):
            x=inputs[0];counts=getattr(x,"_conversion_cpu_totals",None)
            analog=getattr(x,"_conversion_summary_analog",False)
            drop_baseline=self.dcgs
            if counts is None and x.ndim==2:counts=self.pooled_events
            if counts is None:
                counts=getattr(module,"_conversion_root_analog_totals",None)
                if counts is not None:drop_baseline=False
                else:
                    # Analog-only graph edges: copies allocate host memory,
                    # not GPU tensors or cuDNN convolution workspaces.
                    values=(x.detach().cpu()!=0)
                    slots=self.time+1 if self.dcgs else self.time if self.parallel else 1
                    counts=values.reshape(slots,len(values)//slots,*values.shape[1:]).sum(1).double()
                analog=True
            if drop_baseline:counts=counts[1:]
            kwargs=getattr(module,"fwd_kwargs",None)
            if isinstance(module,nn.Conv2d):kwargs={k:getattr(module,k) for k in ("stride","padding","dilation","groups")}
            if isinstance(module,nn.Linear):kwargs=None
            key=(name,tuple(counts.shape[1:]))
            if key not in self.fanouts:self.fanouts[key]=fanout(module.weight,counts.shape[1:],kwargs).double()
            values=(counts.reshape(len(counts),-1)@self.fanouts[key].reshape(-1)).numpy()
            kind="mac" if analog else "sop"
            if self.parallel:self.rows[name][kind]+=values
            else:self.rows[name][kind][self.step]+=values[0]
        return hook
