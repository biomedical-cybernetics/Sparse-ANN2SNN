"""Count events where generated and propagate them to actual Conv/Linear fan-out."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def operations(events, weight, kwargs=None):
    mask = (weight != 0).to(events.dtype)
    return (F.conv2d(events, mask, None, **kwargs) if kwargs is not None else F.linear(events, mask)).sum()


class EventCounter:
    def __init__(self, model, time, parallel=False):
        self.time, self.parallel = time, parallel
        self.rows, self.events, self.handles = {}, {}, []
        self.step = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) or hasattr(module, "fwd_func"):
                self.rows[name] = {"sop": np.zeros(time), "mac": np.zeros(time)}
                self.handles.append(module.register_forward_pre_hook(self.prehook(name)))
            if isinstance(module, nn.AdaptiveAvgPool2d):
                self.handles.append(module.register_forward_hook(self.pool_hook))
            if hasattr(module,"sim_length") or hasattr(module,"pre_th"):
                self.events[name]=np.zeros(time)
                self.handles.append(module.register_forward_hook(self.event_hook(name)))
        # Source ResNet's view after GAP loses tensor attributes. Read the head's
        # explicit transported event count, without inferring it from scaled spikes.
        self.pooled_events = None

    def pool_hook(self, module, inputs, output):
        events = getattr(inputs[0], "_conversion_events", None)
        self.pooled_events = events.sum(dim=(-2, -1)) if events is not None else None

    def event_hook(self,name):
        def hook(module,inputs,output):
            events=getattr(output,"_conversion_events",None)
            if events is None:raise ValueError("Missing generated event tensor")
            if self.parallel:
                counts=events.view((self.time,len(events)//self.time)+events.shape[1:]).flatten(1).sum(1)
                self.events[name]+=counts.cpu().numpy()
            else:self.events[name][self.step]+=float(events.sum())
        return hook

    def prehook(self, name):
        def hook(module, inputs):
            x = inputs[0]
            events = getattr(x, "_conversion_events", None)
            if events is None and x.ndim == 2: events = self.pooled_events
            analog = events is None
            if analog:
                # Established DCGS convention: analog input fan-out is MAC,
                # excluded from SOP-only1.5pJ energy. Retain its nonzero-input proxy.
                events = (x != 0).float()
            kwargs = getattr(module, "fwd_kwargs", None)
            if isinstance(module, nn.Conv2d):
                kwargs = {k: getattr(module, k) for k in ("stride", "padding", "dilation", "groups")}
            if isinstance(module, nn.Linear): kwargs = None
            if self.parallel:
                if len(events) % self.time: raise ValueError("Parallel event axis mismatch")
                chunks = events.view((self.time, len(events) // self.time) + events.shape[1:])
                for i, item in enumerate(chunks):
                    self.rows[name]["mac" if analog else "sop"][i] += float(operations(item, module.weight, kwargs))
            else:
                self.rows[name]["mac" if analog else "sop"][self.step] += float(operations(events, module.weight, kwargs))
        return hook

    def close(self):
        for handle in self.handles: handle.remove()

    def result(self, samples):
        names = list(self.rows)
        layers = np.array([np.cumsum(self.rows[n]["sop"]) / samples for n in names])
        macs = np.array([np.cumsum(self.rows[n]["mac"]) / samples for n in names])
        return {"layer_names": names, "layer_sop": layers.tolist(), "layer_mac": macs.tolist(),
                "sop": layers.sum(0).tolist(), "mac": macs.sum(0).tolist(),
                "energy_uj": (layers.sum(0) * 1.5e-6).tolist(),
                "emitted_event_names":list(self.events),
                "emitted_events_per_sample":{n:(np.cumsum(v)/samples).tolist() for n,v in self.events.items()},
                "analog_input_policy": "DCGS-compatible nonzero input fan-out MAC proxy, excluded from SOP energy",
                "gap_policy": "Transport emitted events through GAP by channel sum; record classifier fan-out",
                "excluded_costs": "neuron arithmetic, parallel averaging/control and calibration; no MAC coefficient"}
