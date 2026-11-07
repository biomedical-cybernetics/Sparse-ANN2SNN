import math
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
import table5_legacy.dst_scheduler as legacy
from cnn_dst_gd_protocol import active_count, conv_sparsity


def runtime_args(args):
    return SimpleNamespace(
        architecture="VGG-16", dataset=args.dataset, bs=args.batch_size, dim=1,
        conv_sparsity=0.0, linear_sparsity=0.99, sparsity=0.99, dropout=0.2,
        one_fc=False, pruning="", remove_method="weight_magnitude_soft",
        regrow_method="CH3_L3n_soft", granet=True, gmp=False,
        granet_init_sparsity=0.5, granet_init_step=0, pruning_T_end=None,
        pruning_scheduler="s_shape", pruning_method="ri", k=6.0,
        sparsity_distribution="uniform", static_topo=False, adaptive_zeta=True,
        chain_removal=True, early_stop=False, EM_S=False, itop=False,
        history_weights=False, new_history_weights=False, init_mode="xavier", WS=False,
        BRF=True, brf_r=0.1, degree_dist="uniform", ssam=False, factor=0.01,
        start_T=1.0, end_T=1.0, delta=0.5, delta_max=0.75,
        delta_d=0.01, delta_schedule="linear",
    )


def pick(scores, eligible, count, *, random=False, delta=None):
    indices = eligible.flatten().nonzero().flatten()
    if not 0 <= count <= len(indices):
        raise ValueError("Insufficient legal regrowth/removal candidates")
    if not count:
        return indices[:0]
    if random:
        return indices[torch.randperm(len(indices), device=indices.device)[:count]]
    values = scores.float().flatten()[indices]
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite topology score")
    if delta is not None and delta < .95:
        values = values.pow(delta / (1 - delta)).add(1e-6)
        return indices[torch.multinomial(values, count, replacement=False)]
    return indices[torch.topk(values, count).indices]


def ch3(mask):
    a = mask.float(); t = a.T
    dd, tt = a @ t, t @ a
    bd, bt = dd != 0, tt != 0
    x, y = (a.sum(1) - dd) * bd, (t.sum(1) - tt) * bt
    x[x == 0] = 1; y[y == 0] = 1
    return (bd / x) @ a + ((bt / y) @ t).T


class ConvMask:
    def __init__(self, module, name, job):
        self.module, self.name, self.job = module, name, job
        shape = module.weight.shape[:2] if job.granularity == "channel" else module.weight.shape
        self.mask = torch.zeros(shape, dtype=torch.bool, device=module.weight.device)
        n = active_count(self.mask.numel(), .5 if job.method == "ours" else .7)
        self.mask.flatten()[torch.randperm(self.mask.numel(), device=self.mask.device)[:n]] = True
        self.count, self.delta, self.capture, self.gradient = 0, .25, False, None
        module.forward = MethodType(lambda mod, x: self.forward(x), module)

    def expanded(self):
        return self.mask[:, :, None, None] if self.mask.ndim == 2 else self.mask

    def forward(self, x):
        w = self.module.weight * self.expanded()
        if self.capture and w.requires_grad:
            w.register_hook(self.cache)
        return self.module._conv_forward(x, w, self.module.bias)

    def cache(self, grad):
        self.gradient = grad.detach().float().clone()
        return grad

    def score(self, value):
        return value.float().mean((-1, -2)) if self.mask.ndim == 2 else value.float()

    @torch.no_grad()
    def evolve(self):
        with torch.autocast(device_type=self.module.weight.device.type, enabled=False):
            old = self.mask.clone()
            target = active_count(old.numel(), conv_sparsity(self.count, self.job.horizon) if self.job.method == "ours" else .7)
            grow = int(.05 * target)
            magnitude = self.module.weight.float().abs()
            if self.job.method.startswith(("mest", "rigl")):
                if self.gradient is None or not torch.isfinite(self.gradient).all():
                    raise ValueError("Missing finite pre-mask dense gradient")
            importance = magnitude + .01 * self.gradient.abs() if self.job.method.startswith("mest") else magnitude
            keep = pick(self.score(importance), old, target - grow)
            self.mask.zero_(); self.mask.flatten()[keep] = True
            if self.job.method == "ours":
                indices = pick(ch3(self.mask), ~self.mask, grow, delta=self.delta)
            elif self.job.method.startswith("rigl"):
                indices = pick(self.score(self.gradient.abs()), ~self.mask, grow)
            else:
                indices = pick(None, ~self.mask, grow, random=True)
            self.mask.flatten()[indices] = True
            self.count += 1
            used_delta = self.delta
            if self.job.method == "ours": self.delta = min(.75, self.delta + .01)
            if int(self.mask.sum()) != target: raise ValueError("Wrong Conv budget")
            return {"layer": self.name, "update": self.count, "active": target,
                    "density_pruned": int(old.sum())-target, "planned_regrow": grow,
                    "changed": int((old != self.mask).sum()), "delta_used": used_delta if self.job.method == "ours" else None}


class LinearPruner(legacy.DSTScheduler):
    def __init__(self, model, optimizer, job):
        self.method, self.capture = job.method, False
        args = runtime_args(SimpleNamespace(dataset=job.dataset, batch_size=job.batch))
        if job.method != "ours":
            args.granet = False; args.BRF = False
            args.remove_method = "MEST" if job.method.startswith("mest") else "weight_magnitude"
            args.regrow_method = "gradient" if job.method.startswith("rigl") else "random"
        def weights(model):
            pairs = [(n, m.weight) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "last_layer"]
            self.names = [n + ".weight" for n, _ in pairs]
            return [w for _, w in pairs], list(range(len(pairs)))
        with patch.object(legacy, "get_W", weights):
            super().__init__(model, optimizer, T_end=job.horizon, delta=1, alpha=.3,
                             sparsity_distribution="uniform", ignore_linear_layers=False, args=args)

    @torch.no_grad()
    def random_sparsify(self):
        if self.method == 'ours': return super().random_sparsify()
        self.backward_masks=[]; self.record_mask=[]
        for w, sparsity in zip(self.W,self.S):
            mask=torch.zeros_like(w,dtype=torch.bool)
            mask.flatten()[torch.randperm(w.numel(),device=w.device)[:active_count(w.numel(),sparsity,True)]]=True
            self.backward_masks.append(mask); w.mul_(mask)

    def should_accumulate_dense_grad(self):
        return self.capture

    def _scores_for_regrowth(self, layer_idx, current_mask, w):
        scores = super()._scores_for_regrowth(layer_idx, current_mask, w)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite Linear topology score")
        return scores

    def _current_prune_rate(self):
        return .99 if self.step >= self.T_end else super()._current_prune_rate()

    @torch.no_grad()
    def link_removal(self):
        for i,(w,mask) in enumerate(zip(self.W,self.backward_masks)):
            keep=int(mask.sum())-int(int(mask.sum())*self.cosine_annealing())
            scores=w.float().abs()
            if self.method.startswith('mest'):
                grad=self.backward_hook_objects[i].dense_grad
                if grad is None or not torch.isfinite(grad).all(): raise ValueError('Missing MEST gradient')
                scores=scores+.01*grad.abs()
            if self.method=='ours':
                eligible=mask.flatten().nonzero().flatten()
                weights=scores.flatten()[eligible].clamp_min(1e-12)
                indices=eligible[torch.multinomial(weights,keep,replacement=False)] if keep else eligible[:0]
            else:
                indices=pick(scores,mask,keep)
            new=torch.zeros_like(mask); new.flatten()[indices]=True; self.backward_masks[i]=new

    @torch.no_grad()
    def link_regrowth(self):
        if self.method == "ours": return super().link_regrowth()
        for i, (w, mask) in enumerate(zip(self.W, self.backward_masks)):
            count = active_count(w.numel(), self.S[i], True) - int(mask.sum())
            grad = self.backward_hook_objects[i].dense_grad
            if self.method.startswith(("rigl", "mest")) and (grad is None or not torch.isfinite(grad).all()):
                raise ValueError("Missing finite Linear dense gradient")
            indices = pick(grad.abs() if grad is not None else None, ~mask, count,
                           random=not self.method.startswith("rigl"))
            mask.flatten()[indices] = True


def ordinary_model(job):
    from Models.VGG import vgg16
    from Models.ResNet import resnet18
    channels, classes = (3, 100) if job.dataset == "CIFAR100" else (18, 11)
    return (vgg16(classes, job.dropout, False, in_channels=channels, hidden_dim=512)
            if job.architecture == "VGG-16" else resnet18(classes, in_channels=channels))


class Topology:
    def __init__(self, model, optimizer, job):
        self.job, self.model, self.conv, self.linear = job, model, [], None
        # The caller isolates all mask initialization RNG from model/data RNG.
        if job.method != "dense":
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and "shortcut" not in name:
                    self.conv.append(ConvMask(module, name + ".weight", job))
            if job.architecture == "VGG-16": self.linear = LinearPruner(model, optimizer, job)

    def capture(self, enabled):
        for layer in self.conv:
            layer.capture = enabled; layer.gradient = None
        if self.linear:
            self.linear.capture = enabled
            for hook in self.linear.backward_hook_objects: hook.dense_grad = None

    def masks(self):
        masks = {l.name: l.mask for l in self.conv}
        if self.linear: masks.update(zip(self.linear.names, self.linear.backward_masks))
        return masks

    @torch.no_grad()
    def evolve(self, epoch):
        with torch.autocast(device_type=next(self.model.parameters()).device.type, enabled=False):
            events = [layer.evolve() for layer in self.conv]
            if self.linear and epoch <= self.job.horizon:
                if self.job.method.startswith(("mest", "rigl")):
                    for h in self.linear.backward_hook_objects:
                        if h.dense_grad is None or not torch.isfinite(h.dense_grad).all(): raise ValueError("Missing gradient")
                self.linear.step = epoch
                self.linear._dst_step(); self.linear.dst_steps += 1
            return events

    def state_dict(self):
        return {"conv": [{"mask": l.mask, "count": l.count, "delta": l.delta} for l in self.conv],
                "linear": self.linear.state_dict() if self.linear else None}

    def load_state_dict(self, state):
        if len(state["conv"]) != len(self.conv): raise ValueError("Wrong topology identity")
        for l, s in zip(self.conv, state["conv"]):
            l.mask.copy_(s["mask"]); l.count, l.delta = s["count"], s["delta"]
        if self.linear: self.linear.load_state_dict(state["linear"])

    def inventory(self):
        masks = self.masks(); rows = []
        for name, m in self.model.named_modules():
            if not isinstance(m, (nn.Conv2d, nn.Linear)): continue
            mask = masks.get(name + ".weight")
            factor = m.weight.numel() // mask.numel() if mask is not None else 1
            zeros = (mask.numel() - int(mask.sum())) * factor if mask is not None else 0
            target = active_count(mask.numel(), .99 if isinstance(m, nn.Linear) else .7, isinstance(m, nn.Linear)) if mask is not None else m.weight.numel()
            rows.append({"layer": name, "shape": list(m.weight.shape), "weights": m.weight.numel(),
                         "zeros": zeros, "sparsity": zeros/m.weight.numel(),
                         "numerical_zero_weights": int((m.weight.detach()==0).sum()),
                         "granularity": "dense" if mask is None else "channel-pair" if factor > 1 else "element",
                         "qualified": mask is None or int(mask.sum()) == target})
        return rows

    def materialize(self):
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for name, mask in self.masks().items():
            if mask.ndim == 2 and state[name].ndim == 4: mask = mask[:, :, None, None]
            state[name].mul_(mask)
        return state
