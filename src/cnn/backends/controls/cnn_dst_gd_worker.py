import argparse, copy, json, os, random, platform, time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import Models
from runtime import atomic_json, configuration_data, capture_rng_state, restore_rng_state, save_checkpoint_atomic
from cnn_dst_gd_protocol import Job
from cnn_dst_gd_model import ordinary_model, Topology

def configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["SNN_DVS_NUM_WORKERS"] = "0"
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

def training_context(job):
    precision = getattr(job, "training_precision", "bf16")
    if precision == "fp32": return nullcontext()
    if precision == "bf16": return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError("Training precision must be fp32 or bf16")

def optimizer_scheduler(model, job):
    if job.dataset == "CIFAR100":
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4, nesterov=True)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=.1, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=job.epochs-5, eta_min=0.)
        schedule = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[5])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=5e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=0.)
    return opt, schedule

def loaders(job, root, smoke=False):
    args = SimpleNamespace(architecture=job.architecture, dataset=job.dataset, bs=2 if smoke else job.batch,
        conv_sparsity=0., linear_sparsity=0., pruning="", one_fc=False,
        dropout=job.dropout or 0., dim=2, distributed=False)
    _, _, train, test = Models.prepare_model_and_loader(args, str(root))
    def make(loader, shuffle):
        return torch.utils.data.DataLoader(loader.dataset, batch_size=args.bs, shuffle=shuffle,
            num_workers=0, pin_memory=True, drop_last=False)
    return make(train, True), make(test, False)

@torch.no_grad()
def evaluate(model, loader, limit=None):
    model.eval(); correct = total = 0; loss_sum = 0.
    with torch.autocast("cuda", enabled=False):
        for index, (x, y) in enumerate(loader):
            if limit is not None and index >= limit: break
            x, y = x.cuda(), y.cuda()
            if y.ndim > 1: y = y.argmax(1)
            out = model(x.float()); loss = nn.functional.cross_entropy(out, y)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite evaluation")
            correct += int((out.argmax(1) == y).sum()); total += len(y); loss_sum += float(loss)*len(y)
    if not total: raise ValueError("Empty evaluation")
    return correct/total, loss_sum/total

def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)

def save_export(prefix, topology, directory):
    for suffix, value in (("model", topology.materialize()), ("masks", topology.masks()), ("pruner", topology.state_dict())):
        save_checkpoint_atomic(directory/f"{prefix}_{suffix}.pth", cpu_copy(value))

@torch.no_grad()
def export_audit(job, topology, x):
    x=x.to(next(topology.model.parameters()).device)
    rng = capture_rng_state()
    try:
        ordinary = ordinary_model(job).cuda().eval()
        ordinary.load_state_dict(topology.materialize(), strict=True)
        topology.model.eval()
        difference = float((ordinary(x.float()) - topology.model(x.float())).abs().max())
        if difference != 0.: raise ValueError(f"Export logits mismatch: {difference}")
        rows = topology.inventory(); macs = 0; handles = []
        by_name = {r['layer']: r for r in rows}
        for name, module in ordinary.named_modules():
            if name not in by_name: continue
            def hook(m, inputs, output, row=by_name[name]):
                nonlocal macs
                macs += (row['weights']-row['zeros']) * (output.shape[-1]*output.shape[-2] if output.ndim==4 else 1)
            handles.append(module.register_forward_hook(hook))
        ordinary(x[:1].float())
        for h in handles: h.remove()
        return {"logit_max_abs_diff": difference, "layers": rows, "mask_based_macs": macs,
                "qualified": all(r['qualified'] for r in rows),
                "stored_parameters": sum(p.numel() for p in ordinary.parameters())}
    finally: restore_rng_state(rng)

def run(args):
    cfg = json.loads(args.config.read_text()); job = Job(**cfg['job']); configuration = configuration_data(job.config())
    if cfg['configuration'] != configuration: raise ValueError("Configuration identity mismatch")
    configure(job.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if getattr(job, "training_precision", "bf16") == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 CUDA required")
    train, test = loaders(job, args.data_root, args.smoke and not getattr(args, 'smoke_production_batch', False))
    configure(job.seed)
    model = ordinary_model(job).cuda(); optimizer, scheduler = optimizer_scheduler(model, job)
    rng = capture_rng_state(); torch.manual_seed(job.seed+100000); np.random.seed((job.seed+100000) % 2**32); random.seed(job.seed+100000)
    topology = Topology(model, optimizer, job)
    restore_rng_state(rng)
    artifacts, raw = args.artifact_dir, args.metadata_dir
    artifacts.mkdir(parents=True, exist_ok=True); raw.mkdir(parents=True, exist_ok=True)
    history = []; best = None; unrestricted = None; start = optimizer_steps = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state.get('configuration') != configuration or state['job'] != job.__dict__ or state.get('probe'):
            raise ValueError("Incompatible resume")
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); topology.load_state_dict(state['pruner'])
        history, best, unrestricted = state['history'], state['best'], state['unrestricted']
        start, optimizer_steps = state['epoch'], state['optimizer_steps']
        for name, value in state['best_artifacts'].items(): save_checkpoint_atomic(artifacts/name, value)
        if state.get('best_export_audit'): atomic_json(raw/'best_export_audit.json',state['best_export_audit'])
        restore_rng_state(state['rng'])
    begun = time.monotonic()
    for epoch in range(start+1, (start+1 if args.smoke else job.epochs)+1):
        model.train(); lr = optimizer.param_groups[0]['lr']; losses = 0.; samples = 0
        batch_count = min(2, len(train)) if args.smoke else len(train)
        for index, (x,y) in enumerate(train):
            if index >= batch_count: break
            topology.capture(index == batch_count-1)
            x,y=x.cuda(),y.cuda()
            if y.ndim > 1: y=y.argmax(1)
            optimizer.zero_grad(set_to_none=True)
            with training_context(job):
                out=model(x); loss=nn.functional.cross_entropy(out,y)
                if getattr(job,'training_precision','bf16')=='fp32' and (torch.is_autocast_enabled() or out.dtype!=torch.float32 or loss.dtype!=torch.float32):
                    raise ValueError('Pure FP32 training boundary violated')
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); optimizer.step(); optimizer_steps += 1
            losses += float(loss.detach())*len(y); samples += len(y)
        scheduler.step()
        acc, val_loss = evaluate(model, test, 1 if args.smoke else None)
        rows = topology.inventory(); qualified=all(r['qualified'] for r in rows)
        row={'epoch':epoch,'lr':lr,'train_loss':losses/samples,'test_top1':acc,
             'test_loss':val_loss,'budget_qualified':qualified,'layers':rows}
        record={'top1':acc,'loss':val_loss,'epoch':epoch}
        if unrestricted is None or acc>unrestricted['top1']:
            unrestricted=record; save_export('unrestricted_best',topology,artifacts)
        if qualified and (best is None or acc>best['top1']):
            best=record; save_export('best',topology,artifacts)
            atomic_json(raw/'best_export_audit.json', export_audit(job,topology,x[:1]))
        row['conv_updates']=topology.evolve(epoch)
        history.append(row); print(json.dumps(row),flush=True)
        if epoch%20==0 or epoch==job.epochs or args.smoke:
            saved={p.name:torch.load(p,map_location='cpu',weights_only=False) for p in artifacts.glob('*best_*.pth')}
            state={'job':job.__dict__,'configuration':configuration,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'scheduler':scheduler.state_dict(),'pruner':topology.state_dict(),'rng':capture_rng_state(),
                   'history':history,'best':best,'unrestricted':unrestricted,'epoch':epoch,'best_artifacts':saved,
                   'updates':epoch,'optimizer_steps':optimizer_steps,'probe':args.smoke}
            state['best_export_audit']=json.loads((raw/'best_export_audit.json').read_text()) if (raw/'best_export_audit.json').exists() else None
            folder=artifacts/'checkpoints'; folder.mkdir(exist_ok=True)
            save_checkpoint_atomic(folder/f'epoch-{epoch:04d}.pth',state)
            save_checkpoint_atomic(folder/'last_checkpoint.pth',state)
    save_export('final',topology,artifacts)
    if not args.smoke and best is None: raise ValueError('No budget-qualified best')
    tensors=list(model.parameters())+[v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]
    if any(not torch.isfinite(t).all() for t in tensors): raise FloatingPointError('Nonfinite model/optimizer state')
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in tensors): raise ValueError('Non-FP32 parameter/state')
    result={'job':job.__dict__,'configuration':configuration,'probe':args.smoke,'best':best,'unrestricted':unrestricted,
            'history':history,'optimizer_steps':optimizer_steps,'scaler_enabled':False,'state_dtype':'float32',
            'elapsed_seconds':time.monotonic()-begun,'conv_updates':[l.count for l in topology.conv],
            'linear_updates':topology.linear.dst_steps if topology.linear else 0,
            'export_audit':export_audit(job,topology,x[:1]),'all_parameters_finite':True}
    result['environment']={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                           'cudnn':torch.backends.cudnn.version(),'numpy':np.__version__,
                           'gpu':torch.cuda.get_device_name(0),'cpu_threads':torch.get_num_threads(),'dataloader_workers':0}
    result['training_precision'] = getattr(job, 'training_precision', 'bf16')
    if os.environ.get('ANN2SNN_DVS_CACHE'):
        result['data_acceleration']={'cache':os.environ['ANN2SNN_DVS_CACHE'],
            'legacy_voxel_fallbacks':getattr(getattr(train.dataset,'quantization_layer',None),'fallbacks',0),
            'cpu_threads':1,'workers':0}
    atomic_json(raw/'metrics.json',result)
    atomic_json(raw/('smoke-complete.json' if args.smoke else '.complete.json'),{'configuration':configuration,'probe':args.smoke})
