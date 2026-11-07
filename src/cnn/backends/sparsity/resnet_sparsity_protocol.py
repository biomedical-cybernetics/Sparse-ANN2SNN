from dataclasses import dataclass, asdict
from pathlib import Path
from resnet_matched_protocol import PROTOCOL

ROOT=Path(__file__).resolve().parent

EXPERIMENT_ID='resnet18-final-sparsity-ablation'

@dataclass(frozen=True)
class Job:
    dataset:str
    sparsity:float
    seed:int
    training_precision: str = "bf16"
    converter:str='ann'
    def __post_init__(self):
        if self.training_precision not in ("fp32", "bf16"):
            raise ValueError("Training precision must be fp32 or bf16")
        if self.dataset not in ('CIFAR10','CIFAR100') or self.sparsity not in (.6,.8) or (type(self.seed) is not int or not 0 <= self.seed < 2**32) or self.converter not in ('ann','dcgs-test','parallel'):raise ValueError('Invalid experiment configuration')
    architecture='ResNet-18'
    method='ours'
    dropout=None
    epochs=200
    horizon=150
    batch=128
    granularity='channel'
    @property
    def classes(self):return 10 if self.dataset=='CIFAR10' else 100
    @property
    def time(self):return 16 if self.converter=='parallel' else 8
    @property
    def is_dcgs(self):return self.converter=='dcgs-test'
    @property
    def calibration_batch(self):return 64 if self.converter=='parallel' else 128
    @property
    def ann_id(self):return f'resnet18-{self.dataset.lower()}-ours-s{round(self.sparsity*100)}-seed-{self.seed}'
    @property
    def job_id(self):return self.ann_id if self.converter=='ann' else f'{self.ann_id}-{self.converter}-time-{self.time}'
    def config(self):
        common={'experiment':EXPERIMENT_ID,'job':asdict(self),'protocol_reference':PROTOCOL,'conv_sparsity':self.sparsity}
        if self.converter!='ann':
            raise ValueError("Use the conversion recipe for inference")
            cfg=ConversionJob('ResNet-18',self.dataset,'ours',self.seed,self.converter,self.time).config()
            cfg.update(common);cfg['source']={'experiment':EXPERIMENT_ID,'ann_id':self.ann_id}
            cfg['primary_endpoint']='native8' if self.is_dcgs else 'native16'
            cfg['bn_fusion_validation']={'primary':'allclose rtol=1e-4,atol=1e-4','approved_fallback':'FP64 formula allclose1e-9;FP32 bound2e-5*max(1,maxabs(reference));equal predictions','operational_precision':'FP32'}
            return cfg
        return dict(common,epochs=200,lr=.1,batch=128,optimizer='SGD',momentum=.9,nesterov=True,weight_decay=5e-4,warmup=5,start_factor=.1,eta_min=0.,precision=('bf16-forward-fp32-state-eval-topology' if self.training_precision == "bf16" else "fp32"),tf32=False,scaler=False,initial_sparsity=.5,density_window=150,conv_updates=200,rewire=.05,delta=[.25,.75,.01],linear_dense=True,cpu_threads=1,workers=0)
