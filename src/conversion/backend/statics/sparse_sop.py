import torch
import torch.nn as nn
from .sop import BaseMonitor, SOPMonitor, unpack_for_conv, unpack_for_linear
from pathlib import Path
import sys

from DCGS.utils import ExpandTemporalDim
from DCGS.neurons import AtNeuron

class SparseSOPMonitor(SOPMonitor):
    '''
    SOPMonitor supporting sparse weights and LASFR calculation.
    New params: check_relu_after_maxpool:bool, True means to do best-effort check.
    Note: Sparse weight support class: nn.Conv2d, CHT:Conv2d_CHT, nn.Linear
    Note: To take the input of Linear/Conv modules as the spike emitted by neuron, you should ensure maxpool is before relu(use Converter.change_maxpool_before_relu(model)).
    '''
    def __init__(self, net: nn.Module,step_mode='s',T=8,coding_type='diff_rate',check_relu_after_maxpool:bool = True):
        #super(SparseSOPMonitor, self).__init__(net, step_mode, T, coding_type)
        BaseMonitor.__init__(self)
        if check_relu_after_maxpool:
            self.MAXPOOL_CLASSES = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
        self.step_mode=step_mode
        self.T=T
        self.coding_type = coding_type
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)

        # re init because create_hook is overriden. Use super().__init__ will create hook with old hook
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook_conv(name,step_mode)))
            elif isinstance(m, nn.Linear):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook_linear(name,step_mode)))
            elif isinstance(m, AtNeuron):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook_at(name,step_mode)))

            if 'maxpool' in name:
                if check_relu_after_maxpool:
                    mod = m
                    if not isinstance(mod, nn.Sequential):
                        print(f'[WARNING] A module with maxpool in name is not an instance of nn.Sequential. Please make sure relu is after maxpool.')
                    else:
                        if not len(mod)==2:
                            print(f'[WARNING] A module with maxpool in name is an nn.Sequential but length is not 2 but {len(mod)}')
                        else:
                            assert isinstance(mod[0], self.MAXPOOL_CLASSES), f'maxpool should be placed before relu. Now module is {mod}'


    def cal_sop_conv_s(self, x: torch.Tensor, m: nn.Conv2d):
        x_spike = torch.zeros_like(x)
        x_spike[x!=0]=1
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                weight_mask = (m.weight!=0).float()
            else:
                raise TypeError(f'conv hook only support nn.Conv2d, but the module class is {type(m)}')
            dilation = getattr(m, 'dilation', 1)
            groups = getattr(m, 'groups', 1)
            sop = (torch.nn.functional.conv2d(x_spike, weight_mask, None, m.stride,m.padding, dilation, groups)).sum().item()
            tot = -1
            in_spikes = x_spike.sum().item()
            features = x_spike.numel()/x_spike.size(0)
        return sop, tot, in_spikes/features

    def cal_sop_linear_s(self, x: torch.Tensor, m: nn.Linear):
        x_spike = torch.zeros_like(x)
        x_spike[x!=0]=1
        with torch.no_grad():
            weight_mask = (m.weight!=0).float()
            sop = (torch.nn.functional.linear(x_spike, weight_mask, None)).sum().item()
            tot = -1
            in_spikes = x_spike.sum().item()
            features = x_spike.numel()/x_spike.size(0)
        return sop, tot, in_spikes/features

    def create_hook_linear(self, name, step_mode):
        if step_mode=='s':
            def hook_s(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_linear_s(unpack_for_linear(x).detach(), m))
            return hook_s
        if step_mode=='m':
            def hook_m(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
                x_t = self.expand(unpack_for_linear(x))
                for i in range(len(x_t)):
                    self.name_records_index[name].append(self.records.__len__())
                    self.records.append(self.cal_sop_linear_s(x_t[i].detach(), m))
            return hook_m

    def create_hook_conv(self, name, step_mode):
        if step_mode=='s':
            def hook_s(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_conv_s(unpack_for_conv(x).detach(), m))
            return hook_s
        if step_mode=='m':
            def hook_m(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
                x_t = self.expand(unpack_for_conv(x))
                for i in range(len(x_t)):
                    self.name_records_index[name].append(self.records.__len__())
                    self.records.append(self.cal_sop_conv_s(x_t[i].detach(), m))
            return hook_m
