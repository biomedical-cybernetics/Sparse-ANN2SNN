import torch
import torch.nn as nn
from typing import Tuple, Union
from pathlib import Path
import sys

from DCGS.utils import ExpandTemporalDim
from DCGS.neurons import AtNeuron
from .base_monitor import BaseMonitor

class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module,step_mode='s',T=8,coding_type='diff_rate'):
        super().__init__()
        self.step_mode=step_mode
        self.T=T
        self.coding_type = coding_type
        if 'diff' in coding_type:
            self.expand = ExpandTemporalDim(T+1)
        else:
            self.expand = ExpandTemporalDim(T)

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

    def cal_sop_conv_s(self, x: torch.Tensor, m: nn.Conv2d):
        x_spike = torch.zeros_like(x)
        x_spike[x!=0]=1
        with torch.no_grad():
            sop = (torch.nn.functional.conv2d(x_spike, torch.ones_like(m.weight), None, m.stride,m.padding, m.dilation, m.groups)).sum()
            tot = (torch.nn.functional.conv2d(torch.ones_like(x_spike), torch.ones_like(m.weight), None, m.stride,m.padding, m.dilation, m.groups)).sum()
        return sop, tot

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


    def cal_sop_linear_s(self, x: torch.Tensor, m: nn.Linear):
        x_spike = torch.zeros_like(x)
        x_spike[x!=0]=1
        with torch.no_grad():
            sop = (torch.nn.functional.linear(x_spike, torch.ones_like(m.weight), None)).sum()
            tot = (torch.nn.functional.linear(torch.ones_like(x_spike), torch.ones_like(m.weight), None)).sum()
        return sop, tot

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

    def cal_sop_at_s(self, A: torch.Tensor,B: torch.Tensor, T , A_s: torch.Tensor,B_s: torch.Tensor):
        tmp_A = torch.ones_like(A)
        tmp_B = torch.ones_like(B)
        sum0 = (tmp_A@tmp_B).sum()
        tmp_A = torch.zeros_like(A)
        tmp_B = torch.zeros_like(B)
        tmp_A[A!=0]=1
        tmp_B[B!=0]=1
        tmp_As = torch.zeros_like(A_s)
        tmp_As[A_s!=0]=1
        tmp_Bs = torch.zeros_like(B_s)
        tmp_Bs[B_s!=0]=1
        out01 = (tmp_A@tmp_B).sum()
        out02 = (tmp_A@tmp_Bs).sum()
        out03 = (tmp_As@tmp_B).sum()
        out0 = out01+out02+out03
        return (out0,sum0),A_s+A/(T+1),B_s+B/(T+1)

    def create_hook_at(self, name, step_mode):
        if step_mode=='s':
            def hook_s(m: AtNeuron, x, y):
                if m.T==0 and 'diff' in m.coding_type:
                    self.name_records_index[name].append(self.records.__len__())
                    self.records.append((1,1))
                else:
                    self.name_records_index[name].append(self.records.__len__())
                    output, _ , _ = self.cal_sop_at_s(x[0].detach(),x[1].detach(), m.T-1, m.tot_a-x[0]/m.T, m.tot_b-x[1]/m.T)
                    self.records.append(output)
            return hook_s
        if step_mode=='m':
            def hook_m(m: AtNeuron, x, y):
                x_1 = self.expand(x[0])
                x_2 = self.expand(x[1])
                A_s = torch.zeros_like(x_1[0])
                B_s = torch.zeros_like(x_2[0])
                for i in range(len(x_1)):
                    if i==0 and 'diff' in m.coding_type:
                        self.name_records_index[name].append(self.records.__len__())
                        self.records.append((1,1))
                        continue
                    self.name_records_index[name].append(self.records.__len__())
                    output,A_s,B_s = self.cal_sop_at_s(x_1[i].detach(),x_2[i].detach(), i, A_s, B_s)
                    self.records.append(output)
            return hook_m

def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x


def unpack_for_linear(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    return x
