#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:04 2022

@author: alberto
"""

import torch
import torch.nn as nn
from torch import Tensor
import math

class ActFun(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, input):
        thresh = 0.3
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        thresh = 0.3
        lens = 0.25
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)

class Hadamard(nn.Linear):     
    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.weight)

class Hadamard_sum(nn.Linear): 
    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(torch.mul(input, self.weight), dim=2)    
    
    
class RSNN(nn.Module):
    
    def __init__(self, d='mnist', num_hidden=256, thresh=0.3, decay=0.3, batch_size=256, win=50, device='cuda'):
        super(RSNN, self).__init__()
        
        self.act_fun = ActFun.apply
        
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        self.decay = decay
        
        self.d = d
        
        if self.d=='mnist':
            self.num_input = 34*34*2
            self.num_output = 10
        if self.d=='shd':
            self.num_input = 700
            self.num_output = 20            
       
        self.define_operations()
    
    def define_operations(self):
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)        
    
    def forward(self, input):
        
        h0_mem = h0_spike = h0_sumspike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        if self.d=='shd':
            dense_input = input.to_dense()
            dense_input = 1*(dense_input > 0).float()
        
        for step in range(self.win):
            
            x = dense_input[:, step, :]
            
            x = x.view(self.batch_size, -1)

            h0_mem, h0_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, x, h0_spike, h0_mem)
                        
            h2_mem, h2_spike = self.mem_update(self.fc_ho, h0_spike, h2_mem, h2_spike)

            h2_sumspike += h2_spike
        
        outputs = h2_sumspike / (self.win)
        
        return outputs    
    
    
    def mem_update(self, operation, x, mem, spike):
        mem = mem * self.decay * (1 - spike) + operation(x)
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
    def mem_update_rnn(self, operation1, operation2, x, h_spike, mem):
        a = operation1(x)
        b = operation2(h_spike)
        c = mem * self.decay * (1 - h_spike) 
        mem = a + b + c
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
class RSNN_delay(RSNN):
    
    def define_operations(self):
        self.fc_ih = Hadamard_sum(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = Hadamard_sum(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = Hadamard_sum(self.num_hidden, self.num_output, bias= False)  
    
        self.d_ih = Hadamard(self.num_input, self.num_hidden, bias= False)
        self.d_hh = Hadamard(self.num_hidden, self.num_hidden, bias= False)
        self.d_ho = Hadamard(self.num_hidden, self.num_output, bias= False)    
        
        # delay init
        self.d_ih.weight.data = torch.nn.Parameter(torch.ones(self.num_hidden, self.num_input, device=self.device, requires_grad=False)) 
        self.d_hh.weight.data = torch.nn.Parameter(torch.ones(self.num_hidden, self.num_hidden, device=self.device, requires_grad=False))
        self.d_ho.weight.data = torch.nn.Parameter(torch.ones(self.num_output, self.num_hidden, device=self.device, requires_grad=False))
    
    def forward(self, input):
        
        h_mem = h_spike = h_sumspike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        counter_ih = d_spikes_ih = torch.ones(self.batch_size, self.num_hidden, self.num_input, device=self.device)
        counter_hh = d_spikes_hh = torch.ones(self.batch_size, self.num_hidden, self.num_hidden, device=self.device)
        counter_ho = d_spikes_ho = torch.ones(self.batch_size, self.num_output, self.num_hidden, device=self.device)
        
        if self.d=='shd':
            dense_input = input.to_dense()
            dense_input = 1*(dense_input > 0).float()
        
        for step in range(self.win):
            
            x = dense_input[:, step, :]
            
            x = x.view(self.batch_size, -1)
            
            # compute the actual spikes gathered by the layers according to the delays
        
            d_x2h_spike, counter_ih = self.delay_spike(self.d_ih, x, counter_ih)
            
            d_h2h_spike, counter_hh = self.delay_spike(self.d_hh, h_spike, counter_hh)
            
            d_h2o_spike, counter_ho = self.delay_spike(self.d_ho, h_spike, counter_ho)
            
            # propagate the spikes
           
            h_mem, h_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, d_x2h_spike, d_h2h_spike, h_mem, h_spike)
                        
            o_mem, o_spike = self.mem_update(self.fc_ho, d_h2o_spike, o_mem, o_spike)

            o_sumspike += o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs       
    
    def delay_spike(self, operation, spike, sp_countdown):
        
        # input: spike vector
        # output: delayed spike representation (matrix)
        
        expanded_spikes = spike.expand(operation.weight.shape[0], self.batch_size, operation.weight.shape[1]).permute(1,0,2)
        delays = operation(expanded_spikes) # hadamard same shape of delay matrix
                                       
        counter = delays + (1-expanded_spikes)*(sp_countdown-1) # if there is no spike countdown, else set delay from matrix
        d_spike = 1*torch.logical_and(sp_countdown>-0.5, sp_countdown<0.5)
        
        return d_spike, counter

    def mem_update(self, operation, d_x, mem, current_spike):
        a = operation(d_x)
        b = mem * self.decay * (1 - current_spike)
        mem = a + b
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
    def mem_update_rnn(self, operation1, operation2, d_x, d_h_spike, mem, current_spike):
        a = operation1(d_x)
        b = operation2(d_h_spike)
        c = mem * self.decay * (1 - current_spike) 
        mem = a + b + c
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh) # reset membrane
        return mem, spike    
    