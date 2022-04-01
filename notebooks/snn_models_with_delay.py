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

from snn_models import *

class Hadamard(nn.Linear):     
    def forward(self, input: Tensor) -> Tensor:
        return torch.mul(input, self.weight)

class Hadamard_sum(nn.Linear): 
    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(torch.mul(input, self.weight), dim=2)   

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
        
        counter_ih = -5.0*torch.ones(self.batch_size, self.num_hidden, self.num_input, device=self.device)        
        counter_hh = -5.0*torch.ones(self.batch_size, self.num_hidden, self.num_hidden, device=self.device)        
        counter_ho = -5.0*torch.ones(self.batch_size, self.num_output, self.num_hidden, device=self.device)

        # for debug only
        self.all_delayed_spikes = torch.ones(self.win, self.batch_size, self.num_hidden, self.num_input, device=self.device)
        self.all_counters = torch.ones(self.win, self.batch_size, self.num_hidden, self.num_input, device=self.device) 
        
        if self.d=='shd':
            dense_input = input.to_dense()
            dense_input = 1*(dense_input > 0).float()
        else:
            dense_input = input
            
        for step in range(self.win):
            
            x = dense_input[:, step, :]
            
            x = x.view(self.batch_size, -1)
            
            # compute the actual spikes gathered by the layers according to the delays
        
            d_x2h_spike, counter_ih = self.delay_spike(self.d_ih, x, counter_ih)
            self.all_delayed_spikes[step,:,:,:] = d_x2h_spike
            self.all_counters[step,:,:,:] = counter_ih
            
            d_h2h_spike, counter_hh = self.delay_spike(self.d_hh, h_spike, counter_hh)
            
            d_h2o_spike, counter_ho = self.delay_spike(self.d_ho, h_spike, counter_ho)
            
            # propagate the spikes
           
            h_mem, h_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, d_x2h_spike, d_h2h_spike, h_mem, h_spike)
                        
            o_mem, o_spike = self.mem_update(self.fc_ho, d_h2o_spike, o_mem, o_spike)

            o_sumspike += o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs       
    
    def delay_spike(self, operation, spike, counter):
        
        # input: spike vector
        # output: delayed spike representation (matrix)
        
        expanded_spikes = spike.expand(operation.weight.shape[0], self.batch_size, operation.weight.shape[1]).permute(1,0,2)
        delays = operation(expanded_spikes) # hadamard same shape of delay matrix
                
        d_spike = 1*torch.logical_and(counter>-0.5, counter<0.5) 

        counter = (counter>0)*(counter-1) + (counter<=0)*(delays + (1-expanded_spikes)*(counter-1))
        
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
                                
                                
class RSNN_delay_global_count(RSNN):
    
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
        
        counter_ih = -5.0*torch.ones(self.batch_size, self.num_hidden, self.num_input, device=self.device)
        sp_acc_ih = d_sp_acc_ih = torch.ones(self.batch_size, self.num_hidden, self.num_input, device=self.device)
        
        counter_hh = -5.0*torch.ones(self.batch_size, self.num_hidden, self.num_hidden, device=self.device)
        sp_acc_hh = d_sp_acc_hh = torch.ones(self.batch_size, self.num_hidden, self.num_hidden, device=self.device)
        
        counter_ho = -5.0*torch.ones(self.batch_size, self.num_output, self.num_hidden, device=self.device)
        sp_acc_ho = d_sp_acc_ho = torch.ones(self.batch_size, self.num_output, self.num_hidden, device=self.device)

        # for debug only
        self.all_delayed_spikes = torch.ones(self.win, self.batch_size, self.num_hidden, self.num_input, device=self.device)
        self.all_counters = torch.ones(self.win, self.batch_size, self.num_hidden, self.num_input, device=self.device) 
        
        if self.d=='shd':
            dense_input = input.to_dense()
            dense_input = 1*(dense_input > 0).float()
        else:
            dense_input = input
            
        for step in range(self.win):
            
            x = dense_input[:, step, :]
            
            x = x.view(self.batch_size, -1)
            
            # compute the actual spikes gathered by the layers according to the delays
        
            d_x2h_spike, counter_ih, sp_acc_ih, d_sp_acc_ih = self.delay_spike(self.d_ih, x, counter_ih, sp_acc_ih, d_sp_acc_ih)
            self.all_delayed_spikes[step,:,:,:] = d_x2h_spike
            self.all_counters[step,:,:,:] = counter_ih
            
            d_h2h_spike, counter_hh, sp_acc_hh, d_sp_acc_hh = self.delay_spike(self.d_hh, h_spike, counter_hh, sp_acc_hh, d_sp_acc_hh)
            
            d_h2o_spike, counter_ho, sp_acc_ho, d_sp_acc_ho = self.delay_spike(self.d_ho, h_spike, counter_ho, sp_acc_ho, d_sp_acc_ho)
            
            # propagate the spikes
           
            h_mem, h_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, d_x2h_spike, d_h2h_spike, h_mem, h_spike)
                        
            o_mem, o_spike = self.mem_update(self.fc_ho, d_h2o_spike, o_mem, o_spike)

            o_sumspike += o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs       
    
    def delay_spike(self, operation, spike, counter, sp_acc, d_sp_acc):
        
        # input: spike vector
        # output: delayed spike representation (matrix)
        
        expanded_spikes = spike.expand(operation.weight.shape[0], self.batch_size, operation.weight.shape[1]).permute(1,0,2)
        delays = operation(expanded_spikes) # hadamard same shape of delay matrix
        
        #counter = delays + (1-expanded_spikes)*(sp_countdown-1) # if there is no spike, countdown, else set delay from matrix
        #print(1.0*(sp_countdown==delays))
        
        dropped_sp = 1.0*(sp_acc - d_sp_acc - delays > 0)         
        
        #print(dropped_sp)
        
        d_spike = 1*torch.logical_and(counter>-0.5, counter<0.5) + 1*torch.logical_and(counter<=-0.5, counter<=0.5)*dropped_sp
        #d_spike = 1*torch.logical_and(counter>-0.5, counter<0.5) 
        
        sp_acc = sp_acc + expanded_spikes
        d_sp_acc = d_sp_acc + d_spike
        
        counter = (counter>0)*(counter-1) + (counter<=0)*(delays + (1-expanded_spikes)*(counter-1))
        
        return d_spike, counter, sp_acc, d_sp_acc

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