#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:04 2022

@author: alberto
"""

import torch
import torch.nn as nn

class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


class RSNN(nn.Module):

    def __init__(self, d='mnist', num_hidden=256, thresh, decay, batch_size, win, device):
        super(RSNN, self).__init__()
        
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        self.decay = decay
        
        if d=='mnist':
            self.num_input = 34*34*2
            self.num_output = 10
        if d=='shd':
            self.num_input = 700
            self.num_output = 20            
        
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)
        
        
        
    def forward(self, input):
        
        h0_mem = h0_spike = h0_sumspike = torch.zeros(1, self.batch_size, self.num_hidden, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(1, self.batch_size, self.num_output, device=self.device)
        
        if d=='shd':
            dense_input = input.to_dense()
            dense_input = 1*(dense_input > 0).float()
        
        for step in range(win):
            
            x = dense_input[:, step, :]
            
            x = x.view(1, batch_size, -1)

            h0_mem, h0_spike = mem_update_rnn(self.fc_ih, self.fc_hh, x, h0_spike, h0_mem, h0_spike)
                        
            h2_mem, h2_spike = mem_update(self.fc_ho, h0_spike, h2_mem, h2_spike)

            h2_sumspike += h2_spike
        
        outputs = h2_sumspike / (win)
        
        return outputs[0]    
    
    
    def mem_update(self, operation, x, mem, spike):
        mem = mem * self.decay * (1 - spike) + operation(x)
        spike = act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
    def mem_update_rnn(self, operation1, operation2, x, h, mem, spike):
        a = operation1(x)
        b = operation2(h)
        c = mem * self.decay * (1 - spike) 
        mem = a + b + c
        spike = act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike