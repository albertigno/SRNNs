#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 5 15:28:04 2022

@author: alberto
"""

import torch
import torch.nn as nn
from torch import Tensor
import math
import matplotlib.pyplot as plt

from snn_models import *

class RSNN_monitor(RSNN):
     
    def define_operations(self):
        self.has_propagated = False
        
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)        
    
    def forward(self, input):
        
        h_mem = h_spike = h_sumspike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        self.all_h_mem = self.all_h_spike = torch.zeros(self.win, self.batch_size, self.num_hidden, device=self.device)
        self.all_o_mem = self.all_o_spike = torch.zeros(self.win, self.batch_size, self.num_output, device=self.device)
            
        for step in range(self.win):
            
            x = input[:, step, :]
            
            x = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, x, h_spike, h_mem)       
            o_mem, o_spike = self.mem_update(self.fc_ho, h0_spike, o_mem, o_spike)
            
            self.all_h_mem[step] = h_mem
            self.all_h_spike[step] = h_spike
            self.all_o_mem[step] = o_mem
            self.all_h_spike[step] = h_spike
            
            o_sumspike += o_spike
        
        outputs = o_sumspike / (self.win)
        self.has_propagated = True
        
        return outputs

    def plot_mem(self):
        pass    

    def plot_spikes(self):
        pass     
    
    def plot_input(self):
        pass
    
    def plot_hidden(self):
        pass
    
    def plot_output(self):
        pass    