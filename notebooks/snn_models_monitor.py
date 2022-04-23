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
import seaborn as sns

from snn_models import *

class RSNN_monitor(RSNN):
     
    def define_operations(self):
        self.has_propagated = False
        
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)
        
        self.i_drop = nn.Dropout(p=0.0)
    
    def forward(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        self.h_sumspike = torch.tensor(0.0)
        
        self.all_x = torch.zeros(self.win, self.batch_size, self.num_input, device=self.device)
        self.all_h_mem = torch.zeros(self.win, self.batch_size, self.num_hidden, device=self.device)
        self.all_h_spike = torch.zeros(self.win, self.batch_size, self.num_hidden, device=self.device)
        self.all_o_mem = torch.zeros(self.win, self.batch_size, self.num_output, device=self.device)
        self.all_o_spike = torch.zeros(self.win, self.batch_size, self.num_output, device=self.device)
        
        for step in range(self.win):
            
            x = input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)
            
            #i_spike = self.i_drop(i_spike)
            
            self.all_x[step] = i_spike

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)
            
            self.all_h_mem[step] = h_mem
            self.all_h_spike[step] = h_spike
            self.all_o_mem[step] = o_mem
            self.all_o_spike[step] = o_spike
            
            self.h_sumspike = self.h_sumspike + h_spike.sum()
            o_sumspike = o_spike + o_sumspike
            
        outputs = o_sumspike / (self.win)
        self.has_propagated = True
        
        return outputs

    def plot_activity(self, layer = 'x', plot='spike', plot_type = 'normal', sample_id='all', max_plots = 20):
        # idx is a list
        # plots the last batch propagated
        # plot type: normal or average

        if plot=='spike':
            cmap='Greys'
            vmin = 0
            vmax= 1
            if layer == 'x':
                data = self.all_x.data.cpu()
            elif layer == 'h':
                data = self.all_h_spike.data.cpu()
            elif layer == 'o':
                data=self.all_o_spike.data.cpu()
        elif plot =='mem':
            vmin = -self.thresh
            vmax = self.thresh
            cmap = 'RdBu'
            if layer == 'x':
                raise ValueError('input layer has no membrane potential!') 
            elif layer == 'h':
                data = self.all_h_mem.data.cpu()
            elif layer == 'o':
                data=self.all_o_mem.data.cpu()               

        if sample_id == 'all':
            sample_id = [x for x in range(self.batch_size)]

        if len(sample_id)>max_plots:
            print('clipping plots to maximum allowed: {}'.format(max_plots))
            sample_id = sample_id[:max_plots]   

        data_to_plot = data[:,sample_id,:].T.reshape((data.shape[2] , self.win*len(sample_id)))

        if plot_type=='avg':
            data_to_plot = data_to_plot.mean(axis=0)
            xlabel = 'Time'
            ylabel = 'Average spiking activity' 
        else:
            xlabel = 'Time'
            ylabel = 'Neuron'            
            if data_to_plot.shape[0]>data_to_plot.shape[1]:
                data_to_plot = data_to_plot.T
                xlabel = 'Neuron'
                ylabel = 'Time'                

        fig = plt.figure(figsize=(20,10))

        if plot_type=='normal':
            plt.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        if plot_type=='avg':
            plt.plot(data_to_plot)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)                

        return fig

    def plot_mem(self, layer = 'h', neuron_id = 'all', sample_id ='all', max_plots=20): 

        if layer == 'x':
            raise ValueError('input layer has no membrane potential!') 
        elif layer == 'h':
            data = self.all_h_mem.data.cpu()
        elif layer == 'o':
            data = self.all_o_mem.data.cpu()           

        if neuron_id == 'all':
            neuron_id = [x for x in range(data.shape[2])]
            
        if sample_id == 'all':
            sample_id = [x for x in range(self.batch_size)]

        if len(sample_id)>max_plots:
            print('clipping samples to maximum allowed: {}'.format(max_plots))
            sample_id = sample_id[:max_plots]              
        if len(neuron_id)>max_plots:
            print('clipping membrane plots to maximum allowed: {}'.format(max_plots))
            neuron_id = neuron_id[:max_plots]   
        print(neuron_id) 
        print(sample_id)
        
        print(data.shape)
        
        data_to_plot = data[:,sample_id][:,:, neuron_id].reshape((len(neuron_id) , self.win*len(sample_id)))      
        print(data_to_plot.shape)
        
        fig = plt.figure(figsize=(20,10))
        plt.plot(data_to_plot.T)
        
        plt.ylabel('Potential (mV)')
        plt.xlabel('Time (ms)')           
        
        return fig
    
    def animate_spikes(self):
        pass
        
        
        