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
from matplotlib.animation import FuncAnimation

from snn_models import *

class RSNN_monitor(RSNN):
     
    def define_operations(self):
        self.has_propagated = False
        
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)
        
        self.i_drop = nn.Dropout(p=0.1)
    
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

        fig = plt.figure(figsize=(9,5))

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
        
        fig = plt.figure(figsize=(9,5))
        plt.plot(data_to_plot.T)
        
        plt.ylabel('Potential (mV)')
        plt.xlabel('Time (ms)')           
        
        return fig

    def animation(self, anim_frames=500, class_names = None):
        
        def square(num):
            '''
            get two closest factors of num so we can plot a vector of length num as an square-ish matrix
            '''
            factor1 = [x for x in range(1,num+1) if num%x==0 ]
            factor2 = [int(num/x) for x in factor1]
            idx = np.argmin(np.abs(np.array(factor2) - np.array(factor1)))
            return factor1[idx], factor2[idx]    

        xh, yh = square(self.num_hidden)

        hidden_spk = self.all_h_spike.permute(1,0,2).flatten(0,1).view(-1, xh, yh).detach().cpu().numpy()
        output_spk = self.all_o_spike.T.reshape(-1, anim_frames).detach().cpu().numpy()
        #output_spk = self.all_o_spike.view(-1,self.num_output).detach().cpu().numpy()
        
        if self.dataset == 'nmnist':
            num_channels = 2
        else:
            num_channels = 1
            
        ni = int(self.num_input / num_channels)
        xx, yx = square(ni)

        hidden_spk[0,0,0] = 1.0

        fig = plt.figure(figsize=(9,5))

        gs = fig.add_gridspec(2,3)

        if num_channels==1:
            ax1 = fig.add_subplot(gs[:, 0])
            x_spk = self.all_x.permute(1,0,2).flatten(0,1).view(-1, xx, yx).detach().cpu().numpy()
            x_spk[0,0,0] = 1.0
            im_x = ax1.imshow(x_spk[0,:,:])
        elif num_channels ==2:
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.axis('off')

            x_spk_1 = self.all_x.permute(1,0,2).flatten(0,1)[:,:ni].view(-1, xx, yx).detach().cpu().numpy()
            x_spk_2 = self.all_x.permute(1,0,2).flatten(0,1)[:,ni:].view(-1, xx, yx).detach().cpu().numpy()
            #x_spk_1 = np.swapaxes(x_spk_1, 1,2)
            #x_spk_2 = np.swapaxes(x_spk_2, 1,2)

            x_spk_1[0,0,0] = 1.0 
            x_spk_2[0,0,0] = 1.0
            im_x_1 = ax1.imshow(x_spk_1[0,:,:])
            im_x_2 = ax2.imshow(x_spk_2[0,:,:])
        
        ax1.axis('off')
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.axis('off')
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax5 = fig.add_subplot(gs[:, 2])
        ax5.axis('off')

        ax5.set_ylim(0,self.num_output)

        ax1.set_title('input spikes')
        ax3.set_title('hidden spikes')
        ax4.set_title('hidden spike count')
        ax4.set_xlabel('time')
        ax4.set_ylabel('number of spikes')
        ax5.set_title('output spikes')
        
        r = 0.5
        # class_names = ['em_stp', 'mv_ahd', 'mv_bk1', 'mv_bk2', 'sl_dwn', 'sa_eng', 'so_eng', 'st_ahd', 'tn_lft', 'tn_rht', 'none']
        
        if class_names == None:
            class_names = list(range(self.num_output))
        
        for pos, name in enumerate(class_names):
            ax5.annotate(str(name),(0.2, 0.4+ float(pos)))

        circles = [plt.Circle((0.8, r + float(pos)), 0.2, edgecolor = 'k', facecolor = 'white') for pos in range(len(class_names))]    

        for pos, circle in enumerate(circles):
            ax5.add_artist(circle)

        im_h = ax3.imshow(hidden_spk[0,:,:])
        ax4.set_xlim(0,anim_frames)
        ax4.set_ylim(0,int(self.num_hidden/3))
        spk, = ax4.plot([])
        avg_spk, = ax4.plot([])

        t = np.arange(anim_frames)
        spk_data = np.zeros(anim_frames)
        avg_spk_data = np.zeros(anim_frames)

        def animate(frame_num):

            spk_data[frame_num] = hidden_spk[frame_num,:,:].sum()
            avg_spk_data[frame_num] = spk_data[:frame_num].mean()

            if num_channels==1:
                im_x.set_data(x_spk[frame_num,:,:])
            else:
                im_x_1.set_data(x_spk_1[frame_num,:,:])
                im_x_2.set_data(x_spk_2[frame_num,:,:])
            im_h.set_data(hidden_spk[frame_num,:,:])
            spk.set_data((t, spk_data))
            avg_spk.set_data((t, avg_spk_data))
            #im_h1.set_data(hidden_spk[t,:,:])    
            #return (im_x, im_h)

            for pos, circle in enumerate(circles):
                if int(output_spk[pos, frame_num])==1:
                    circle.set_facecolor('blue')
                else:
                    circle.set_facecolor('white')
                    
        anim = FuncAnimation(fig, animate, frames=anim_frames, interval=10)
        
        return anim
        #plt.show()

        
        
        
        