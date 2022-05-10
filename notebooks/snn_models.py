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
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
class RSNN(nn.Module):
    
    def __init__(self, dataset='nmnist', num_hidden=256, thresh=0.3, tau_m=0.83, vreset = 0.0, batch_size=256, win=50, device='cuda'):
        super(RSNN, self).__init__()
        
        self.act_fun = ActFun.apply
        
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        self.vreset = vreset
        self.dataset = dataset
        
        self.epoch = 0
        
        if self.dataset=='nmnist':
            self.num_input = 34*34*2
            self.num_output = 10
        if self.dataset=='shd':
            self.num_input = 700
            self.num_output = 20
        if self.dataset.split('_')[0]=='custom':
            self.num_input = int(self.dataset.split('_')[1])
            self.num_output = int(self.dataset.split('_')[2])        

        if tau_m!='adp':
            self.tau_m_h = nn.Parameter(torch.Tensor([tau_m]), requires_grad=False)
            self.tau_m_o = torch.Tensor([tau_m])
        else:
            self.tau_m_h = nn.Parameter(torch.Tensor(self.num_hidden))
            nn.init.normal_(self.tau_m_h, 0.83, 0.1)
            self.tau_m_o = nn.Parameter(torch.Tensor(self.num_output))
            nn.init.normal_(self.tau_m_o, 0.83, 0.1)

        self.acc = list() # stores accuracy every time test() is called
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization
        self.train_loss = list()
        self.test_loss = list()
        self.test_spk_count = list()
        
        self.define_operations()
    
    def define_operations(self):
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)        
    
    def forward(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
            
        for step in range(self.win):

            x = input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)
            
            self.h_sumspike = self.h_sumspike + h_spike.sum()
            o_sumspike = o_sumspike + o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs        
    
    def mem_update(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike) - o_spike*self.vreset
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike
    
    def mem_update_rnn(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_h).to(self.device)
        a = self.fc_ih(i_spike) # process spikes from input
        b = self.fc_hh(o_spike) # process recurrent spikes
        c = mem * alpha * (1-o_spike) - o_spike*self.vreset
        mem = a + b + c
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike
        
    def train_step(self, train_loader=None, optimizer=None, criterion=None, num_samples=0, spkreg=0.0):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size 
        sr = spkreg/self.win
        
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, labels) + sr*spk_count
            running_loss += loss.item()
            total_loss_train += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total]) 

    def test(self, test_loader = None, criterion=None):
        
        self.save_model()
        
        correct = 0
        total = 0
        total_loss_test = 0
        total_spk_count = 0
        
        snn_cpu = RSNN() # copy of self, doing this to always evaluate on cpu
        snn_cpu.load_model('rsnn', batch_size= self.batch_size)
        
        for images, labels in test_loader:
            images = images.float()
            labels = labels.float()
            outputs = snn_cpu(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == reference).sum()
            total_loss_test += loss.item() 
            total_spk_count += spk_count
            
        acc = 100. * float(correct) / float(total)
        
        # try to improve this
        if self.acc == []:
            self.acc.append([self.epoch, acc]) 
            self.test_loss.append([self.epoch, total_loss_test / total])
        else:
            if self.acc[-1][0] < self.epoch:
                self.acc.append([self.epoch, acc]) 
                self.test_loss.append([self.epoch, total_loss_test / total])               

        if self.test_spk_count == []:
            self.test_spk_count.append([self.epoch, total_spk_count * (self.batch_size / total)]) 
        else:
            if self.test_spk_count[-1][0] < self.epoch:
                self.test_spk_count.append([self.epoch, total_spk_count * (self.batch_size / total)])                 
                
        print('avg spk_count per neuron for all {} timesteps {}'.format(self.win, total_spk_count * (self.batch_size / total)))   
        print('Test Accuracy of the model on the test samples: %.3f' % (acc))

    def conf_matrix(self, test_loader = None, labels = None, criterion=nn.MSELoss() ):
        
        self.save_model()
        snn_cpu = RSNN() # copy of self, doing this to always evaluate on cpu
        snn_cpu.load_model('rsnn', batch_size= self.batch_size)
        
        all_preds = list()
        all_refs = list()
        
        for images, labels in test_loader:
            images = images.float()
            labels = labels.float()
            outputs = snn_cpu(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)
            
            all_preds = all_preds + list(predicted.numpy())
            all_refs = all_refs + list(reference.numpy())
        
        print(confusion_matrix(all_refs, all_preds))
        
    def save_model(self, modelname = 'rsnn'):

        state = {
            'net': self.state_dict(),
            'epoch': self.epoch,
            'acc_record': self.acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'test_spk' : self.test_spk_count,
            'dataset': self.dataset,
            'num_hidden': self.num_hidden,
            'thresh':self.thresh,
            'vreset':self.vreset,
            'win': self.win
        }         
        
        torch.save(state, './checkpoint/' + modelname,  _use_new_zipfile_serialization=False)
 
    def save_to_numpy(self, directory = 'default'):
        layers_location = 'checkpoint/'+ directory

        if not os.path.isdir(layers_location):
            os.mkdir(layers_location)

        weights_biases = []
        snn_state_dict = self.state_dict()
        
        with open(layers_location+'/model_info', 'a') as logs:
            spk = self.test_spk_count[-1][1].detach().cpu().numpy()
            logs.write("avg spk neuron/sample {}".format(spk)) 
            logs.write("\navg spk neuron/timestep {}".format(spk*(self.num_hidden/self.win)))  
        
        for k in snn_state_dict:
            np.savez(layers_location+'/'+k,snn_state_dict[k].data.cpu().numpy())
            weights_biases.append(snn_state_dict[k].data.cpu().numpy())        
 
    def lr_scheduler(self, optimizer, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.98 every lr_decay_epoch epochs."""

        if self.epoch % lr_decay_epoch == 0 and self.epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.98

        return optimizer    

    def plot_loss(self):
        
        test_loss = np.array(self.test_loss)
        train_loss = np.array(self.train_loss)        
        fig = plt.figure()
        plt.plot(train_loss[:,0], train_loss[:,1], label ='train loss')
        plt.plot(test_loss[:,0], test_loss[:,1], label = 'test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        
        return fig   
    
    def load_model(self, modelname=None, location = '', batch_size=256, device='cpu'):
        params = torch.load('./checkpoint'+location+'/'+modelname, map_location=torch.device('cpu'))
        
        if len(params['net']['tau_m_h'])>1:
            tau_m = 'adp'
        else:
            tau_m = 0.8305
        
        self.__init__(params['dataset'], params['num_hidden'], params['thresh'], tau_m, params['vreset'], batch_size, params['win'], device)
        self.load_state_dict(params['net'])
        self.acc = params['acc_record'] 
        self.train_loss = params['train_loss']
        self.test_loss = params['test_loss']     
        
        if 'test_spk' in params.keys():
            self.test_spk_count = params['test_spk']

    def plot_weights(self, w, mode='histogram', ):
        
        name='weight distribution'
        
        if w == 'hh':
            w = self.fc_hh
            name = 'hidden-to-hidden weight distribution'
        elif w == 'ih':
            w = self.fc_ih
            name = 'input-to-hidden weight distribution'
        elif w == 'ho':     
            w = self.fc_ho          
            name = 'hidden-to-output weight distribution'        
        
        w = w.weight.data.cpu().numpy()    
        vmin = np.min(w)
        vmax = np.max(w)
        
        if mode=='histogram':
            if self.device.type == 'cpu':
                w = list(w.reshape(1,-1)[0])
                n, bins, fig = plt.hist(w, bins=200)
            else:
                fig = sns.histplot(w.reshape(1,-1)[0], bins = 200)
            plt.xlabel('weight', fontsize=14)
            plt.ylabel('frequency', fontsize=14)
            plt.title(name, fontsize=16)
            return fig
        elif mode=='matrix':
            fig = plt.figure(figsize=(10,10))
            c= 'RdBu'
            plt.imshow(w, cmap=c, vmin=vmin, vmax=vmax)
            plt.xlabel('input', fontsize=14)
            plt.ylabel('output', fontsize=14)
            plt.title('weights', fontsize=16)
            return fig               

    def quantize_weights(self, bits):
        
        def reduce_precision(weights, bits):
            scale = (1+bits)*(weights.max()-weights.min())/(2*bits+3)
            m = scale*torch.round((weights/scale)*2**bits)/(2**bits)
            return m   

        with torch.no_grad():
            self.fc_hh.weight.data = torch.nn.Parameter(reduce_precision(self.fc_hh.weight.data, 1))
            self.fc_ih.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ih.weight.data, 1))
            self.fc_ho.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ho.weight.data, 1))            
    
    def prune_weights(self, percentage):
        pass
    
    def mask_weights(self, layer, mask, override=False, trainable=True):
        #print(mask)
        #print(self.fc_hh.weight.data)
        if layer.weight.data.shape == mask.shape:    
            new_weight = mask if override else layer.weight.data * mask
            layer.weight = torch.nn.Parameter(new_weight, requires_grad=trainable)             

        else:
            print('Mask weights failed: dimension mismatch')
    
    
# Eexperimental class for training with multiple delays

class RSNN_d(RSNN):
    
    def define_operations(self):
        self.fc_ih_0 = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_ih_2 = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_ih_4 = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)    
        
        self.max_d = 9
        
    def forward(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization
        
        extended_input = torch.zeros(self.batch_size, self.win+self.max_d, self.num_input, device=self.device)
        
        extended_input[:, self.max_d:, :] = input
        
        middle = int(self.max_d/2.0)
        
        for step in range(self.max_d, self.win+self.max_d):
            
            x_0 = extended_input[:, step, :]
            x_2 = extended_input[:, step-middle, :]
            x_4 = extended_input[:, step-self.max_d, :]

            x_0_spike = x_0.view(self.batch_size, -1)
            x_2_spike = x_2.view(self.batch_size, -1)
            x_4_spike = x_4.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(x_0_spike, x_2_spike, x_4_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)
            
            self.h_sumspike = self.h_sumspike + h_spike.sum()
            o_sumspike = o_sumspike + o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs        
    
    def mem_update(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike) - o_spike*self.vreset
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike
    
    def mem_update_rnn(self, x_0, x_2, x_4, o_spike, mem):
        beta= 0.1
        alpha = torch.exp(-1. / self.tau_m_h).to(self.device)
        a0 = self.fc_ih_0(x_0) # process spikes from input
        a2 = self.fc_ih_2(x_2)
        a4 = self.fc_ih_4(x_4)
        
        b = self.fc_hh(o_spike) # process recurrent spikes
        c = mem * alpha * (1-o_spike) - o_spike*self.vreset
        mem = a0 + a2 + a4 + b + c
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike     
    
    def plot_weights(self, w, mode='histogram'):
        
        name = 'weight distribution'
            
        w = w.weight.data.cpu().numpy()    
        vmin = np.min(w)
        vmax = np.max(w)
        
        if mode=='histogram':
            fig = plt.figure()
            sns.histplot(w.reshape(1,-1)[0], bins = 200)
            plt.xlabel('weight', fontsize=14)
            plt.ylabel('frequency', fontsize=14)
            plt.title(name, fontsize=16)
            return fig
        elif mode=='matrix':
            fig = plt.figure(figsize=(10,10))
            c= 'RdBu'
            plt.imshow(w, cmap=c, vmin=vmin, vmax=vmax)
            plt.xlabel('input', fontsize=14)
            plt.ylabel('output', fontsize=14)
            plt.title('weights', fontsize=16)
            return fig
    
    def pool_delays(self):
        
        '''
        Create one delay per synapse in multi-delay model, by choosing the one with highest absolute value
        '''
        
        q = torch.abs(torch.stack([self.fc_ih_0.weight.data, self.fc_ih_2.weight.data, self.fc_ih_4.weight.data]))
        
        q_argmax = torch.argmax(q, dim=0)
        
        m0 = q_argmax==0
        m2 = q_argmax==1
        m4 = q_argmax==2        
        
        self.mask_weights(self.fc_ih_0, m0, override=False, trainable=False)
        self.mask_weights(self.fc_ih_2, m2, override=False, trainable=False)
        self.mask_weights(self.fc_ih_4, m4, override=False, trainable=False)
    
class RSNN_f(RSNN):    

    def define_operations(self):
        
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias= False)    
        
        self.max_d = 9
        
        self.delay_index = nn.Parameter(torch.Tensor(self.num_input))
        nn.init.uniform_(self.delay_index, 0, self.max_d-1)

    def forward(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        extended_input = torch.zeros(self.batch_size, self.win+self.max_d, self.num_input, device=self.device)
        extended_input[:, self.max_d:, :] = input
        
        #idx = torch.ones(self.num_input, dtype=torch.long) # just to test
        #idx = torch.clone(self.delay_index.data).detach().to(torch.long)
        
        idx = self.delay_index.data.to(torch.long)
        
        print(idx)

        for step in range(self.max_d, self.win+self.max_d):
            
            x = extended_input[:, step-self.max_d:step, :]
            
            x = x[:, idx, range(self.num_input)]

            x_spike = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(x_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)

            o_sumspike = o_sumspike + o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs        