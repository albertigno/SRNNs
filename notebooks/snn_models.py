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
    
    def __init__(self, dataset='mnist', num_hidden=256, thresh=0.3, tau_m=0.83, batch_size=256, win=50, device='cuda'):
        super(RSNN, self).__init__()
        
        self.act_fun = ActFun.apply
        
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        
        self.dataset = dataset
        
        self.epoch = 0
        
        if self.dataset=='mnist':
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
            #nn.init.uniform_(self.tau_m_h, 0.5, 10.0)
            nn.init.normal_(self.tau_m_h, 0.83, 0.1)
            #nn.init.normal_(self.tau_m_h, 9.0, 0.1)
            #self.decay_o = nn.Parameter(torch.Tensor(self.num_output))    
            #nn.init.uniform_(self.decay_o, 0.1, 0.9)
            self.tau_m_o = torch.Tensor([0.83])
        
        
        self.acc = list() # store accuracy every time test() is called
        self.train_loss = list()
        self.test_loss = list()
        
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
            
            x = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(self.fc_ih, self.fc_hh, x, h_spike, h_mem)
                        
            o_mem, o_spike = self.mem_update(self.fc_ho, h_spike, o_mem, o_spike)

            o_sumspike = o_sumspike + o_spike
        
        outputs = o_sumspike / (self.win)
        
        return outputs        
    
    def mem_update(self, operation, x, mem, spike):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - spike) + operation(x)
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
    def mem_update_rnn(self, operation1, operation2, x, h_spike, mem):
        beta= 0.1
        alpha = torch.exp(-1. / self.tau_m_h).to(self.device)
        a = operation1(x)
        b = operation2(h_spike)
        c = mem * alpha * (1-h_spike) - h_spike*beta
        mem = a + b + c
        spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, spike
    
    # TBD
        
    def train_step(self, train_loader=None, optimizer=None, criterion=None, num_samples=0):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size

        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self(images)
            loss = criterion(outputs, labels)
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
        correct = 0
        total = 0
        total_loss_test = 0
        for images, labels in test_loader:
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == reference).sum()
            total_loss_test += loss.item() 
            
        acc = 100. * float(correct) / float(total)
        
        # try to improve this
        if self.acc == []:
            self.acc.append([self.epoch, acc]) 
            self.test_loss.append([self.epoch, total_loss_test / total])
        else:
            if self.acc[-1][0] < self.epoch:
                self.acc.append([self.epoch, acc]) 
                self.test_loss.append([self.epoch, total_loss_test / total])               

   
        print('Test Accuracy of the model on the test samples: %.3f' % (acc))

    def save_model(self, modelname = 'rsnn'):

        state = {
            'net': self.state_dict(),
            'epoch': self.epoch,
            'acc_record': self.acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'dataset': self.dataset,
            'num_hidden': self.num_hidden,
            'thresh':self.thresh,
            'win': self.win
        }         
        
        torch.save(state, './checkpoint/' + modelname,  _use_new_zipfile_serialization=False)
 
    def save_to_numpy(self, directory = 'default'):
        layers_location = 'checkpoint/'+ directory

        if not os.path.isdir(layers_location):
            os.mkdir(layers_location)

        weights_biases = []
        snn_state_dict = self.state_dict()
        
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
    
    def load_model(self, modelname=None, batch_size=256, device='cpu'):
        params = torch.load('./checkpoint/'+modelname, map_location=torch.device('cpu'))
        
        if len(params['net']['tau_m_h'])>1:
            tau_m = 'adp'
        else:
            tau_m = 0.8305
        
        self.__init__(params['dataset'], params['num_hidden'], params['thresh'], tau_m, batch_size, params['win'], device)
        self.load_state_dict(params['net'])
        self.acc = params['acc_record'] 
        self.train_loss = params['train_loss']
        self.test_loss = params['test_loss']        
    
    def plot_weights(self, layer = 'hh', mode='histogram'):
        
        if layer == 'hh':
            w = self.fc_hh
            name = 'hidden-to-hidden weight distribution'
        elif layer == 'ih':
            w = self.fc_ih
            name = 'input-to-hidden weight distribution'
        elif layer == 'ho':     
            w = self.fc_ho          
            name = 'hidden-to-output weight distribution'
            
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
    
    def mask_weights(self, mask):
        pass