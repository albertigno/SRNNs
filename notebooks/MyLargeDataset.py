# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:14:11 2018

@author: yjwu
"""

from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio
import h5py
from sklearn.preprocessing import OneHotEncoder

class MyDataset(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h',lens = 15):
        if method=='h':
            data = h5py.File(path)
            image,label = data['image'],data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images =  self.images[:,:,:,:,:]
            self.labels = torch.from_numpy(label).float()

        elif method=='nmnist_r':
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
            self.images = self.images.permute(0,3,1,2,4)


        elif method=='nmnist_h':
            data = h5py.File(path)
            image, label = data['image'], data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 3, 1, 2, 4)
            
        elif method=='emd':
            data = sio.loadmat(path)
            image, label = data['image'], data['label']
            #image = np.transpose(image)
            #label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 2, 3, 1)
            print("final shape of images: " + str(self.images.shape))
            
        elif method=='emd_spike':
            data = sio.loadmat(path)
            image, label = data['image'], data['label']
            #image = np.transpose(image)
            #label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 2, 3, 4, 1)  
            print("final shape of images: " + str(self.images.shape))
        elif method=='hd_digits':
            data = h5py.File(path, 'r')
            image, label = data['spikes'], data['labels']
            
            x, y = self.sparse_data_generator_from_hdf5_spikes(image, label, lens, 700, 1.4, shuffle=False)
            
            #self.images = x.to_dense()
            self.images = x
            #self.images = self.images.permute(0, 2, 1)  
            
            integer_encoded = y.cpu()
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

            self.labels = torch.from_numpy(onehot_encoded).float()
            print('ora si merito2')
            #y = y.to_dense()
            
            print(x.shape)
            print(y.shape)
            #image = np.transpose(image)
            #label = np.transpose(label)
            #self.images = torch.from_numpy(image)
            #self.images = self.images[:, :, :, :, :]
            #self.labels = torch.from_numpy(label).float()
            #self.images = self.images.permute(0, 3, 1, 2, 4)          
        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()
            
        #self.num_sample = int((len(self.images)//100)*100)
        self.num_sample = len(self.images)
        print('num sample: {}'.format(self.num_sample))
        print(self.images.size(),self.labels.size())

        
    def sparse_data_generator_from_hdf5_spikes(self, X, y, nb_steps, nb_units, max_time, shuffle=True):
        """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 

        Args:
            X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
            y: The labels
        """
        
        device = 'cuda:0'
        
        labels_ = np.array(y,dtype=np.int)
        sample_index = np.arange(len(labels_))

        # compute discrete firing times
        firing_times = X['times']
        units_fired = X['units']

        time_bins = np.linspace(0, max_time, num=nb_steps)

        if shuffle:
            np.random.shuffle(sample_index)

        total_batch_count = 0
        counter = 0
        
        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(sample_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([len(labels_),nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_,device=device)

        return X_batch.to(device=device), y_batch.to(device=device)

        
    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample