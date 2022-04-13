#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:18:38 2021

@author: alberto
"""

import pyNN.nest as sim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import json
import h5py

dt = 1.0

name = 'nmnist'

if name=='heidelberg':
    win = 50
    duration = int(win*dt)
    dataset_path = './../../datasets/hd_digits/hd_test_spk.mat'
    #dataset_path = './../../datasets/hd_digits/hd_train_spk.mat'
    data = scipy.io.loadmat(dataset_path)
    image, label = data['image'], data['label']
    sample_size = 700
    
elif name=='ibm':
    #win = 50 # normal
    win = 25
    duration = int(win*dt)
    dataset_path = './../../datasets/ibm_gestures/gestures_t50_s32_test.mat'
    #dataset_path = './../../datasets/hd_digits/hd_train_spk.mat'
    #data = scipy.io.loadmat(dataset_path)
    data = h5py.File(dataset_path)
    
    image, label = data['image'], data['label']
    
    image = np.transpose(image)
    label = np.transpose(label)  
    
    
    image = image.transpose(0, 3, 1, 2, 4)
    sample_size = 32*32*2       

elif name=='ibm2':
    win = 50 # normal
    #win = 25
    duration = int(win*dt)
    dataset_path = './../../datasets/ibm_gestures/gestures_v2_t200_s32_test.mat'
    #dataset_path = './../../datasets/hd_digits/hd_train_spk.mat'
    #data = scipy.io.loadmat(dataset_path)
    data = h5py.File(dataset_path)
    
    image, label = data['image'], data['label']
    
    image = np.transpose(image)
    label = np.transpose(label)  
    
    
    image = image.transpose(0, 3, 1, 2, 4)
    sample_size = 32*32*2       


elif name=='syn':
    win = 50
    duration = int(win*dt)
    dataset_path = './../../datasets/hd_digits/synthetic_input.mat'
    #dataset_path = './../../datasets/hd_digits/hd_test_spk.mat'
    data = scipy.io.loadmat(dataset_path)
    image, label = data['image'], None
    sample_size = 700    
 
else:
    win = 25
    duration = int(win*dt)
    #dataset_path = './../../datasets/nmnist/NMNIST_small_test_data.mat'
    dataset_path = './../../datasets/nmnist/NMNIST_test_data.mat'
    #dataset_path = './../../datasets/nmnist/NMNIST_train_data.mat'
    
    data = scipy.io.loadmat(dataset_path)
    #data = h5py.File(dataset_path)
    
    image, label = data['image'], data['label']
    #image = np.transpose(image)
    #label = np.transpose(label)    
    
    #image = image[:10000, ... ]
    #label = label[:10000, ...]
    
    image = image.transpose(0, 3, 1, 2, 4)
    sample_size = 34*34*2


print('loading finished')

sim.setup(timestep=dt, min_delay=dt, max_delay=10.0)


input_celltype = sim.SpikeSourceArray()
input_pop = sim.Population(sample_size, input_celltype)

input_pop. record(['spikes'])

inicio = 0
num_to_test = 100
acc = 0.0

num_timesteps = duration/dt

next_sample_delay = 25 # nmnist
#next_sample_delay = 30 # shd


total_duration = duration+next_sample_delay

for idx, sample_idx in enumerate(range(inicio,num_to_test+inicio)):

    sample_image = image[sample_idx]
    flat_image = np.empty((win, sample_size))
    
    if name=='heidelberg' or name=='syn':
        for t in range(win): 
            flat_image[t,:] = sample_image[t,:].reshape((1,-1))
        fl_i = flat_image.swapaxes(1,0).tolist()          
    # if name=='heidelberg':
    #     for t in range(win): 
    #         flat_image[t,:] = sample_image[t,:].reshape((1,-1))        
    #     fl_i = np.flipud(flat_image.swapaxes(1,0)).tolist()                              
    else:
        for t in range(win): 
            flat_image[t,:] = sample_image[:,:,:,t].reshape((1,-1))
    
        fl_i = flat_image.swapaxes(1,0).tolist()
    
    spike_times = []
    
    time_offset = 2 # spikes en cero no permitidos en nest
    
    segment_start = time_offset + (next_sample_delay + duration/dt)*idx
    
    print(segment_start)
    
    for neuron_id in range(sample_size):
        spike_times.append([segment_start+float(i) for i,x in enumerate(fl_i[neuron_id]) if x == 1])
    
    input_pop.set(spike_times=spike_times)
    
    sim.run(total_duration)
        
    # sim.reset()

spiketrains_input = input_pop.get_data().segments[-1].spiketrains

def spk_filter(spiketrain, start, end):
    return list(filter(lambda x: x>=start and x<end, spiketrain))

def data_to_list(data, start, end):    
    return [spk_filter(list(np.array(x)), start, end) for x in data]


d = data_to_list(spiketrains_input, 0, total_duration*num_to_test)

label = np.array(label, dtype='float')
label_list = [ list(x) for x in label ]
results_t = {"input_spikes":d, "label" : label_list, "start": inicio}

with open("{}_{}_delay_{}.json".format(name,num_to_test, next_sample_delay), 'w') as outfile:
    json.dump(results_t, outfile)


if num_to_test < 100:
    
    fig = plt.figure()

    plt.eventplot(spiketrains_input, linelengths=0.7, colors='k', label='pre_spikes') 
    plt.ylabel('Neuron index')
    plt.xlabel('Time (ms)') 
    for x in range(num_to_test-1):
        plt.vlines(7+total_duration*(x+1), -1, sample_size, 'g', 'dashed')
    #fig.savefig('nest_last.png',dpi=300)
    plt.show()

