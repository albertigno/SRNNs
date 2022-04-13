#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:48:36 2021

@author: alberto
"""

import pyNN.nest as sim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import json
import os

dt = 1.0

name = 'nmnist'
#name = 'heidelberg'
num_to_test = 20

n_h = 256

#modelname = 'rnn_{}/'.format(n_h)
modelname = 'rnn_lsm_{}/'.format(n_h)


weights = []
biases =[]

plots = False # only for less than 10 in n_h

if name=='heidelberg':
    win = 50
    next_sample_delay = 30
    sample_size = 700
    input_file = "heidelberg_{}_delay_{}.json".format(num_to_test, next_sample_delay)
    #weights_path = './../SpeechHeidelberg/checkpoint/heidelberg_'+modelname
    weights_path = './../HD_tests/checkpoint/heidelberg_'+modelname
    
    weights.append(np.load(weights_path+'fc_ih.weight.npz')['arr_0'])
    weights.append(np.load(weights_path+'fc_hh.weight.npz')['arr_0'])
    weights.append(np.load(weights_path+'fc_ho.weight.npz')['arr_0'])
    
    # biases.append(np.load(weights_path+'fc_ih.bias.npz')['arr_0'])
    # biases.append(np.load(weights_path+'fc_hh.bias.npz')['arr_0'])
    # biases.append(np.load(weights_path+'fc_ho.bias.npz')['arr_0'])
    
    # weights.append(np.load(weights_path+'rnn0.weight_ih_l0.npz')['arr_0'])
    # biases.append(np.load(weights_path+'rnn0.bias_ih_l0.npz')['arr_0'])
    
    # weights.append(np.load(weights_path+'rnn0.weight_hh_l0.npz')['arr_0'])
    # biases.append(np.load(weights_path+'rnn0.bias_hh_l0.npz')['arr_0'])
    
    # weights.append(np.load(weights_path+'fc.weight.npz')['arr_0'])
    # biases.append(np.load(weights_path+'fc.bias.npz')['arr_0'])
    
    # biases[0] = biases[1] + biases[0]
    
    v_th_h = 0.77
    off = 0.34

    #v_th_h = 0.88
    #off = 0.54        
    
    inicio = 0
        
else:
    win = 25
    next_sample_delay = 25
    
    sample_size = 34*34*2
    
    input_file = "nmnist_{}_delay_{}.json".format(num_to_test, next_sample_delay)
    weights_path = './../LSM Frontiers/checkpoint/nmnist_'+modelname

    weights.append(np.load(weights_path+'fc_ih.weight.npz')['arr_0'])
    weights.append(np.load(weights_path+'fc_hh.weight.npz')['arr_0'])
    weights.append(np.load(weights_path+'fc_ho.weight.npz')['arr_0'])
    
    v_th_h = 0.55
    off = 0.3
    
    inicio = 0


def spk_count(spiketrain, start, end):
    
    spikecounts = np.zeros(len(spiketrain))
    for neuron, spk in enumerate(spiketrain):
        spikecounts[neuron] =  len(list(filter(lambda x: x>=start and x<end, spk)))
        
    return spikecounts.argmax()

def spk_to_array(spiketrain):
    n = len(spiketrain)
    t = total_duration*num_to_test
    spks = np.zeros((n,t))
    for neu_idx, spk in enumerate(spiketrain):
        for time in spk:
            spks[neu_idx, int(time)-1] = 1
    return spks


delay = dt

sim.setup(timestep=dt, min_delay=delay, max_delay=10.0)


#sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 50)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 120)

#sim.set_number_of_neurons_per_core(sim.IF_cond_exp,255)

total_duration = win+next_sample_delay

with open(input_file) as jsonfile:
    data = json.load(jsonfile)

spike_times = data['input_spikes']
label = np.array(data['label'])

delay = 1.0
v_th = 0.3
q=0.1
tau_syn = 0.01/q
#e_rev =30.0
#cm = e_rev/200.0

#e_rev = 2000
#cm = e_rev/181.0

#e_rev = 1000
#cm = e_rev/181.0
#cm = e_rev/200.0

#e_rev = 100.0
#cm = e_rev/181.0

e_rev = 10000
#cm = e_rev/174.0
#cm = e_rev/174.0


#cm = e_rev/181.0

cellvalues = {'v': 0.0}


#cm = e_rev/181.0
cm = e_rev/345.0
#cm = e_rev/333.0

cellparams = {'v_thresh': v_th, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': e_rev, 'e_rev_I': -e_rev, 'i_offset': 0.0,
              'cm': cm/q, 'tau_m': 0.8305, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

celltype = sim.IF_cond_exp(**cellparams)


input_celltype = sim.SpikeSourceArray()
input_pop = sim.Population(sample_size, input_celltype)

liquid_pop = sim.Population(len(weights[0]), celltype)
output_pop = sim.Population(len(weights[-1]), celltype)    

i_o = 0.7
cellparams_h = {'v_thresh': v_th_h, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 0.1, 'e_rev_I': -v_th, 'i_offset': i_o,
               'cm': 50.0, 'tau_m': 100.0, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

def load_populations():
    
    for layer in range(len(weights)):
        
        inh_synapses = []
        exc_synapses = []
        
        for i in range(weights[layer].shape[1]):
            for j in range(weights[layer].shape[0]):
                if float(weights[layer][j, i])<0.0:
                    inh_synapses.append([i, j, -1.0*weights[layer][j, i], delay])
                else:
                    exc_synapses.append([i, j, weights[layer][j, i], delay])          
            
        if layer == 0:
            
            p1_inh = sim.Projection(input_pop,liquid_pop, connector=sim.FromListConnector(inh_synapses),
                            receptor_type='inhibitory')
            p1_exc = sim.Projection(input_pop,liquid_pop, connector=sim.FromListConnector(exc_synapses),
                            receptor_type='excitatory')
    
        elif layer == 1:
            
            p2_inh = sim.Projection(liquid_pop,liquid_pop, connector=sim.FromListConnector(inh_synapses),
                            receptor_type='inhibitory')
            p2_exc = sim.Projection(liquid_pop,liquid_pop, connector=sim.FromListConnector(exc_synapses),
                            receptor_type='excitatory')
    
        elif layer == 2:
                    
            p3_inh = sim.Projection(liquid_pop,output_pop, connector=sim.FromListConnector(inh_synapses),
                            receptor_type='inhibitory')
            p3_exc = sim.Projection(liquid_pop,output_pop, connector=sim.FromListConnector(exc_synapses),
                            receptor_type='excitatory')
            
# h_pop = sim.Population(len(weights[0]), sim.IF_cond_exp(**cellparams_h))

# sim.Projection(h_pop,liquid_pop, connector=sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=dt),
#                     receptor_type='inhibitory')

h_pop = sim.Population(1, sim.IF_cond_exp(**cellparams_h))

sim.Projection(h_pop,liquid_pop, connector=sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=dt),
                    receptor_type='inhibitory')

load_populations()

liquid_pop.initialize(**cellvalues)
output_pop.initialize(**cellvalues)    
h_pop.initialize(v=off) 

if plots:    
    input_pop. record(['spikes'])
    liquid_pop. record(['spikes','v','gsyn_exc','gsyn_inh'])
    output_pop. record(['spikes','v'])
    h_pop.record(['spikes', 'v'])

else:
    output_pop.record(['spikes'])
    
input_pop.set(spike_times=spike_times)  

sim.run(total_duration*num_to_test / dt)

spiketrains_output = output_pop.get_data().segments[-1].spiketrains

## PREDICTIONS

out_spks_nope = spk_to_array(spiketrains_output)
out_spks = np.zeros((len(spiketrains_output),win*num_to_test))

preds = []
for x in range(num_to_test):
    a = x*win
    b = x*(win+next_sample_delay)
    
    out_spks[:,a:a+win] = out_spks_nope[:,b:b+win]
    
    preds.append(out_spks[:,a:a+win].sum(axis=1).argmax())
                                
    
print(label[inicio:inicio+num_to_test].argmax(axis=1))

print(np.array(preds))

acc = np.float(np.sum(np.array(preds) == label[inicio:inicio+num_to_test].argmax(axis=1)))
print('accuracy: ' +str(100*(acc/num_to_test)) + '%')


plt.close('all')

if plots and num_to_test<=10:
    
    plots_folder = './results/nest_'+name+'_'+modelname
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    membranes_output = output_pop.get_data().segments[-1].filter(name='v')[0]
    spiketrains_liquid = liquid_pop.get_data().segments[-1].spiketrains
    membranes_liquid = liquid_pop.get_data().segments[-1].filter(name='v')[0]
    spiketrains_input = input_pop.get_data().segments[-1].spiketrains    
    spiketrains_h = h_pop.get_data().segments[-1].spiketrains
    membranes_h = h_pop.get_data().segments[-1].filter(name='v')[0]  
    g_e = np.array(liquid_pop.get_data().segments[-1].filter(name='gsyn_exc')[0])
    g_i = np.array(liquid_pop.get_data().segments[-1].filter(name='gsyn_inh')[0])

    # fig = plt.figure('input')
    # plt.eventplot(spiketrains_input, linelengths=0.7, colors='k', label='pre_spikes') 
    # plt.ylabel('Neuron index')
    # plt.xlabel('Time (ms)') 
    # for x in range(num_to_test-1):
    #     plt.vlines(4+total_duration*(x+1), -1, sample_size, 'g', 'dashed')
    # #fig.savefig('nest_last.png',dpi=300)
    # plt.show()
    
    fig = plt.figure('hidden')
    plt.eventplot(spiketrains_liquid, linelengths=0.7, colors='k', label='pre_spikes') 
    plt.ylabel('Neuron index')
    plt.xlabel('Time (ms)') 
    for x in range(num_to_test-1):
        plt.vlines(4+total_duration*(x+1), -1, 25, 'g', 'dashed')
    plt.savefig(plots_folder+'/hidden_spk.png',dpi=300)
    plt.show()
    
    mems_nope = np.array(membranes_liquid)
    spks_nope = spk_to_array(spiketrains_liquid)
    
    mems = np.zeros((win*num_to_test,len(spiketrains_liquid)))
    spks = np.zeros((len(spiketrains_liquid),win*num_to_test))
    
    for x in range(num_to_test):
        a = x*win
        b = x*(win+next_sample_delay)
        mems[a:a+win,:] = mems_nope[b:b+win,:]
        spks[:,a:a+win] = spks_nope[:,b:b+win]
        
        
    np.save(plots_folder+'/mems.npy', mems.T)
    np.save(plots_folder+'/spks.npy', spks)

    means_n = spks.mean(axis=1)
    means_t = spks.mean(axis=0)
    
    plotname = '{}, {} neurons'.format(name,n_h)
    
    plt.figure()
    plt.title('Recurrent layer activity per neuron, ' + plotname)
    plt.plot(means_n, label='avg spike rate')
    plt.plot(mems.mean(axis=0), label= 'avg membrane potential')
    plt.xlabel('Neuron')
    plt.savefig(plots_folder+'/mems_n.png',dpi=300)
    plt.legend()
    
    plt.figure()
    plt.title('Recurrent layer activity per timestep, ' + plotname)
    plt.plot(means_t, label='avg spike rate')
    plt.plot(mems.mean(axis=1), label='avg membrane potential')
    plt.xlabel('Time (ms)')
    plt.savefig(plots_folder+'/mems_t.png',dpi=300)
    plt.legend()
    
    fig = plt.figure('homeostatic')
    plt.eventplot(spiketrains_h, linelengths=0.7, colors='k', label='pre_spikes') 
    plt.ylabel('Neuron index')
    plt.xlabel('Time (ms)') 
    for x in range(num_to_test-1):
        plt.vlines(4+total_duration*(x+1), -1, 25, 'g', 'dashed')
    plt.savefig(plots_folder+'/resetter.png',dpi=300)
    plt.show()
    
    plt.figure()
    plt.title('Recurrent layer g per timestep, ' + plotname)
    plt.plot(g_e.mean(axis=1), label='g_e')
    plt.plot(g_i.mean(axis=1), label='g_i')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.savefig(plots_folder+'/g_syn.png',dpi=300)
    plt.show()

if n_h<=100:

    fig = plt.figure('output')
    plt.eventplot(spiketrains_output, linelengths=0.7, colors='k', label='pre_spikes') 
    plt.ylabel('Neuron index')
    plt.xlabel('Time (ms)') 
    for x in range(num_to_test-1):
        plt.vlines(4+total_duration*(x+1), -1, 10, 'g', 'dashed')
    plt.savefig(plots_folder+'/output_spk.png',dpi=300)
    plt.show()