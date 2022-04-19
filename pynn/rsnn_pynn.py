#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 12 2022

@author: alberto
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import json
import os

class RSNN_PyNN():
    
    def __init__(self, sim, timestep =1.0, dataset='mnist', modelname='256/', num_to_test= 20):
        
        self.sim = sim
        self.name = dataset
        self.timestep = timestep
        self.num_to_test = num_to_test
        self.sim.setup(timestep, min_delay=1.0, max_delay=2.0)
        
        weights = []
        
        if self.name=='heidelberg':
            self.win = 50
            self.next_sample_delay = 30
            sample_size = 700
            input_file = "./input_spiketrains/heidelberg_{}_delay_{}.json".format(self.num_to_test, self.next_sample_delay)
            weights_path = './models/heidelberg_'+modelname
            weights.append(np.load(weights_path+'fc_ih.weight.npz')['arr_0'])
            weights.append(np.load(weights_path+'fc_hh.weight.npz')['arr_0'])
            weights.append(np.load(weights_path+'fc_ho.weight.npz')['arr_0'])

            v_th_h = 0.77
            off = 0.34   

            inicio = 0
        
        else:
            self.win = 25
            self.next_sample_delay = 25
            sample_size = 34*34*2
            input_file = "./input_spiketrains/nmnist_{}_delay_{}.json".format(self.num_to_test, self.next_sample_delay)
            weights_path = './models/nmnist_'+modelname
            weights.append(np.load(weights_path+'fc_ih.weight.npz')['arr_0'])
            weights.append(np.load(weights_path+'fc_hh.weight.npz')['arr_0'])
            weights.append(np.load(weights_path+'fc_ho.weight.npz')['arr_0'])
            v_th_h = 0.55
            off = 0.3

        self.inicio = 0    
        
        self.total_duration = self.win+self.next_sample_delay

        with open(input_file) as jsonfile:
            data = json.load(jsonfile)

        spike_times = data['input_spikes']
        self.label = np.array(data['label'])

        delay = 1.0
        v_th = 0.3
        q=0.1
        tau_syn = 0.01/q        
        e_rev = 10000
        
        cellvalues = {'v': 0.0}

        #cm = e_rev/181.0
        cm = e_rev/345.0
        #cm = e_rev/333.0

        cellparams = {'v_thresh': v_th, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': e_rev, 'e_rev_I': -e_rev, 'i_offset': 0.0,
                      'cm': cm/q, 'tau_m': 0.8305, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

        celltype = self.sim.IF_cond_exp(**cellparams)

        input_celltype = self.sim.SpikeSourceArray()
        self.input_pop = self.sim.Population(sample_size, input_celltype)

        self.liquid_pop = self.sim.Population(len(weights[0]), celltype)
        self.output_pop = self.sim.Population(len(weights[-1]), celltype)    

        i_o = 0.7
        cellparams_h = {'v_thresh': v_th_h, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 0.1, 'e_rev_I': -v_th, 'i_offset': i_o,
                       'cm': 50.0, 'tau_m': 100.0, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}


        # h_pop = sim.Population(len(weights[0]), sim.IF_cond_exp(**cellparams_h))

        # sim.Projection(h_pop,liquid_pop, connector=sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=dt),
        #                     receptor_type='inhibitory')

        h_pop = self.sim.Population(1, sim.IF_cond_exp(**cellparams_h))
        
        

        self.sim.Projection(h_pop,self.liquid_pop, connector=sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=1.0),
                            receptor_type='inhibitory')

        self.load_populations(weights, delay=1.0)

        self.liquid_pop.initialize(**cellvalues)
        self.output_pop.initialize(**cellvalues)    
        h_pop.initialize(v=off) 

        self.output_pop.record(['spikes'])

        self.input_pop.set(spike_times=spike_times)  


    def run(self):

        self.sim.run(self.total_duration*self.num_to_test / self.timestep)

        spiketrains_output = self.output_pop.get_data().segments[-1].spiketrains

        ## PREDICTIONS

        out_spks_nope = self.spk_to_array(spiketrains_output)
        out_spks = np.zeros((len(spiketrains_output),self.win*self.num_to_test))

        preds = []
        for x in range(self.num_to_test):
            a = x*self.win
            b = x*(self.win+self.next_sample_delay)

            out_spks[:,a:a+self.win] = out_spks_nope[:,b:b+self.win]

            preds.append(out_spks[:,a:a+self.win].sum(axis=1).argmax())


        print(self.label[self.inicio:self.inicio+self.num_to_test].argmax(axis=1))

        print(np.array(preds))

        acc = np.float(np.sum(np.array(preds) == self.label[self.inicio:self.inicio+self.num_to_test].argmax(axis=1)))
        print('accuracy: ' +str(100*(acc/self.num_to_test)) + '%')        
        

    def load_populations(self, weights, delay):

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

                p1_inh = self.sim.Projection(self.input_pop,self.liquid_pop, connector=self.sim.FromListConnector(inh_synapses),
                                receptor_type='inhibitory')
                p1_exc = self.sim.Projection(self.input_pop,self.liquid_pop, connector=self.sim.FromListConnector(exc_synapses),
                                receptor_type='excitatory')

            elif layer == 1:

                p2_inh = self.sim.Projection(self.liquid_pop,self.liquid_pop, connector=self.sim.FromListConnector(inh_synapses),
                                receptor_type='inhibitory')
                p2_exc = self.sim.Projection(self.liquid_pop,self.liquid_pop, connector=self.sim.FromListConnector(exc_synapses),
                                receptor_type='excitatory')

            elif layer == 2:

                p3_inh = self.sim.Projection(self.liquid_pop,self.output_pop, connector=self.sim.FromListConnector(inh_synapses),
                                receptor_type='inhibitory')
                p3_exc = self.sim.Projection(self.liquid_pop,self.output_pop, connector=self.sim.FromListConnector(exc_synapses),
                                receptor_type='excitatory')    

        

    def spk_count(self, spiketrain, start, end):

        spikecounts = np.zeros(len(spiketrain))
        for neuron, spk in enumerate(spiketrain):
            spikecounts[neuron] =  len(list(filter(lambda x: x>=start and x<end, spk)))

        return spikecounts.argmax()

    def spk_to_array(self, spiketrain):
        n = len(spiketrain)
        t = self.total_duration*self.num_to_test
        spks = np.zeros((n,t))
        for neu_idx, spk in enumerate(spiketrain):
            for time in spk:
                spks[neu_idx, int(time)-1] = 1
        return spks    

    
