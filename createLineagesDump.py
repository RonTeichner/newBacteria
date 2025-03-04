#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:53:09 2025

@author: ron.teichner
"""

from main_func import SDE_1D, plot, get_divisionTime, plot_cs_and_lineage, get_CR_str, calc_FTPD_wrapper, createLineages, get_FTPD_bins, rand_FTPD, calc_logLikelihood
import torch
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from time import sleep

noiseLess = False
if noiseLess:
    noiseStr = '_noiseless_'
else:
    noiseStr = ''

mu_eta = 0.5
sigma_eta = 0.05
gamma_shape, gamma_scale = 25, 9.4e-4


nGenerations = 100
batch_size = 10


for mechanismType in ['sizer', 'adder']:
    if mechanismType == 'sizer':
        mu_u, sigma_u = 1.0, 0.1 # [mu m]
    elif mechanismType == 'adder':
        mu_u, sigma_u = 0.5*1.0, 0.5*0.1 # [mu m]
        
    for tau_u in [2,40,200]:

        filename = mechanismType + noiseStr + '_tau_' + str(tau_u) + '_lineages.pt'
        print('starting ' + filename)
        sleep(1)
        if not(os.path.isfile(filename)):
            observations, u, ts = createLineages(mu_eta, sigma_eta, gamma_shape, gamma_scale, mu_u, sigma_u, tau_u, nGenerations, batch_size, mechanismType, noiseLess=noiseLess, sdeRes='constant')
            # observations: [f, xb, xd, T])
            pickle.dump({'observations': observations, 'u': u, 'ts': ts}, open(filename, 'wb'))
