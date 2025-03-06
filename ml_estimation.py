#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:53:09 2025

@author: ron.teichner
"""

from main_func import SDE_1D, plot, get_divisionTime, plot_cs_and_lineage, get_CR_str, calc_FTPD_wrapper, createLineages, get_FTPD_bins, rand_FTPD, calc_logLikelihood, ml_optimization, FPTDDataset
import torch
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from time import sleep
import time

nGenerations = 100   
nLineages = 10
noiseLess = False
shuffle = True

if noiseLess:
    noiseStr = '_noiseless_'
else:
    noiseStr = ''
    
if shuffle:
    shuffleStr = '_shuffle_'
else:
    shuffleStr = ''



for tau_u in [40,200,2]:
    for mechanismType in ['sizer', 'adder']:
        starttime = time.time()
        #filename = mechanismType + '_tau_' + str(tau_u) + '_lineages.pt'
        filename = mechanismType + noiseStr + '_tau_' + str(tau_u) + '_lineages.pt'
        
        lineageDict = pickle.load(open(filename, 'rb'))
        observations, u, ts = lineageDict['observations'], lineageDict['u'], lineageDict['ts']
        # observations: [f, xb, xd, T])
        #print(mechanismType + ' tau_u = ' + str(tau_u) + f': min(T) = {observations[:,:,3].min()}, max(T) = {observations[:,:,3].max()}')
        print(mechanismType + noiseStr + ' tau_u = ' + str(tau_u) + f': min(u) = {str(round(u.min(),3))}, max(u) = {str(round(u.max(),3))}; min(f) = {str(round(observations[:,:,0].min(),3))}, max(f) = {str(round(observations[:,:,0].max(),3))}, mean(xd[0]) = {str(round(observations[:,0,2].mean(),5))}, var(xd[0]) = {str(round(observations[:,0,2].var(),5))}; mean(xd) = {str(round(observations[:,:,2].mean(),5))}, var(xd) = {str(round(observations[:,:,2].var(),5))}, min(xb) = {str(round(observations[:,:,1].min(),5))}, , max(xb) = {str(round(observations[:,:,1].max(),5))}')

u0_lim = torch.asarray([0.2, 1.6])
xb_lim = torch.asarray([0.2, 1.0])
alpha_lim = torch.asarray([0.01, 0.04])
mu_u_lim = torch.asarray([0.1, 2.5])
sigma_u_lim = torch.asarray([0.01, 0.25])
tau_u_lim = torch.asarray([1, 450])


#dataset = FPTDDataset('sizer', 'validation')
bounds = [(mu_u_lim[0].item(), mu_u_lim[1].item()),
          (sigma_u_lim[0].item(), sigma_u_lim[1].item()),
          (tau_u_lim[0].item(), tau_u_lim[1].item())]



resultsList = list()
for lineageIdx in range(nLineages):
    for tau_u in [40,200,2]:
        for mechanismType in ['sizer', 'adder']:
            
            if mechanismType == 'sizer':
                mu_u_gt, sigma_u_gt = 1.0, 0.1 # [mu m]
            elif mechanismType == 'adder':
                mu_u_gt, sigma_u_gt = 0.5*1.0, 0.5*0.1 # [mu m]
            
            starttime = time.time()
            #filename = mechanismType + '_tau_' + str(tau_u) + '_lineages.pt'
            filename = mechanismType + noiseStr + '_tau_' + str(tau_u) + '_lineages.pt'
            
            lineageDict = pickle.load(open(filename, 'rb'))
            observations, u, ts = lineageDict['observations'][:,:nGenerations], lineageDict['u'], lineageDict['ts']
            # observations: [f, xb, xd, T])
        
            if shuffle:
                observations = observations[:,np.random.permutation(np.arange(nGenerations))]
        
            nlogLikelihood_gt = calc_logLikelihood(np.array([mu_u_gt, sigma_u_gt, tau_u]), observations[lineageIdx], mechanismType)
            
            gtDict = {'mechanismType': mechanismType, 'lineageIdx': lineageIdx, 'mu_u_gt': mu_u_gt, 'sigma_u_gt': sigma_u_gt, 'tau_u': tau_u, 'nll': nlogLikelihood_gt}
            print('')
            print('gt:' + mechanismType + noiseStr + shuffleStr + f' lineage {lineageIdx} out of {nLineages} nGens={nGenerations}: mu_u_gt = {str(round(mu_u_gt, 4))}, sigma_u_gt = {str(round(sigma_u_gt, 4))}, tau_u_gt = {str(round(tau_u, 4))}. nll_gt = {str(round(nlogLikelihood_gt, 4))}')
            
            mu_u_est, sigma_u_est, tau_u_est, nloglikelihood_est = ml_optimization(observations[lineageIdx], 'sizer', bounds)
            sizer_est_dict = {'mu_u_est': mu_u_est, 'sigma_u_est': sigma_u_est, 'tau_u_est': tau_u_est, 'nll': nloglikelihood_est}
            
            print(f'sizer-est: mu_u_est = {str(round(mu_u_est, 4))}, sigma_u_est = {str(round(sigma_u_est, 4))}, tau_u_est = {str(round(tau_u_est, 4))}. nll_est = {str(round(nloglikelihood_est, 4))}')
            
            mu_u_est, sigma_u_est, tau_u_est, nloglikelihood_est = ml_optimization(observations[lineageIdx], 'adder', bounds)
            adder_est_dict = {'mu_u_est': mu_u_est, 'sigma_u_est': sigma_u_est, 'tau_u_est': tau_u_est, 'nll': nloglikelihood_est}
            #print(mechanismType + f': mu_u_gt = {str(round(mu_u_gt, 4))}, sigma_u_gt = {str(round(sigma_u_gt, 4))}, tau_u_gt = {str(round(tau_u, 4))}. nll_gt = {str(round(nlogLikelihood_gt, 4))}')
            print(f'adder-est: mu_u_est = {str(round(mu_u_est, 4))}, sigma_u_est = {str(round(sigma_u_est, 4))}, tau_u_est = {str(round(tau_u_est, 4))}. nll_est = {str(round(nloglikelihood_est, 4))}')
            
            resultsDict = {'gtDict': gtDict, 'sizer_est_dict': sizer_est_dict, 'adder_est_dict': adder_est_dict}
            resultsList.append(resultsDict)
            pickle.dump(resultsList, open('new_results_nGnes_' + noiseStr + shuffleStr + str(nGenerations) + '_.pt', 'wb'))
            stoptime = time.time()
            print(f'lineage processing time is {str(round(stoptime-starttime))} sec')
