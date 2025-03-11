#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:52:28 2025

@author: ron.teichner
"""

from main_func import SDE_1D, plot, get_divisionTime, plot_cs_and_lineage, get_CR_str, calc_FTPD_wrapper, createLineages, get_FTPD_bins, rand_FTPD, calc_logLikelihood
import torch
import numpy as np
import pickle
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

axx,ab = 16/3, 9/3

noiseLess = False
if noiseLess:
    noiseStr = '_noiseless_'
else:
    noiseStr = ''
    
tau_cs = [2,40,200]
sizerMechnismList, adderMechanismList = list(), list()
for tau_u in tau_cs:
    for mechanismType in ['sizer', 'adder']:
        filename = 'largeDB_' + mechanismType + noiseStr + '_tau_' + str(tau_u) + '_lineages.pt'
        
        lineageDict = pickle.load(open(filename, 'rb'))
        observations, u, ts = lineageDict['observations'], lineageDict['u'], lineageDict['ts']
        if mechanismType == 'sizer':
            sizerMechnismList.append(observations)
        else:
            adderMechanismList.append(observations)


fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=False, figsize=(axx*3,ab*2))
for j,observationList, mechanismType in zip(np.arange(2), [sizerMechnismList, adderMechanismList], ['sizer', 'adder']):
    
    for i,observations in enumerate(observationList):
      ax = axs[j,i]
      xb, xd = observations[:,:,1].flatten(), observations[:,:,2].flatten() # observations: [f, xb, xd, T])
      xbMin, xbMax = np.quantile(xb, 5/100), np.quantile(xb, 95/100)
      xd_Minus_xbMin, xd_Minus_xbMax = np.quantile(xd-xb, 5/100), np.quantile(xd-xb, 95/100)
      H, xedges, yedges = np.histogram2d(xb, xd-xb, bins=(np.linspace(xbMin, xbMax, 30), np.linspace(xd_Minus_xbMin, xd_Minus_xbMax, 30)), density=True)#, density=True)
      # Histogram does not follow Cartesian convention (see Notes),
      # therefore transpose H for visualization purposes.
      H = H.T
      X, Y = np.meshgrid(xedges, yedges)
      im = ax.pcolormesh(X, Y, H, cmap='rainbow')
      #ax.pcolormesh(xedges, yedges, H, cmap='rainbow')
      corr = pd.Series(xb).corr(pd.Series(xd-xb))
      ax.set_xlim([xbMin, xbMax])
      ax.set_ylim([xd_Minus_xbMin, xd_Minus_xbMax])
      if j==1:
          ax.set_xlabel(r'$x_b\, \mathrm{ [\mu m]}$',fontsize=16)
      if i==0:
          ax.set_ylabel(r'$\Delta\, \mathrm{ [\mu m]}$',fontsize=16)
      ax.set_title(r'$\tau_c=$' + f'{tau_cs[i]} [min]; ' + r' $\rho=$' + f'{str(round(corr, 2))}',fontsize=16)
      #plt.colorbar(im)
      ax.grid()


    fig.tight_layout()
plt.show()

##############
fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=False, figsize=(axx*2,ab*3))
for j,observationList, mechanismType in zip(np.arange(2), [sizerMechnismList, adderMechanismList], ['sizer', 'adder']):
    
    for i,observations in enumerate(observationList):
      ax = axs[i,j]
      xb, xd = observations[:,:,1].flatten(), observations[:,:,2].flatten() # observations: [f, xb, xd, T])
      xbMin, xbMax = np.quantile(xb, 5/100), np.quantile(xb, 95/100)
      xd_Minus_xbMin, xd_Minus_xbMax = np.quantile(xd-xb, 5/100), np.quantile(xd-xb, 95/100)
      H, xedges, yedges = np.histogram2d(xb, xd-xb, bins=(np.linspace(xbMin, xbMax, 30), np.linspace(xd_Minus_xbMin, xd_Minus_xbMax, 30)), density=True)#, density=True)
      # Histogram does not follow Cartesian convention (see Notes),
      # therefore transpose H for visualization purposes.
      H = H.T
      X, Y = np.meshgrid(xedges, yedges)
      im = ax.pcolormesh(X, Y, H, cmap='rainbow')
      #ax.pcolormesh(xedges, yedges, H, cmap='rainbow')
      corr = pd.Series(xb).corr(pd.Series(xd-xb))
      ax.set_xlim([xbMin, xbMax])
      ax.set_ylim([xd_Minus_xbMin, xd_Minus_xbMax])
      if i==2:
          ax.set_xlabel(r'$x_b\, \mathrm{ [\mu m]}$',fontsize=16)
      if j==0:
          ax.set_ylabel(r'$\Delta\, \mathrm{ [\mu m]}$',fontsize=16)
      ax.set_title(r'$\tau_c=$' + f'{tau_cs[i]} [min]; ' + r' $\rho=$' + f'{str(round(corr, 2))}',fontsize=16)
      #plt.colorbar(im)
      ax.grid()


    fig.tight_layout()
plt.show()