#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:24:45 2025

@author: ron.teichner
"""

import torch
from torch import nn
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchsde
from scipy import interpolate
from main_func import SDE_1D, plot, get_divisionTime, plot_cs_and_lineage, get_CR_str, calc_FTPD_wrapper, createLineages
from IPython import display
import pickle

plt.close('all')

mu_u, sigma_u = 2.4, 0.15357#1.0, 0.1 # [mu m]
tau_u = 40#[2, 40, 200] # [min]

xb_0_mu, xb_0_sigma = 0.5, np.sqrt(0.0025)
mu_eta = 0.5
sigma_eta = 0.05**2
gamma_shape, gamma_scale = 25, 9.4e-4

mechanismType = 'sizer' #'adder'

batch_size, state_size = 100000, 1
u0 = 0.63#mu_u + sigma_u*torch.randn(batch_size, state_size)
xb = 0.41#mu_u/2
alpha = 0.01835#gamma_shape*gamma_scale
tau_u = 192.2#40
query_T = 25
observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder = calc_FTPD_wrapper(u0, xb, alpha, mu_u, sigma_u, tau_u, batch_size, query_T, mechanismType, sdeRes='adaptive')

titleStr = ''
paramStr = r'$\mu_u=$'+f'{str(round(mu_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\sigma_u=$'+f'{str(round(sigma_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\tau_u=$'+f'{str(round(tau_u))}'+r'$\mathrm{min}$'
l = 0
plot_cs_and_lineage(ts[:int((observations_sizer[:,0,3].max()+5)/t_step)], u[:,l:l+3], observations_sizer[l:l+3], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType, FPTs=observations_sizer[:,:,3].flatten(), paramStr=paramStr, hist=hist_sizer, bin_edges=bin_edges_sizer, plotOnlyInputHist=True)

for l in range(u.shape[1]):
    plot_cs_and_lineage(ts[:int((observations_sizer[:,0,3].max()+5)/t_step)], u[:,l:l+1], observations_sizer[l:l+1], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr + ' ' + str(l), mechanismType=mechanismType, FPTs=observations_sizer[:,:,3].flatten(), paramStr=paramStr, hist=hist_sizer, bin_edges=bin_edges_sizer, plotOnlyInputHist=True)    


nGenerations = 25
observations_sizer, u, ts = createLineages(mu_eta, sigma_eta, gamma_shape, gamma_scale, mu_u, sigma_u, tau_u, nGenerations, batch_size, mechanismType)

for l in range(3):
    titleStr = f'lineage {l}; ' + mechanismType + r': $\tau_c=$' + f'{tau_u} [min]; ' + r'$\sigma(x_d)=$' + f'{str(round(observations_sizer[:,:,2].std(), 3))},' + r'$\sigma(\Delta)=$' + f'{str(round((observations_sizer[:,:,2]-observations_sizer[:,:,1]).std(), 3))}' + get_CR_str(observations_sizer)
    plot_cs_and_lineage(ts, u[:,l:l+1], observations_sizer[l:l+1], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType)





for tau_u in [40,200]:
    observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder = calc_FTPD_wrapper(mu_u, mu_u/2, gamma_shape*gamma_scale, mu_u, sigma_u, tau_u, batch_size, query_T)
    saveFileName = f'FPTD_tau_{str(round(tau_u))}.pt'
    #pickle.dump({'observations':observations_sizer, 'ts':ts, 't_step':t_step, 'u':u}, open(saveFileName, 'wb'))
    
    titleStr = ''
    paramStr = r'$\mu_u=$'+f'{str(round(mu_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\sigma_u=$'+f'{str(round(sigma_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\tau_u=$'+f'{str(round(tau_u))}'+r'$\mathrm{min}$'
    l = 0
    plot_cs_and_lineage(ts[:int((observations_sizer[:,0,3].max()+5)/t_step)], u[:,l:l+3], observations_sizer[l:l+3], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType, FPTs=observations_sizer[:,:,3].flatten(), paramStr=paramStr)

'''
plt.figure()  
for tau_u in [2,40,200]:
    saveFileName = f'FPTD_tau_{str(round(tau_u))}.pt'
    FPTDdict = pickle.load(open(saveFileName, 'rb'))
    observations, ts, t_step, u = FPTDdict['observations'], FPTDdict['ts'], FPTDdict['t_step'], FPTDdict['u']
    FPTs = observations[:,:,3].flatten()
    plt.hist(x=FPTs, bins=np.min([50,int(len(FPTs)/10)]),  density=True, histtype='step', linewidth=1, label=r'$\tau_u=$'+f'{str(round(tau_u))}')
    hist, bin_edges = np.histogram(FPTs, 100, density=True)
    (np.diff(bin_edges)*hist).sum() == 1
    # All but the last (righthand-most) bin is half-open. In other words, if bins is:
    # [1, 2, 3, 4]
    # then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.
plt.xlabel('min')
plt.grid()
plt.legend(loc='upper left', fontsize=14)
plt.savefig('FPTD_hist.png', dpi=150)
plt.close()




for tau_u in [2,40,200]:
    observations_sizer, ts, t_step, u, f_T_sizer, f_T_adder = calc_FTPD_wrapper(mu_u, mu_u/2, gamma_shape*gamma_scale, mu_u, sigma_u, tau_u, batch_size, query_T)
    saveFileName = f'FPTD_tau_{str(round(tau_u))}.pt'
    #pickle.dump({'observations':observations_sizer, 'ts':ts, 't_step':t_step, 'u':u}, open(saveFileName, 'wb'))
    
    titleStr = ''
    paramStr = r'$\mu_u=$'+f'{str(round(mu_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\sigma_u=$'+f'{str(round(sigma_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\tau_u=$'+f'{str(round(tau_u))}'+r'$\mathrm{min}$'
    l = 0
    plot_cs_and_lineage(ts[:int((observations[:,0,3].max()+5)/t_step)], u[:,l:l+4], observations[l:l+4], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType, FPTs=observations[:,:,3].flatten(), paramStr=paramStr)

tau_u = 40
saveFileName = f'FPTD_tau_{str(round(tau_u))}.pt'
FPTDdict = pickle.load(open(saveFileName, 'rb'))
observations, ts, t_step, u = FPTDdict['observations'], FPTDdict['ts'], FPTDdict['t_step'], FPTDdict['u']
FPTs = observations[:,:,3].flatten()

idx1 = np.argmin(np.abs(observations[:,:,3] - np.quantile(FPTs, 15/100)))
idx2 = np.argmin(np.abs(observations[:,:,3] - np.quantile(FPTs, 50/100)))
idx3 = np.argmin(np.abs(observations[:,:,3] - np.quantile(FPTs, 85/100)))

observations[0] = observations[idx1]
observations[1] = observations[idx2]
observations[2] = observations[idx3]

u[:,0] = u[:,idx1]
u[:,1] = u[:,idx2]
u[:,2] = u[:,idx3]


titleStr = ''
paramStr = r'$\mu_u=$'+f'{str(round(mu_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\sigma_u=$'+f'{str(round(sigma_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\tau_u=$'+f'{str(round(tau_u))}'+r'$\mathrm{min}$'
l = 0
filename='tau_40.png'
plot_cs_and_lineage(ts[:int((observations[:,0,3].max()+5)/t_step)], u[:,l:l+3], observations[l:l+3], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType, filename=filename)

plt.figure()  
for tau_u in [2,40,200]:
    saveFileName = f'FPTD_tau_{str(round(tau_u))}.pt'
    FPTDdict = pickle.load(open(saveFileName, 'rb'))
    observations, ts, t_step, u = FPTDdict['observations'], FPTDdict['ts'], FPTDdict['t_step'], FPTDdict['u']
    FPTs = observations[:,:,3].flatten()
    plt.hist(x=FPTs, bins=np.min([100,int(len(FPTs)/10)]),  density=True, histtype='step', linewidth=1, label=r'$\tau_u=$'+f'{str(round(tau_u))}')
plt.xlabel('min')
plt.grid()
plt.legend(loc='upper left', fontsize=14)
plt.savefig('FPTD_hist.png', dpi=150)
plt.close()
    
  

nGenerations = 1
nMinPerGeneration = 60.0
t_step, sim_duration = 1.0/60, nMinPerGeneration*nGenerations # [min]

sde = SDE_1D(mu_u, sigma_u, tau_u)
ts = np.arange(0, sim_duration, t_step) # [min]

sys.setrecursionlimit(5000)

with torch.no_grad():
    u = torchsde.sdeint(sde, u0, torch.from_numpy(ts), adaptive=False, dt=t_step).cpu().numpy()

    
nMin2Plot = 400
titleStr = r'$\tau_c=$' + f'{tau_u}; Empirical variance = {str(round(u.flatten().var(), 4))}; Expected variance = {str(round(sigma_u**2,4))}'
plot(ts[:int(nMin2Plot/t_step)], u[:,:3], xlabel='min', ylabel='$c\, \mathrm{[\mu m]}$', title=titleStr)






observations = np.zeros((batch_size, int(nGenerations), 4)) # eta, xb, xd, T
for l in range(batch_size):
    print(f'lineage {l}')
    single_cs_series = u[:,l,0]
    for g in range(int(nGenerations)):
        if g == 0:
            f = np.nan
            xbNegative = True
            while xbNegative:
                xb = xb_0_mu + xb_0_sigma*np.random.randn()
                xbNegative = xb <= 0 or xb >= single_cs_series[0]
            t_xb = 0.0
        else:
            f_above_1 = True
            while f_above_1:
                f = mu_eta + sigma_eta*np.random.randn()
                f_above_1 = np.abs(f) >= 1
            xb = observations[l, g-1, 2]*f # \mu m
            t_xb = observations[l, :g, 3].sum() # min
        t_xd, xd = get_divisionTime(xb, t_xb, single_cs_series, ts, gamma_shape, gamma_scale, mechanismType)
        observations[l, g] = np.array([f, xb, xd, t_xd - t_xb])
        
for l in range(3):
    titleStr = f'lineage {l}; ' + mechanismType + r': $\tau_c=$' + f'{tau_u} [min]; ' + r'$\sigma(x_d)=$' + f'{str(round(observations[:,:,2].std(), 3))},' + r'$\sigma(\Delta)=$' + f'{str(round((observations[:,:,2]-observations[:,:,1]).std(), 3))}' + get_CR_str(observations)
    plot_cs_and_lineage(ts, u[:,l:l+1], observations[l:l+1], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType)

l=0
titleStr = f'lineage {l}:{l+3}; ' + mechanismType + r': $\tau_c=$' + f'{tau_u} [min]; ' + r'$\sigma(x_d)=$' + f'{str(round(observations[:,:,2].std(), 3))},' + r'$\sigma(\Delta)=$' + f'{str(round((observations[:,:,2]-observations[:,:,1]).std(), 3))}' + get_CR_str(observations)
plot_cs_and_lineage(ts, u[:,l:l+3], observations[l:l+3], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType=mechanismType)

'''
