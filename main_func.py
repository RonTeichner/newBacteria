#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:26:02 2025

@author: ron.teichner
"""

import torch
from torch import nn
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import torchsde
from scipy import interpolate
from scipy.optimize import minimize
import os.path
import pickle
from torch.utils.data import Dataset, DataLoader


class FPTDTestDataset(Dataset):
    def __init__(self, mechanismType):
        self.mechanismType = mechanismType
        self.batch_size = 100000
        
        self.datasetSize = 1024
        
        self.u0_lim = torch.asarray([0.2, 1.6])
        self.xb_lim = torch.asarray([0.2, 1.0])
        self.alpha_lim = torch.asarray([0.01, 0.04])
        self.mu_u_lim = torch.asarray([0.1, 2.5])
        self.sigma_u_lim = torch.asarray([0.01, 0.25])
        self.tau_u_lim = torch.asarray([1, 450])
        

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        
        params, hist = rand_FTPD_wrapper(self.u0_lim, self.xb_lim, self.alpha_lim, self.mu_u_lim, self.sigma_u_lim, self.tau_u_lim, self.batch_size, self.mechanismType)
        
        return params, torch.tensor(hist, dtype=torch.float)


class FPTDDataset(Dataset):
    def __init__(self, mechanismType, datasetType):
        self.mechanismType = mechanismType
        self.batch_size = 100000
        if datasetType == 'validation':
            self.val_filename = self.mechanismType + '_validationData.pt'
            self.datasetSize = 4096
        elif datasetType == 'train':
            self.val_filename = self.mechanismType + '_trainData.pt'
            self.datasetSize = int(4096*4)
        
        self.sdeRes = 'adaptive'
        self.u0_lim = torch.asarray([0.2, 1.6])
        self.xb_lim = torch.asarray([0.2, 1.0])
        self.alpha_lim = torch.asarray([0.01, 0.04])
        self.mu_u_lim = torch.asarray([0.1, 2.5])
        self.sigma_u_lim = torch.asarray([0.01, 0.25])
        self.tau_u_lim = torch.asarray([1, 450])
        self.maxT = 160 # min
        
        if not(os.path.isfile(self.val_filename)):
            params_list, hist_list = list(), list()
            for i in range(self.datasetSize):
                print(datasetType + f': staring {i} out of {self.datasetSize}')
                
                params, hist = rand_FTPD_wrapper(self.u0_lim, self.xb_lim, self.alpha_lim, self.mu_u_lim, self.sigma_u_lim, self.tau_u_lim, self.batch_size, self.mechanismType, sdeRes=self.sdeRes, maxT=self.maxT)
                
                params_list.append(params)
                hist_list.append(torch.tensor(hist, dtype=torch.float))
            pickle.dump({'params_list': params_list, 'hist_list': hist_list}, open(self.val_filename, 'wb'))
        dataDict = pickle.load(open(self.val_filename, 'rb'))
        self.params_list, self.hist_list = dataDict['params_list'], dataDict['hist_list']

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        paramsDict = self.params_list[idx]
        x = torch.cat((paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']))
        return x, self.hist_list[idx]

class ResBlock(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size,output_size)
        self.layer2 = torch.nn.Linear(input_size, output_size)
        self.shortcut = torch.nn.Sequential()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = torch.nn.ReLU()(self.layer1(input))
        input = torch.nn.ReLU()(self.layer2(input))
        input = input + shortcut
        return torch.nn.ReLU()(input)
    
class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_outputs, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds
        
        self.bin_edges = torch.tensor(get_FTPD_bins(), dtype=torch.float)
        self.bin_width = torch.diff(self.bin_edges)[0]
        self.resBlock = ResBlock
        self.intermediateWidth = int(num_outputs)
        
        self.t = torch.arange(0,60,5) # min
        #np.diff(bin_edges)*hist).sum()
        
        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features + 3*len(self.t), self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            self.resBlock(self.intermediateWidth, self.intermediateWidth),
            
            
            # output layer
            #torch.nn.Linear(self.intermediateWidth, 512),
            #torch.nn.ReLU(),
            #torch.nn.Linear(512, num_outputs),
            torch.nn.Linear(self.intermediateWidth, num_outputs),
            torch.nn.Softmax(dim=-1)
            #torch.nn.ReLU()
        )

    def forward(self, x):
        
        # x = torch.cat((paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']))
        batchSize = x.shape[0]
        u0, xb, alpha, mu_u, sigma_u, tau_u = x[:,0:1].expand(-1, len(self.t)), x[:,1:2].expand(-1, len(self.t)), x[:,2:3].expand(-1, len(self.t)), x[:,3:4].expand(-1, len(self.t)), x[:,4:5].expand(-1, len(self.t)), x[:,5:6].expand(-1, len(self.t))
        cellSize = xb * torch.exp(alpha*self.t.unsqueeze(0).expand(batchSize,-1))
        #logCellSize = torch.log(cellSize)
        addedSize = xb * torch.exp(alpha*self.t.unsqueeze(0).expand(batchSize,-1)) - xb
        
        noiselessThreshold = mu_u + (u0 - mu_u)*torch.exp(- self.t.unsqueeze(0).expand(batchSize,-1)/tau_u)
        
        x = (x - self.means[None].expand(x.shape[0],-1))/self.stds[None].expand(x.shape[0],-1)
        
        hists = self.all_layers(torch.cat((x, (cellSize-self.means[1])/self.stds[1], (addedSize-self.means[1])/self.stds[1], (noiselessThreshold-self.means[3])/self.stds[3]), dim=1)) / self.bin_width
        
        #hists = hists / (torch.diff(self.bin_edges)[None].expand(hists.shape[0],-1)*hists).sum(dim=1).unsqueeze(-1).expand(-1,hists.shape[1])
        return hists
    


class SDE_1D(nn.Module):

    def __init__(self, mu_c, sigma_c, tau_c):
        super().__init__()
        self.mu_c = nn.Parameter(torch.tensor(mu_c), requires_grad=False)
        self.sigma_c = nn.Parameter(torch.tensor(sigma_c), requires_grad=False)
        self.tau_c = nn.Parameter(torch.tensor(tau_c), requires_grad=False)
        self.WienerGain = nn.Parameter(torch.sqrt(2*torch.pow(self.sigma_c, 2)/self.tau_c), requires_grad=False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, c):
        return (self.mu_c - c)/self.tau_c

    def g(self, t, c):
        return self.WienerGain*torch.ones_like(c)


def plot_cs_and_lineage(ts, samples, observations, xlabel, ylabel, title='', mechanismType='sizer', FPTs=None, paramStr='', xlim=None, ylim=None, filename=None, hist=None, bin_edges=None, plotOnlyInputHist=False, likelihoodDict=None):
    samples = np.transpose(samples[:,:,0])
    colors = ['brown', 'magenta', 'purple']
    assert FPTs is None or likelihoodDict is None
    if not(likelihoodDict is None):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(30,4))
    else:
        plt.figure()#figsize=(20/1.5,5/1.5))
    if not(FPTs is None):
        plt.suptitle(paramStr)
        plt.subplot(1,2,1)
    for i, sample in enumerate(samples):
        singleLineageObservations = observations[i]# : [f, xb, xd, T])
        t_xb = np.concatenate((np.zeros((1,)), singleLineageObservations[:,3].cumsum()))[:-1] - singleLineageObservations[0,3]
        t_xd = singleLineageObservations[:,3].cumsum() - singleLineageObservations[0,3]
        xb = singleLineageObservations[:,1]
        xd = singleLineageObservations[:,2]
        tIndices = t_xb < ts[-1]
        t_xb, t_xd, xb, xd = t_xb[tIndices], t_xd[tIndices], xb[tIndices], xd[tIndices]
        if not(likelihoodDict is None):
            axs[0].plot(t_xb, xb, 'og', markersize=10, linewidth=6)
            axs[0].plot(t_xd, xd, 'or', markersize=10, linewidth=6)
        else:
            plt.plot(t_xb, xb, 'og', markersize=10, linewidth=6)
            plt.plot(t_xd, xd, 'or', markersize=10, linewidth=6)
        for t_xb_idx in range(len(t_xb)):
            T = t_xd[t_xb_idx] - t_xb[t_xb_idx]
            alpha = 1/T*np.log(xd[t_xb_idx]/xb[t_xb_idx])
            tVec = np.linspace(t_xb[t_xb_idx], t_xd[t_xb_idx], 10)
            xVals = xb[t_xb_idx]*np.exp(alpha*(tVec-tVec[0]))
            if not(likelihoodDict is None):
                axs[0].plot(tVec, xVals, '--k')
                
                axs[0].plot(tVec, xVals-xb[t_xb_idx], '--g')
            else:
                plt.plot(tVec, xVals, '--k')
                if mechanismType == 'adder':
                    plt.plot(tVec, xVals-xb[t_xb_idx], '--g')
        if samples.shape[0] > 1:
            if not(likelihoodDict is None):
                axs[0].plot(ts, sample[:len(ts)], color=colors[i], linewidth = 0.5, label=f'u {i}')
            else:
                plt.plot(ts, sample[:len(ts)], color=colors[i], linewidth = 0.5, label=f'u {i}')
        else:
            if not(likelihoodDict is None):
                axs[0].plot(ts, sample[:len(ts)], color=colors[i], linewidth = 0.5, label=f'u')
            else:
                plt.plot(ts, sample[:len(ts)], color=colors[i], linewidth = 0.5, label=f'u')
    if not(likelihoodDict is None):
        axs[0].set_title(title, fontsize=16)
        axs[0].set_xlabel(xlabel, fontsize=16)
        axs[0].set_ylabel(ylabel, fontsize=16)
        if len(t_xb) > 1:
            axs[0].set_xlim([t_xb[0], t_xb[-1]])
    else:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(t_xb) > 1:
            plt.xlim([t_xb[0], t_xb[-1]])
    #plt.legend()
    if not(likelihoodDict is None):
        enableLikelihood = False
        if not enableLikelihood:
            axs[1].plot(t_xb[1:], np.asarray(likelihoodDict['sizer_likelihoods'])[1:len(t_xb)], label=likelihoodDict['sizerEstStr'])
            axs[1].plot(t_xb[1:], np.asarray(likelihoodDict['adder_likelihoods'])[1:len(t_xb)], label=likelihoodDict['adderEstStr'])
            axs[1].plot(t_xb[1:], np.asarray(likelihoodDict['sizer_likelihoods'])[1:len(t_xb)]-np.asarray(likelihoodDict['adder_likelihoods'])[1:len(t_xb)], label='ll(sizer) - ll(adder);' + f'mean = {str(round((np.asarray(likelihoodDict["sizer_likelihoods"])-np.asarray(likelihoodDict["adder_likelihoods"])).mean(),3))}, median = {str(round((np.median(np.asarray(likelihoodDict["sizer_likelihoods"])-np.asarray(likelihoodDict["adder_likelihoods"]))),3))}')
            axs[1].axhline(y=0, color='k', linestyle='--')
            #axs[1].legend()
            
            axs[1].set_ylabel('loglikelihood')
        else:
            axs[1].plot(t_xb, np.exp(np.asarray(likelihoodDict['sizer_likelihoods'])[:len(t_xb)]), label=likelihoodDict['sizerEstStr'])
            axs[1].plot(t_xb, np.exp(np.asarray(likelihoodDict['adder_likelihoods'])[:len(t_xb)]), label=likelihoodDict['adderEstStr'])
            axs[1].plot(t_xb, np.exp(np.asarray(likelihoodDict['sizer_likelihoods'])[:len(t_xb)]-np.asarray(likelihoodDict['adder_likelihoods'])[:len(t_xb)]), label='l(sizer) / l(adder);' + f'mean = {str(round((np.exp(np.asarray(likelihoodDict["sizer_likelihoods"])-np.asarray(likelihoodDict["adder_likelihoods"]))).mean(),3))}')
            axs[1].axhline(y=0, color='k', linestyle='--')
            #axs[1].legend()
            
            axs[1].set_ylabel('likelihood')
        axs[1].legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    if not(FPTs is None):
        plt.subplot(1,2,2)
        if not plotOnlyInputHist:
            plt.hist(x=FPTs, bins=np.min([50,int(len(FPTs)/10)]),  density=True, histtype='step', linewidth=1, label=paramStr)
        #plt.legend()
        plt.grid()
        plt.xlabel('min')
        if not(xlim is None):
            plt.xlim=xlim
        if not(ylim is None):
            plt.ylim=ylim
            
        if not(hist is None):
            plt.plot(bin_edges[:-1] + np.diff(bin_edges)/2, hist)
    
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=150)
        plt.close()
    
def get_divisionTime(xb, t_xb, cs, ts, gamma_shape, gamma_scale, mechanismType, alpha=None):

  #alpha = 0.0235 + (0.0235*0.005)*np.random.randn()
  if alpha is None:
      alpha = np.random.gamma(gamma_shape, gamma_scale)

  x = xb*np.exp(alpha*(ts - t_xb))
  if mechanismType == 'sizer':
    xd_plus_idx = np.where(x >= cs)[0][0]
  elif mechanismType == 'adder':
    xd_plus_idx = np.where(x-xb >= cs)[0][0]
  xd_plus, xd_minus = x[xd_plus_idx], x[xd_plus_idx-1]
  cs_plus, cs_minus = cs[xd_plus_idx], cs[xd_plus_idx-1]
  ts_plus, ts_minus = ts[xd_plus_idx], ts[xd_plus_idx-1]

  ts_highRes = np.arange(ts_minus, ts_plus, 1e-3)
  x_highRes = xd_minus*np.exp(alpha*(ts_highRes-ts_minus))
  cs_highRes = (cs_plus-cs_minus)/(ts_plus-ts_minus)*(ts_highRes-ts_minus) + cs_minus
  t_xd = ts_highRes[np.argmin(np.abs(cs_highRes-x_highRes))]
  xd = xb*np.exp(alpha*(t_xd - t_xb))
    
  return t_xd, xd
    
def calc_FTPD_wrapper(u0, xb, alpha, mu_u, sigma_u, tau_u, batch_size, query_T=None, mechanismType=None, sdeRes='adaptive'):
    
    outputList = calc_FTPD(u0, xb, alpha, mu_u, sigma_u, tau_u, batch_size, query_T=query_T, mechanismType=mechanismType, sdeRes=sdeRes)
    
    while len(outputList) == 1:
        currentGenTime = outputList[0]
        outputList = calc_FTPD(u0, xb, alpha, mu_u, sigma_u, tau_u, batch_size, query_T=query_T, mechanismType=mechanismType, sdeRes=sdeRes, nMinPerGeneration=currentGenTime*1.5, noRecursion=True)
        if len(outputList) > 1:
            break
        
    
    observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder = outputList
    
    return observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder
    

def calc_FTPD(u0, xb, alpha, mu_u, sigma_u, tau_u, batch_size, query_T=None, mechanismType=None, sdeRes='adaptive', nMinPerGeneration=360, noRecursion=False):
    
    if not noRecursion:
        r_observations_sizer, r_observations_adder, _, _, _, _, _, _, _, _, _ = calc_FTPD(u0, xb, alpha, mu_u, sigma_u, tau_u, 250, query_T=query_T, mechanismType=mechanismType, sdeRes=sdeRes, nMinPerGeneration=360, noRecursion=True)
        if mechanismType == 'sizer':
            T_max = r_observations_sizer[:,0,3].max()
        elif mechanismType == 'adder':
            T_max = r_observations_adder[:,0,3].max()
        nMinPerGeneration = T_max*1.5
    
    nGenerations = 1
    t_step, sim_duration = 1.0/60*1, nMinPerGeneration*nGenerations # [min]

    sde = SDE_1D(mu_u, sigma_u, tau_u)
    ts = np.arange(0, sim_duration, t_step) # [min]
    t_xb = 0

    sys.setrecursionlimit(5000)
    
    if sdeRes == 'adaptive':
        adaptiveType = True
    elif sdeRes == 'constant':
        adaptiveType = False

    with torch.no_grad():
        u = torchsde.sdeint(sde, u0*torch.ones(batch_size, 1), torch.from_numpy(ts), adaptive=adaptiveType, dt=t_step)#.cpu().numpy()
    
    if mechanismType is None or mechanismType=='sizer':
        observations_sizer = np.zeros((batch_size, int(nGenerations), 4)) # eta, xb, xd, T
        observations_sizer[:,0,0] = np.nan
        observations_sizer[:,0,1] = xb
        
        x = torch.tensor(xb*np.exp(alpha*(ts - t_xb)))    
        indices_sizer = torch.argmax((x.unsqueeze(-1).unsqueeze(-1).expand(-1,u.shape[1],-1) > u).to(torch.long), dim=0)[:,0]
        
        if not (x.unsqueeze(-1).unsqueeze(-1).expand(-1,u.shape[1],-1) > u)[:,:,0].any(axis=0).all():
            return [nMinPerGeneration]
        
        observations_sizer[:,0,2] = x[indices_sizer]
        observations_sizer[:,0,3] = ts[indices_sizer]
    else:
        observations_sizer = None
    
    if mechanismType is None or mechanismType=='adder':
        observations_adder = np.zeros((batch_size, int(nGenerations), 4))
        observations_adder[:,0,0] = np.nan
        observations_adder[:,0,1] = xb
        
        Delta = torch.tensor(xb*np.exp(alpha*(ts - t_xb)) - xb)
        indices_adder = torch.argmax((Delta.unsqueeze(-1).unsqueeze(-1).expand(-1,u.shape[1],-1) > u).to(torch.long), dim=0)[:,0]
        
        if not (Delta.unsqueeze(-1).unsqueeze(-1).expand(-1,u.shape[1],-1) > u)[:,:,0].any(axis=0).all():
            return [nMinPerGeneration]
        
        observations_adder[:,0,2] = Delta[indices_adder] + xb
        observations_adder[:,0,3] = ts[indices_adder]
    else:
        observations_adder = None
    
    
    if False:
        titleStr = ''
        paramStr = r'$\mu_u=$'+f'{str(round(mu_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\sigma_u=$'+f'{str(round(sigma_u,2))}'+r'$\mathrm{\mu m}, $'+r'$\tau_u=$'+f'{str(round(tau_u))}'+r'$\mathrm{min}$'
        l = 0
        plot_cs_and_lineage(ts[:int((observations_sizer[:,0,3].max()+5)/t_step)], u[:,l:l+4], observations_sizer[l:l+4], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=titleStr, mechanismType='sizer', FPTs=observations_sizer[:,:,3].flatten(), paramStr=paramStr)
    
    
    if mechanismType is None or mechanismType=='sizer':
        FPTs = observations_sizer[:,:,3].flatten()        
        hist_sizer, bin_edges_sizer = np.histogram(FPTs, bins=get_FTPD_bins(), density=True)
        binStarts, binStops = bin_edges_sizer[:-1], bin_edges_sizer[1:] 
        
        if not(query_T is None):
            indices = np.logical_and(query_T >= binStarts, query_T < binStops)
            if indices.any():
                f_T_sizer = hist_sizer[indices][0]
            else:
                f_T_sizer = 0
        else:
            f_T_sizer = None
    else:
        f_T_sizer = None
        hist_sizer, bin_edges_sizer = None, None
    
    if mechanismType is None or mechanismType=='adder':
        FPTs = observations_adder[:,:,3].flatten()
        #dt = 0.5 # min        
        #((FPTs > query_T).sum()/len(FPTs) - (FPTs > (query_T+dt)).sum()/len(FPTs))/dt
        hist_adder, bin_edges_adder = np.histogram(FPTs, bins=get_FTPD_bins(), density=True)
        binStarts, binStops = bin_edges_adder[:-1], bin_edges_adder[1:] 
        if not(query_T is None):
            indices = np.logical_and(query_T >= binStarts, query_T < binStops)
            if indices.any():
                f_T_adder = hist_sizer[indices][0]
            else:
                f_T_adder = 0
        else:
            f_T_adder = None
    else:
        f_T_adder = None
        hist_adder, bin_edges_adder = None, None
    
    return [observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder]

def get_FTPD_bins():
    return np.arange(0, 360 + 0.5, 0.5)

def rand_FTPD_wrapper(u0_lim, xb_lim, alpha_lim, mu_u_lim, sigma_u_lim, tau_u_lim, batch_size, mechanismType, sdeRes='adaptive', maxT=np.inf):
    minT = maxT
    while minT >= maxT:
        params, hist, minT = rand_FTPD(u0_lim, xb_lim, alpha_lim, mu_u_lim, sigma_u_lim, tau_u_lim, batch_size, mechanismType, sdeRes)
    
    return params, hist

def rand_FTPD(u0_lim, xb_lim, alpha_lim, mu_u_lim, sigma_u_lim, tau_u_lim, batch_size, mechanismType, sdeRes):
    
    u0 = u0_lim[0] + torch.rand(1)*u0_lim.diff()
    alpha = alpha_lim[0] + torch.rand(1)*alpha_lim.diff()
    mu_u = mu_u_lim[0] + torch.rand(1)*mu_u_lim.diff()
    sigma_u = sigma_u_lim[0] + torch.rand(1)*sigma_u_lim.diff()
    tau_u = tau_u_lim[0] + torch.rand(1)*tau_u_lim.diff()
    
    if mechanismType == 'sizer':
        previous_xd = u0
        xb = previous_xd*0.3 + torch.rand(1)*(previous_xd*0.7-previous_xd*0.3)
    elif mechanismType == 'adder':
        previous_added_size = u0
        previous_xd_min = xb_lim[0] + previous_added_size
        previous_xd_max = xb_lim[1] + previous_added_size
        xb = previous_xd_min + torch.rand(1)*(previous_xd_max - previous_xd_min)
    
    params = {'u0': u0, 'barrier': {'xb': xb, 'alpha': alpha, 'mechanismType': mechanismType}, 'theta_u': {'mu_u': mu_u, 'sigma_u': sigma_u, 'tau_u': tau_u}}
    observations_sizer, observations_adder, ts, t_step, u, hist_sizer, bin_edges_sizer, hist_adder, bin_edges_adder, f_T_sizer, f_T_adder = calc_FTPD_wrapper(u0.item(), xb.item(), alpha.item(), mu_u.item(), sigma_u.item(), tau_u.item(), batch_size, mechanismType=mechanismType, sdeRes=sdeRes)
    
    if mechanismType == 'sizer':
        hist = hist_sizer
        bin_edges = bin_edges_sizer
        minT = observations_sizer[:,:,3].min()
    elif mechanismType == 'adder':
        hist = hist_adder
        bin_edges = bin_edges_adder
        minT = observations_adder[:,:,3].min()
        
    assert np.abs((np.diff(bin_edges)*hist).sum() - 1) < 1e-3, f'hist.sum()-1 = {(np.diff(bin_edges)*hist).sum()-1}'
    
    return params, hist, minT
    
    
    

    
def createLineages(mu_eta, sigma_eta, gamma_shape, gamma_scale, mu_u, sigma_u, tau_u, nGenerations, batch_size, mechanismType, noiseLess=False, sdeRes='adaptive'):
        nMinPerGeneration = 60.0
        t_step, sim_duration = 1.0/60*1, nMinPerGeneration*nGenerations # [min]

        sde = SDE_1D(mu_u, sigma_u, tau_u)
        ts = np.arange(0, sim_duration, t_step) # [min]
        
        u0 = mu_u + sigma_u*torch.randn(batch_size, 1)
        while (u0 <= 0).any():
            u0 = mu_u + sigma_u*torch.randn(batch_size, 1)
            

        sys.setrecursionlimit(5000)
        
        if sdeRes == 'adaptive':
            adaptiveType = True
        elif sdeRes == 'constant':
            adaptiveType = False

        with torch.no_grad():
            u = torchsde.sdeint(sde, u0*torch.ones(batch_size, 1), torch.from_numpy(ts), adaptive=adaptiveType, dt=t_step).cpu().numpy()
            while (u <= 0).any():
                u = torchsde.sdeint(sde, u0*torch.ones(batch_size, 1), torch.from_numpy(ts), adaptive=adaptiveType, dt=t_step).cpu().numpy()
        
        observations = np.zeros((batch_size, int(nGenerations), 4)) # eta, xb, xd, T
        for l in range(batch_size):
            print(f'lineage {l}')
            single_cs_series = u[:,l,0]
            u0 = single_cs_series[0]
            for g in range(int(nGenerations)):
                if g == 0:
                    f = mu_eta
                    
                    if mechanismType == 'sizer':
                        xd = u0
                        xb = 0.5*u0
                    elif mechanismType == 'adder':
                        xd = 2*u0
                        xb = u0
                    
                    t_xd = 0.0
                    T = np.log(xd/xb)/(gamma_shape*gamma_scale)
                    t_xb = t_xd - T
                    initBacteria_t_xb = t_xb
                else:
                    f_above_1 = True
                    while f_above_1:
                        if noiseLess:
                            f = mu_eta
                        else:
                            f = mu_eta + sigma_eta*np.random.randn()
                        f_above_1 = f >= 1 or f <= 0
                    xb = observations[l, g-1, 2]*f # \mu m
                    t_xb = initBacteria_t_xb + observations[l, :g, 3].sum() # min
                    if noiseLess:
                        t_xd, xd = get_divisionTime(xb, t_xb, single_cs_series, ts, gamma_shape, gamma_scale, mechanismType, alpha=gamma_shape*gamma_scale)
                    else:
                        t_xd, xd = get_divisionTime(xb, t_xb, single_cs_series, ts, gamma_shape, gamma_scale, mechanismType)
                observations[l, g] = np.array([f, xb, xd, t_xd - t_xb])
        return observations, u, ts
    
def get_CR_str(observations):
  xb, xd = observations[:,:,1], observations[:,:,2]
  Delta = observations[:,:,2] - observations[:,:,1]
  T = observations[:,:,3]

  observations_alpha = observations.copy()
  observations_alpha[:,:,2] = 1/T*np.log(xd/xb)


  shuffledObservations_alpha = observations_alpha[:,:,1:].copy()
  for p in range(shuffledObservations_alpha.shape[0]):
    for t in range(shuffledObservations_alpha.shape[1]):
      for f in range(shuffledObservations_alpha.shape[2]):
        shuffledObservations_alpha[p,t,f] = shuffledObservations_alpha[np.random.randint(shuffledObservations_alpha.shape[0]), np.random.randint(shuffledObservations_alpha.shape[1]), f]

  xb_shuffled, alpha_shuffled, T_shuffled = shuffledObservations_alpha[:,:,0], shuffledObservations_alpha[:,:,1], shuffledObservations_alpha[:,:,2]
  xd_shuffled = xb_shuffled*np.exp(alpha_shuffled*T_shuffled)
  Delta_shuffled = xd_shuffled - xb_shuffled

  CR_xd = xd.std()/xd_shuffled.std()
  CR_Delta = Delta.std()/Delta_shuffled.std()

  CR_str = r'; $CR(x_d)=$' + f'{str(round(CR_xd, 3))}' + r', $CR(\Delta)=$' + f'{str(round(CR_Delta, 3))}; '
  return CR_str

def ml_optimization(lineage, mechanismType, bounds):
    
    N = int(2**5)
    best_nll = np.inf
    mu_us = np.arange(bounds[0][0], bounds[0][1], 0.1)
    for i,mu_u in enumerate(mu_us):
        print(f'{i}/{len(mu_us)}', end=" ", flush=True)
        for sigma_u in np.arange(bounds[1][0], bounds[1][1], 0.01):
            #print(f'sigma_u = {sigma_u}', end=" ")
            for tau_u in np.arange(bounds[2][0], bounds[2][1], 5):
                nll = calc_logLikelihood(np.array([mu_u, sigma_u, tau_u]), lineage, mechanismType)
                if nll < best_nll:
                    best_nll = nll
                    initialGuess = np.array([mu_u, sigma_u, tau_u])
                    #print(f'mu_u = {str(round(mu_u, 4))}, sigma_u = {str(round(sigma_u, 4))}, tau_u = {str(round(tau_u, 4))}. nll = {str(round(nll,3))}')
                
    res = minimize(calc_logLikelihood, initialGuess, bounds=bounds, args=(lineage, mechanismType), options={'disp': False})            

    
    mu_u_est, sigma_u_est, tau_u_est = res.x
    nll = res.fun
    
    
    return mu_u_est, sigma_u_est, tau_u_est, nll

def calc_logLikelihood(x, lineage, mechanismType, batchProcessing=True, returnLikelihoods=False):
    
    # lineage: [f, xb, xd, T])
    mu_u, sigma_u, tau_u = x
    
    
    trainStatisticsDict = pickle.load(open('trainStatistics_' + mechanismType + '_.pt', 'rb'))
    trainingDataMeans, trainingDataStds = trainStatisticsDict['trainingDataMeans'], trainStatisticsDict['trainingDataStds']
    
    model = PyTorchMLP(num_features=6, num_outputs=len(get_FTPD_bins())-1, means=trainingDataMeans, stds=trainingDataStds)
    model.load_state_dict(torch.load('./model_' + mechanismType + '_.pt', weights_only=True))
    model.eval()
    
    f, xb, xd, T = lineage[:,0], lineage[:,1], lineage[:,2], lineage[:,3]
    
    if mechanismType == 'sizer':
        logNormalInitialCond = -np.log(sigma_u * np.sqrt(2*np.pi)) - 0.5*np.power((xd[0] - mu_u)/sigma_u, 2)
    else:
        logNormalInitialCond = -np.log(sigma_u * np.sqrt(2*np.pi)) - 0.5*np.power(((xd[0]-xb[0]) - mu_u)/sigma_u, 2)
    
    paramsDict = dict()
    paramsDict['theta_u'] = dict()
    paramsDict['barrier'] = dict()
    paramsDict['theta_u']['mu_u'] = torch.tensor(mu_u, dtype=torch.float)[None][None]
    paramsDict['theta_u']['sigma_u'] = torch.tensor(sigma_u, dtype=torch.float)[None][None]
    paramsDict['theta_u']['tau_u'] = torch.tensor(tau_u, dtype=torch.float)[None][None]
    
    bin_edges = get_FTPD_bins()
    binStarts, binStops = bin_edges[:-1], bin_edges[1:] 
    likelihoodList = list()
    if not batchProcessing:
        
        
        logLikelihood = logNormalInitialCond
        likelihoodList.append(logNormalInitialCond)
        for k in range(1, lineage.shape[0]):
            
            if mechanismType == 'sizer':
                paramsDict['u0'] = torch.tensor(xd[k-1], dtype=torch.float)[None][None]
            else:
                paramsDict['u0'] = torch.tensor(xd[k-1] - xb[k-1], dtype=torch.float)[None][None]
            
            paramsDict['barrier']['xb'] = torch.tensor(xb[k], dtype=torch.float)[None][None]
            paramsDict['barrier']['alpha'] = torch.tensor(1/T[k]*np.log(xd[k]/xb[k]), dtype=torch.float)[None][None]
            
            x = torch.cat((paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']), dim=1)
            hists = model(x)[0].detach().cpu().numpy()
            
            query_T = T[k]
            indices = np.logical_and(query_T >= binStarts, query_T < binStops)
            if indices.any():
                f_T = hists[indices][0]
                if f_T > 0:
                    log_f_T = np.log(f_T)
                else:
                    likelihoodList.append(np.inf)
                    if returnLikelihoods:
                        return np.inf, likelihoodList
                    else:
                        return np.inf
            else:
                likelihoodList.append(np.inf)
                if returnLikelihoods:
                    return np.inf, likelihoodList
                else:
                    return np.inf
            logLikelihood += log_f_T
            likelihoodList.append(log_f_T)
    else:
            
    
        # Prepare the tensors for batch processing
        u0 = torch.tensor([xd[k-1] if mechanismType == 'sizer' else xd[k-1] - xb[k-1] for k in range(1, lineage.shape[0])], dtype=torch.float).unsqueeze(1)
        xb_batch = torch.tensor(xb[1:], dtype=torch.float).unsqueeze(1)
        alpha_batch = torch.tensor([1/T[k]*np.log(xd[k]/xb[k]) for k in range(1, lineage.shape[0])], dtype=torch.float).unsqueeze(1)
        
        # Concatenate the tensors
        x_batch = torch.cat((u0, xb_batch, alpha_batch, paramsDict['theta_u']['mu_u'].expand(u0.size(0), -1), paramsDict['theta_u']['sigma_u'].expand(u0.size(0), -1), paramsDict['theta_u']['tau_u'].expand(u0.size(0), -1)), dim=1)
        
        # Process the batch through the model
        hists_batch = model(x_batch).detach().cpu().numpy()
        
        # Compute the log likelihood for each element in the batch
        logLikelihood = logNormalInitialCond
        likelihoodList.append(logNormalInitialCond)
        for k in range(1, lineage.shape[0]):
            query_T = T[k]
            indices = np.logical_and(query_T >= binStarts, query_T < binStops)
            if indices.any():
                f_T = hists_batch[k-1][indices][0]
                if f_T > 0:
                    log_f_T = np.log(f_T)
                else:
                    likelihoodList.append(np.inf)
                    if returnLikelihoods:
                        return np.inf, likelihoodList
                    else:
                        return np.inf
            else:
                likelihoodList.append(np.inf)
                if returnLikelihoods:
                    return np.inf, likelihoodList
                else:
                    return np.inf
            logLikelihood += log_f_T
            likelihoodList.append(log_f_T)
    
    
    nll = -logLikelihood    
    #print(f'mu_u = {str(round(mu_u, 4))}, sigma_u = {str(round(sigma_u, 4))}, tau_u = {str(round(tau_u, 4))}. nll = {str(round(nll,3))}')
    if returnLikelihoods:
        return nll, likelihoodList
    else:
        return nll

    
def plot(ts, samples, xlabel, ylabel, title=''):
    samples = np.transpose(samples[:,:,0])
    plt.figure(figsize=(20/1.5,5/1.5))
    for i, sample in enumerate(samples):
        plt.plot(ts, sample[:len(ts)], linewidth = 0.5, label=f'c {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
