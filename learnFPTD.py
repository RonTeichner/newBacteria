#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:49:34 2025

@author: ron.teichner
"""
import torch
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from main_func import SDE_1D, plot, get_divisionTime, plot_cs_and_lineage, get_CR_str, calc_FTPD_wrapper, createLineages, get_FTPD_bins, rand_FTPD, PyTorchMLP, FPTDDataset, ResBlock

enablePlots = False
enableTrain = True

for mechanismType in ['sizer','adder']:


    binDiff = torch.tensor(get_FTPD_bins(), dtype=torch.float).diff()[0]
    
    training_data = FPTDDataset(mechanismType, 'train')
    validation_data = FPTDDataset(mechanismType, 'validation')
    
    trainingData = torch.zeros((0,6))
    j=0
    bin_edges = get_FTPD_bins()
    binCenters = (bin_edges[:-1] + np.diff(bin_edges)/2)
    maxT_list = list()
    for paramsDict, hist in zip(training_data.params_list, training_data.hist_list):
        maxT_list.append(binCenters[np.argmax(hist > 0)])
        trainingData = torch.cat((trainingData, torch.cat((paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']))[None]), dim=0)
        if hist[0]*binDiff == 1:
            j+=1
    print(f'{str(round(j/len(training_data)*100,1))}% FPTDs with p(T<5min)=1 in training set')
    if enablePlots:
        plt.figure(), plt.hist(x=maxT_list, bins=int(len(maxT_list)/10),  density=True, histtype='step', cumulative=True, linewidth=1, label='maxT'), plt.legend(), plt.xlabel('min'), plt.grid(), plt.show()
    
    
    validationData = torch.zeros((0,6))
    j=0
    for paramsDict, hist in zip(validation_data.params_list, validation_data.hist_list):
        validationData = torch.cat((validationData, torch.cat((paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']))[None]), dim=0)
        if hist[0]*binDiff == 1:
            j+=1
    print(f'{str(round(j/len(validation_data)*100,1))}% FPTDs with p(T<5min)=1 in validation set')
    
    trainingDataMeans = trainingData.mean(dim=0)
    trainingDataStds = trainingData.std(dim=0)
    
    pickle.dump({'trainingDataMeans': trainingDataMeans, 'trainingDataStds': trainingDataStds}, open('trainStatistics_' + mechanismType + '_.pt', 'wb'))
    
    trainLoader = DataLoader(training_data, batch_size=16, shuffle=True)
    validationLoader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=True)
    
    model = PyTorchMLP(num_features=6, num_outputs=len(get_FTPD_bins())-1, means=trainingDataMeans, stds=trainingDataStds)
    
    binCenters = (model.bin_edges[:-1] + torch.diff(model.bin_edges)/2).detach().cpu().numpy()
        
    
    if enableTrain:
    
        
        
        lr = 0.001
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, momentum=0.9)#, weight_decay=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25)
        criterion = torch.nn.MSELoss()
        
        epoch = -1
        bestValLoss = np.inf
        while True:
            epoch += 1
            
            model = model.train()
            for batch_idx, (x, hist) in enumerate(trainLoader):
                optimizer.zero_grad()
                
                esthists = model(x)
                train_loss = criterion(esthists, hist)
                train_loss.backward()
                optimizer.step()
            
            
            trainNormalizedLoss = 0.5*(torch.diff(model.bin_edges)[None].expand(hist.shape[0],-1)*torch.abs(esthists-hist)).sum(dim=1).mean().item()
            
            assert torch.abs((torch.diff(model.bin_edges)*esthists[0]).sum() - 1) < 1e-3, f'train model sum to {(torch.diff(model.bin_edges)*esthists[0]).sum()}'
            assert torch.abs((torch.diff(model.bin_edges)*hist[0]).sum() - 1) < 1e-3, f'train gt sum to 1 {(torch.diff(model.bin_edges)*hist[0]).sum()}'
            
            if enablePlots: 
                plt.figure()
                plt.subplot(1,2,1)
                plt.plot(binCenters, esthists[0].detach().cpu().numpy(), label='model')
                plt.plot(binCenters, hist[0].detach().cpu().numpy(), linestyle='dashed', label='gt') 
                
                if (np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())==0).any():
                    start = binCenters[np.max([0, np.where(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())==0)[0][-1] -1])]
                else:
                    start = binCenters[np.argmin(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy()))]
                stop = binCenters[np.min([np.where(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())>=0.999)[0][0] + 3, len(binCenters)-1])]
            
	
                plt.xlim([start, stop])
                plt.xlabel('min')
                plt.legend(loc='upper left')
                plt.grid()
                plt.title(f'FPTD train epoch {epoch}')
            
            model.eval()
            for batch_idx, (x, hist) in enumerate(validationLoader):
                
                esthists = model(x)
                val_loss = criterion(esthists, hist)
                
            scheduler.step(val_loss)
            
            valNormalizedLoss = 0.5*(torch.diff(model.bin_edges)[None].expand(hist.shape[0],-1)*torch.abs(esthists-hist)).sum(dim=1).mean().item()
            
            if valNormalizedLoss < bestValLoss:
                bestValLoss = valNormalizedLoss
                #torch.save(model.state_dict(), f'./model_epoch_{str(epoch).zfill(6)}.pt')
                torch.save(model.state_dict(), './model_' + mechanismType + '_.pt')
    
            
            assert torch.abs((torch.diff(model.bin_edges)*esthists[0]).sum() - 1) < 1e-3, f'validation model sum to {(torch.diff(model.bin_edges)*esthists[0]).sum()}'
            assert torch.abs((torch.diff(model.bin_edges)*hist[0]).sum() - 1) < 1e-3, f'validation gt sum to 1 {(torch.diff(model.bin_edges)*hist[0]).sum()}' 
            
            if enablePlots: 
                plt.subplot(1,2,2)
                plt.plot(binCenters, esthists[0].detach().cpu().numpy(), label='model')
                plt.plot(binCenters, hist[0].detach().cpu().numpy(), linestyle='dashed', label='gt') 
                
                if (np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())==0).any(): 
                    start = binCenters[np.max([0, np.where(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())==0)[0][-1] -1])] 
                else: 
                    start = binCenters[np.argmin(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy()))] 
                stop = binCenters[np.min([np.where(np.cumsum((torch.diff(model.bin_edges)*hist[0]).numpy())>=0.999)[0][0] + 3, len(binCenters)-1])] 
                
                plt.xlim([start, stop])
                plt.xlabel('min')
                plt.legend(loc='upper left')
                plt.grid()
                plt.title('FPTD validation')
                plt.tight_layout()
                plt.show()
            print(mechanismType + f' epoch {epoch}: lr = {optimizer.param_groups[0]["lr"]}; trainLoss = {str(round(trainNormalizedLoss*100,2))}%, valLoss = {str(round(valNormalizedLoss*100,2))}%; best so far = {str(round(bestValLoss*100,2))}')
            
            if optimizer.param_groups[0]["lr"] < 1e-7:
                break
    else:
        model.load_state_dict(torch.load('./model_' + mechanismType + '_.pt', weights_only=True))
        model.eval()
        
        for batch_idx, (x, hist) in enumerate(validationLoader):
            
            esthists = model(x)
        x = x.detach().cpu().numpy()
        
        valNormalizedLoss = list()
        for i in range(esthists.shape[0]): 
            valNormalizedLoss.append(0.5*(torch.diff(model.bin_edges)[None].expand(hist[i:i+1].shape[0],-1)*torch.abs(esthists[i:i+1]-hist[i:i+1])).sum(dim=1).mean().item())
        
        plt.figure(), plt.hist(x=100*np.asarray(valNormalizedLoss), bins=int(len(valNormalizedLoss)/10),  density=True, histtype='step', cumulative=True, linewidth=1, label=''), plt.legend(), plt.xlabel('normalized error model '+mechanismType + ' [%]'), plt.grid(), plt.show()
        
        
    # paramsDict['u0'], paramsDict['barrier']['xb'], paramsDict['barrier']['alpha'], paramsDict['theta_u']['mu_u'], paramsDict['theta_u']['sigma_u'], paramsDict['theta_u']['tau_u']))[None]), dim=0)    
        for i in range(esthists.shape[0]): 
            if enablePlots: 
                plt.figure()
                plt.plot(binCenters, esthists[i].detach().cpu().numpy(), label='model')
                plt.plot(binCenters, hist[i].detach().cpu().numpy(), linestyle='dashed', label='gt') 
                if (np.cumsum((torch.diff(model.bin_edges)*hist[i]).numpy())==0).any(): 
                    start = binCenters[np.max([0, np.where(np.cumsum((torch.diff(model.bin_edges)*hist[i]).numpy())==0)[0][-1] -1])] 
                else: 
                    start = binCenters[np.argmin(np.cumsum((torch.diff(model.bin_edges)*hist[i]).numpy()))] 
                stop = binCenters[np.min([np.where(np.cumsum((torch.diff(model.bin_edges)*hist[i]).numpy())>=0.999)[0][0] + 3, len(binCenters)-1])]
            
                plt.xlim([start, stop])
            
                plt.xlabel('min')
                plt.legend(loc='upper left')
                plt.grid()
                plt.title('FPTD validation. ' + r'$u_0=$'+f'{str(round(x[i,0],2))}'+r'$\mu m, x_b=$'+f'{str(round(x[i,1],2))}'+r'$\mu m, \alpha=$'+f'{str(round(x[i,2],5))}'+r'$\mathrm{min}^{-1}, \mu_u=$'+f'{str(round(x[i,3],1))}'+r'$\mu m, \sigma_u=$'+f'{str(round(x[i,4],5))}'+r'$\mu m, \tau_u=$'+f'{str(round(x[i,5],1))}'+r'$\mathrm{min}$')
                plt.tight_layout()
                plt.show()
            
            
