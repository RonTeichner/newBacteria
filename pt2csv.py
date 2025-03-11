#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:27:07 2025

@author: ronteichner
"""

import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from main_func import calc_logLikelihood, plot_cs_and_lineage

newResults = False

if newResults:
    new_results_nGnes_100_ = pickle.load(open('./new_results_nGnes_100_.pt', 'rb'))
    resultsList = [new_results_nGnes_100_]
    nGensList = [100]
    noiseless = [0]
else:
    new_results_nGnes_100_ = pickle.load(open('./new_results_nGnes_100_.pt', 'rb'))
    results_nGnes__noiseless_2_ = pickle.load(open('./results_nGnes__noiseless_2_.pt', 'rb'))
    results_nGnes__noiseless_100_ = pickle.load(open('./results_nGnes__noiseless_100_.pt', 'rb'))
    results_nGnes_1_ = pickle.load(open('./results_nGnes_1_.pt', 'rb'))
    results_nGnes_2_ = pickle.load(open('./results_nGnes_2_.pt', 'rb'))
    results_nGnes_20_ = pickle.load(open('./results_nGnes_20_.pt', 'rb'))
    results_nGnes_100_ = pickle.load(open('./results_nGnes_100_.pt', 'rb'))
    
    
    
    resultsList = [new_results_nGnes_100_, results_nGnes__noiseless_2_, results_nGnes__noiseless_100_, results_nGnes_1_, results_nGnes_2_, results_nGnes_20_, results_nGnes_100_]
    nGensList = [100, 2, 100, 1, 2, 20, 100]
    noiseless = [0, 1, 1, 0, 0, 0, 0]

features = ['sizer_gt', 'noiseless', 'nGens', 'lineage', 
                      'mu_u_gt', 'sigma_u_gt', 'tau_u_gt', 'nll_gt',
                      'mu_u_est_sizer', 'sigma_u_est_sizer', 'tau_u_est_sizer', 'nll_sizer',
                      'mu_u_est_adder', 'sigma_u_est_adder', 'tau_u_est_adder', 'nll_adder']


data = np.zeros((0,len(features)))
for results, nGens, noiseless in zip(resultsList, nGensList, noiseless):
    for res in results:
        if res['gtDict']['mechanismType'] == 'sizer':
            sizer_gt = 1
        else:
            sizer_gt = 0
        row = np.concatenate((np.array([sizer_gt]), np.array([noiseless]), np.array([nGens]), np.array([res['gtDict']['lineageIdx']]),
                        np.array([res['gtDict']['mu_u_gt']]),           np.array([res['gtDict']['sigma_u_gt']]),            np.array([res['gtDict']['tau_u']]),         np.array([res['gtDict']['nll']]),
                        np.array([res['sizer_est_dict']['mu_u_est']]),   np.array([res['sizer_est_dict']['sigma_u_est']]),    np.array([res['sizer_est_dict']['tau_u_est']]), np.array([res['sizer_est_dict']['nll']]),
                        np.array([res['adder_est_dict']['mu_u_est']]),   np.array([res['adder_est_dict']['sigma_u_est']]),    np.array([res['adder_est_dict']['tau_u_est']]), np.array([res['adder_est_dict']['nll']])))
        data = np.concatenate((data, row[None]), axis=0)

df = pd.DataFrame(columns=features, data=data)
if not newResults:
    df.to_csv('./simResults.csv')
else:
    df.to_csv('./simNewResults.csv')
    
###############
plt.figure()
filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 2)]
rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_sizer = (rs>0).sum()
nTotal_sizer = filteredDf.shape[0]
plt.plot(rs, 1*np.ones(filteredDf.shape[0]), 'b.', label=r'sizer, nGens=100, $\tau_u=2$')

filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 2)]
ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_adder = (ra<0).sum()
nTotal_adder = filteredDf.shape[0]
plt.plot(ra, 2*np.ones(filteredDf.shape[0]), 'b+',  label=r'adder, nGens=100, $\tau_u=2$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')
###########
filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 40)]
rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_sizer = (rs>0).sum()
nTotal_sizer = filteredDf.shape[0]
plt.plot(rs, 3*np.ones(filteredDf.shape[0]), 'g.', label=r'sizer, nGens=100, $\tau_u=40$')

filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 40)]
ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_adder = (ra<0).sum()
nTotal_adder = filteredDf.shape[0]
plt.plot(ra, 4*np.ones(filteredDf.shape[0]), 'g+',  label=r'adder, nGens=100, $\tau_u=40$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')

###########
filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 200)]
rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_sizer = (rs>0).sum()
nTotal_sizer = filteredDf.shape[0]
plt.plot(rs, 5*np.ones(filteredDf.shape[0]), 'r.', label=r'sizer, nGens=100, $\tau_u=200$')

filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 100) & (df['tau_u_gt'] == 200)]
ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
nCorrect_adder = (ra<0).sum()
nTotal_adder = filteredDf.shape[0]
plt.plot(ra, 6*np.ones(filteredDf.shape[0]), 'r+',  label=r'adder, nGens=100, $\tau_u=200$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')


plt.grid()
plt.xlabel('log(r)')
plt.ylim([-2,10])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

if not newResults:
    ###############
    plt.figure()
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 2)]
    rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_sizer = (rs>0).sum()
    nTotal_sizer = filteredDf.shape[0]
    plt.plot(rs, 1*np.ones(filteredDf.shape[0]), 'b.', label=r'sizer, nGens=20, $\tau_u=2$')

    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 2)]
    ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_adder = (ra<0).sum()
    nTotal_adder = filteredDf.shape[0]
    plt.plot(ra, 2*np.ones(filteredDf.shape[0]), 'b+',  label=r'adder, nGens=20, $\tau_u=2$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 40)]
    rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_sizer = (rs>0).sum()
    nTotal_sizer = filteredDf.shape[0]
    plt.plot(rs, 3*np.ones(filteredDf.shape[0]), 'g.', label=r'sizer, nGens=20, $\tau_u=40$')

    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 40)]
    ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_adder = (ra<0).sum()
    nTotal_adder = filteredDf.shape[0]
    plt.plot(ra, 4*np.ones(filteredDf.shape[0]), 'g+',  label=r'adder, nGens=20, $\tau_u=40$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')

    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 200)]
    rs = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_sizer = (rs>0).sum()
    nTotal_sizer = filteredDf.shape[0]
    plt.plot(rs, 5*np.ones(filteredDf.shape[0]), 'r.', label=r'sizer, nGens=20, $\tau_u=200$')

    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 20) & (df['tau_u_gt'] == 200)]
    ra = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    nCorrect_adder = (ra<0).sum()
    nTotal_adder = filteredDf.shape[0]
    plt.plot(ra, 6*np.ones(filteredDf.shape[0]), 'r+',  label=r'adder, nGens=20, $\tau_u=200$; ' + f'#lineages={nTotal_sizer+nTotal_adder}, %correct={str(round(100*(nCorrect_sizer+nCorrect_adder)/(nTotal_sizer+nTotal_adder)))}, performance={str(round(np.concatenate((-ra,rs)).mean()))}')


    
    plt.grid()
    plt.xlabel('log(r)')
    plt.ylim([-2,10])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    
    #################
    plt.figure()
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 1*np.ones(filteredDf.shape[0]), 'b.', label=r'sizer, nGens=2, $\tau_u=2$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 2*np.ones(filteredDf.shape[0]), 'b+',  label=r'adder, nGens=2, $\tau_u=2$')
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 3*np.ones(filteredDf.shape[0]), 'g.', label=r'sizer, nGens=2, $\tau_u=40$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 4*np.ones(filteredDf.shape[0]), 'g+',  label=r'adder, nGens=2, $\tau_u=40$')
    
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 5*np.ones(filteredDf.shape[0]), 'r.', label=r'sizer, nGens=2, $\tau_u=200$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 0) & (df['nGens'] == 2) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 6*np.ones(filteredDf.shape[0]), 'r+',  label=r'adder, nGens=2, $\tau_u=200$')
    
    
    plt.grid()
    plt.xlabel('log(r)')
    plt.ylim([-2,10])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    ###############
    plt.figure()
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 1*np.ones(filteredDf.shape[0]), 'b.', label=r'sizer, noiseless, nGens=100, $\tau_u=2$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 2*np.ones(filteredDf.shape[0]), 'b+',  label=r'adder, noiseless, nGens=100, $\tau_u=2$')
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 3*np.ones(filteredDf.shape[0]), 'g.', label=r'sizer, noiseless, nGens=100, $\tau_u=40$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 4*np.ones(filteredDf.shape[0]), 'g+',  label=r'adder, noiseless, nGens=100, $\tau_u=40$')
    
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 5*np.ones(filteredDf.shape[0]), 'r.', label=r'sizer, noiseless, nGens=100, $\tau_u=200$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 100) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 6*np.ones(filteredDf.shape[0]), 'r+',  label=r'adder, noiseless, nGens=100, $\tau_u=200$')
    
    
    plt.grid()
    plt.xlabel('log(r)')
    plt.ylim([-2,10])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    ###############
    plt.figure()
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 1*np.ones(filteredDf.shape[0]), 'b.', label=r'sizer, noiseless, nGens=2, $\tau_u=2$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 2)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 2*np.ones(filteredDf.shape[0]), 'b+',  label=r'adder, noiseless, nGens=2, $\tau_u=2$')
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 3*np.ones(filteredDf.shape[0]), 'g.', label=r'sizer, noiseless, nGens=2, $\tau_u=40$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 40)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 4*np.ones(filteredDf.shape[0]), 'g+',  label=r'adder, noiseless, nGens=2, $\tau_u=40$')
    
    ###########
    filteredDf = df[(df['sizer_gt'] == 1) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 5*np.ones(filteredDf.shape[0]), 'r.', label=r'sizer, noiseless, nGens=2, $\tau_u=200$')
    
    filteredDf = df[(df['sizer_gt'] == 0) & (df['noiseless'] == 1) & (df['nGens'] == 2) & (df['tau_u_gt'] == 200)]
    r = -(filteredDf['nll_sizer'] - filteredDf['nll_adder'])
    plt.plot(r, 6*np.ones(filteredDf.shape[0]), 'r+',  label=r'adder, noiseless, nGens=2, $\tau_u=200$')
    
    
    plt.grid()
    plt.xlabel('log(r)')
    plt.ylim([-2,10])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
#############################################################################################################################
if newResults:
    nGenerations = 100   
    nLineagesToPlot = 60
    noiseLess = False
    
    if noiseLess:
        noiseStr = '_noiseless_'
        noiseless = 1
    else:
        noiseStr = ''
        noiseless = 0
    
    
    
    for tau_u in [40,200,2]:
        for mechanismType in ['adder', 'sizer']:
            if mechanismType == 'sizer':
                sizer_gt = 1
            else:
                sizer_gt = 0
            #filename = mechanismType + '_tau_' + str(tau_u) + '_lineages.pt'
            filename = mechanismType + noiseStr + '_tau_' + str(tau_u) + '_lineages.pt'
            
            lineageDict = pickle.load(open(filename, 'rb'))
            observations, u, ts = lineageDict['observations'], lineageDict['u'], lineageDict['ts']
            for lineageIdx in range(observations.shape[0]):
                
                filteredDf = df[(df['lineage'] == lineageIdx) & (df['sizer_gt'] == sizer_gt) & (df['noiseless'] == noiseless) & (df['nGens'] == nGenerations) & (df['tau_u_gt'] == tau_u)]
                assert filteredDf.shape[0]<=1
                if filteredDf.shape[0]==1:
                    sizer_nlogLikelihood, sizer_likelihoods = calc_logLikelihood(np.array([filteredDf['mu_u_est_sizer'].to_numpy()[0], filteredDf['sigma_u_est_sizer'].to_numpy()[0], filteredDf['tau_u_est_sizer'].to_numpy()[0]]), observations[lineageIdx], 'sizer', returnLikelihoods=True)
                    adder_nlogLikelihood, adder_likelihoods = calc_logLikelihood(np.array([filteredDf['mu_u_est_adder'].to_numpy()[0], filteredDf['sigma_u_est_adder'].to_numpy()[0], filteredDf['tau_u_est_adder'].to_numpy()[0]]), observations[lineageIdx], 'adder', returnLikelihoods=True)
                    
                    sizerEstStr = r'sizer est: $\mu_u = $' + f'{str(round(filteredDf["mu_u_est_sizer"].to_numpy()[0],2))}, ' + r'$\sigma_u = $' + f'{str(round(filteredDf["sigma_u_est_sizer"].to_numpy()[0],4))}, ' + r'$\tau_u = $' + f'{str(round(filteredDf["tau_u_est_sizer"].to_numpy()[0],2))}, ll = {str(round(-sizer_nlogLikelihood))}'
                    adderEstStr = r'adder est: $\mu_u = $' + f'{str(round(filteredDf["mu_u_est_adder"].to_numpy()[0],2))}, ' + r'$\sigma_u = $' + f'{str(round(filteredDf["sigma_u_est_adder"].to_numpy()[0],4))}, ' + r'$\tau_u = $' + f'{str(round(filteredDf["tau_u_est_adder"].to_numpy()[0],2))}, ll = {str(round(-adder_nlogLikelihood))}'
                    
                    likelihoodDict = {'sizer_likelihoods': sizer_likelihoods, 'sizerEstStr': sizerEstStr, 'adder_likelihoods': adder_likelihoods, 'adderEstStr': adderEstStr, 'sizer_nlogLikelihood': sizer_nlogLikelihood, 'adder_nlogLikelihood': adder_nlogLikelihood}
                    
                    #plot_cs_and_lineage(ts, u[:,lineageIdx:lineageIdx+1], observations[lineageIdx:lineageIdx+1,:nLineagesToPlot], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=mechanismType + noiseStr + r' $\tau_u = $' + str(tau_u) + r', $\mu_u = $' + f'{str(round(filteredDf["mu_u_gt"].to_numpy()[0],2))}'+ r', $\sigma_u = $' + f'{str(round(filteredDf["sigma_u_gt"].to_numpy()[0],2))}' , mechanismType=mechanismType, likelihoodDict=likelihoodDict)
                    plot_cs_and_lineage(ts, u[:,lineageIdx:lineageIdx+1], observations[lineageIdx:lineageIdx+1,:nLineagesToPlot], xlabel='min', ylabel='$\mathrm{[\mu m]}$', title=mechanismType + noiseStr + r' $\tau_u = $' + str(tau_u) + r', $\mu_u = $' + f'{str(round(filteredDf["mu_u_gt"].to_numpy()[0],2))}'+ r', $\sigma_u = $' + f'{str(round(filteredDf["sigma_u_gt"].to_numpy()[0],2))}' , mechanismType=mechanismType)#, likelihoodDict=likelihoodDict)
                
            # observations: [f, xb, xd, T])
            
        
