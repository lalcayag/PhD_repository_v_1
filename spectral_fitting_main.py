# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:54:03 2019

@author: lalc
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import tkinter as tkint
import tkinter.filedialog
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from os import listdir
from os.path import isfile, join

import spectralfitting.spectralfitting as sf
import ppiscanprocess.spectra_construction as sc
import mann_model.mann as mn

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


import pickle
import re
import emcee
import corner
from scipy.stats import chi


# In[MCMC]
# In[Peltier fitting]
        
with open('S_ring.pkl', 'rb') as V_t:
     k1r1D_ave,Shr1D_ave = pickle.load(V_t)
bound_tot = [(500,1500),(0,1),(0,2), (0,2),(100,1000), (0,3), (0,2), (0,1),(60,68),(0,5)]
param_fix_tot = [1000,.1,.85,23/(2*np.pi)**2,200,0,1.6,.09/(2*np.pi)**2,60,2]
param_init_tot = param_fix_tot
lab_tot = ['$z_i$', '$w_*$', '$c_1^f$', '$c_2^f$', '$z$', '$u_*$', '$c_1^n$', '$c_2^n$', '$w$','$n$']
param_tot = np.zeros(10)
param_set = np.arange(10)
fig_letter = ['$(a)$','$(b)$','$(c)$','$(d)$','$(e)$','$(f)$']
fonts = matplotlib.font_manager.FontProperties(weight='black', size=24)
file_tot = ['z_i', 'w_star', 'c_1_f', 'c_2_f', 'z', 'u_star', 'c_1_n', 'c_2_n', 'Leq','N']

# Model 0
param_ind0 = [0,1,9]
param_ind_not0 = [e for e in param_set if e not in set(param_ind0)]
param_init0 = [param_init_tot[i] for i in param_ind0]
param_fix0 = [param_fix_tot[i] for i in param_ind_not0]
bound0 = [bound_tot[i] for i in param_ind0]
lab0 = [lab_tot[i] for i in param_ind0]
ndim0 = len(param_init0)
file0 = [file_tot[i] for i in param_ind0]
# Model 1
param_ind1 = [0,1,4,5,9]
param_ind_not1 = [e for e in param_set if e not in set(param_ind1)]
param_init1 = [param_init_tot[i] for i in param_ind1]
param_fix1 = [param_fix_tot[i] for i in param_ind_not1]
bound1 = [bound_tot[i] for i in param_ind1]
lab1 = [lab_tot[i] for i in param_ind1]
ndim1= len(param_init1)
file1 = [file_tot[i] for i in param_ind1]
scan_list = np.r_[np.arange(150,400),np.arange(650, 800)]
pmcmc_0 = []
res_0 = []
pmcmc_1 = []
res_1 = []
scan_i = 200
for scan_i in np.arange(0,600,10):
    print(scan_i)
    kb= 3*10**-2
    ku = np.mean(np.array(k1r1D_ave)[scan_i:scan_i+10,:],axis=0)
    ind = (ku<kb) & (ku>0)#k_1[scan_i]<kb
    F_obs = np.mean(np.array(Shr1D_ave)[scan_i:scan_i+10,:],axis=0)
    n_run = 100
    nwalkers = 1000   
    
    param_ind_not0 = [e for e in param_set if e not in set(param_ind0)]
    param_init0 = [param_init_tot[i] for i in param_ind0]
    param_fix0 = [param_fix_tot[i] for i in param_ind_not0]
    bound0 = [bound_tot[i] for i in param_ind0]
    lab0 = [lab_tot[i] for i in param_ind0]
    ndim0 = len(param_init0)
    file0 = [file_tot[i] for i in param_ind0]
    pos0 = [(np.array(bound0)[:,0] + np.diff(np.array(bound0),axis=1).T*np.random.rand(ndim0)).squeeze() for i in range(nwalkers)] 
    sampleri0 = emcee.EnsembleSampler(nwalkers, ndim0, LPST, a = 5, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],F_obs[ind],bound0),))
    sampleri0.run_mcmc(pos0, n_run)
    ind_accep0 = (sampleri0.acceptance_fraction<.5)&(sampleri0.acceptance_fraction>.2)
    pmcmc0 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampleri0.chain[ind_accep0 , int(n_run*.5):, :].\
                             reshape((-1, ndim0)), [16, 50, 84],axis=0)))
    pmcmc0 = np.array(list(pmcmc0))
    pmcmc_0.append(pmcmc0) 
    res0 = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix0,param_ind0,ku[ind],param_init = pmcmc0[:,0],bound= bound0)
    res_0.append(res0)
    
    param_ind_not1 = [e for e in param_set if e not in set(param_ind1)]
    param_init1 = [param_init_tot[i] for i in param_ind1]
    param_fix1 = [param_fix_tot[i] for i in param_ind_not1]
    bound1 = [bound_tot[i] for i in param_ind1]
    lab1 = [lab_tot[i] for i in param_ind1]
    ndim1 = len(param_init1)
    file1 = [file_tot[i] for i in param_ind1]
    pos1 = [(np.array(bound1)[:,0] + np.diff(np.array(bound1),axis=1).T*np.random.rand(ndim1)).squeeze() for i in range(nwalkers)] 
    sampleri1 = emcee.EnsembleSampler(nwalkers, ndim1, LPST, a = 5, args=((spectra_error,spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],F_obs[ind],bound1),))
    sampleri1.run_mcmc(pos1, n_run)
    ind_accep1 = (sampleri1.acceptance_fraction<.5)&(sampleri1.acceptance_fraction>.2)
    pmcmc1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(sampleri1.chain[ind_accep1, int(n_run*.7):, :].\
                             reshape((-1, ndim1)), [16, 50, 84],axis=0)))
    pmcmc1 = np.array(list(pmcmc1))
    pmcmc_1.append(pmcmc1) 
    res1 = spectra_fitting(F_obs[ind],spectra_peltier2,spectra_noise,param_fix1,param_ind1,ku[ind],param_init = pmcmc1[:,0],bound= bound1)
    res_1.append(res1)


prod0 = np.array([-1,1,-1])
fig, ax1 = plt.subplots()
prod0 = np.array([-1,1,-1])
ax1.set_xscale('log')
ax1.set_yscale('log')
#F_obs = 2*Shr1D_ave[scan_i]
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res0.x,args=(param_fix0,param_ind0,ku[ind],)),'-o') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0],args=(param_fix0,param_ind0,ku[ind],)),'-o') 
#ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0]+
#                prod0*pmcmc0[:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc0[:,0]-
#                prod0*pmcmc0[:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')
prod0 = np.array([-1,1,-1])
f, ax1 = plt.subplots()
prod0 = np.array([-1,1,-1])
ax1.set_xscale('log')
ax1.set_yscale('log')
#F_obs = 2*Shr1D_ave[scan_i]
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res1.x,args=(param_fix1,param_ind1,ku[ind],)),'-o') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(pmcmc1[:,0],args=(param_fix1,param_ind1,ku[ind],)),'-o') 
#ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0]+
#                prod0*pmcmc0[:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc0[:,0]-
#                prod0*pmcmc0[:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')
fig = corner.corner(sampleri0.chain[ind_accep0, int(n_run*.15):, :].reshape((-1, ndim0)),\
                        labels=lab0,\
                        quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        truths=res0.x,title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})

fig = corner.corner(sampleri1.chain[ind_accep1, int(n_run*.15):, :].reshape((-1, ndim1)),\
                        labels=lab1,\
                        quantiles=[0.16, 0.5, 0.84],show_titles=True,
                        truths=res1.x,title_kwargs={"fontsize": 18}, thick_kwargs={"fontsize": 18})


param_mean_0 = np.array([np.array(p)[:,0] for p in pmcmc_0])
param_conf1_0 = np.array([np.array(p)[:,1] for p in pmcmc_0])
param_conf2_0 = np.array([np.array(p)[:,2] for p in pmcmc_0])

param_mean_1 = np.array([np.array(p)[:,0] for p in pmcmc_1])
param_conf1_1 = np.array([np.array(p)[:,1] for p in pmcmc_1])
param_conf2_1 = np.array([np.array(p)[:,2] for p in pmcmc_1])

for paramind in range(ndim0):
    
    fig, ax = plt.subplots()
    ax.plot(param_mean_0[:,paramind],lw=4,color='k',label='Expected value')
    ax.plot(param_mean_0[:,paramind]-param_conf1_0[:,paramind],lw=2,color='grey')
    ax.plot(param_mean_0[:,paramind]+param_conf2_0[:,paramind],lw=2,color='grey')
    ax.fill_between(np.arange(0,len(param_mean_0[:,paramind])),param_mean_0[:,paramind]-param_conf1_0[:,paramind],
                    param_mean_0[:,paramind]+param_conf2_0[:,paramind], color="grey",alpha=0.5, edgecolor="")
    ax.set_ylabel(lab0[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')


for paramind in range(ndim1):
    
    fig, ax = plt.subplots()
    ax.plot(param_mean_1[:,paramind],lw=4,color='k',label='Expected value')
    ax.plot(param_mean_1[:,paramind]-param_conf1_1[:,paramind],lw=2,color='grey')
    ax.plot(param_mean_1[:,paramind]+param_conf2_1[:,paramind],lw=2,color='grey')
    ax.fill_between(np.arange(0,len(param_mean_1[:,paramind])),param_mean_1[:,paramind]-param_conf1_1[:,paramind],
                    param_mean_1[:,paramind]+param_conf2_1[:,paramind], color="grey",alpha=0.5, edgecolor="")
    ax.set_ylabel(lab0[paramind],fontsize=18)
    ax.text(0.05, 0.95, fig_letter[paramind], transform=ax.transAxes, fontproperties = fonts,verticalalignment='top')

prod0 = np.array([-1,1,-1])
f, ax1 = plt.subplots()
prod0 = np.array([-1,1,-1])
ax1.set_xscale('log')
ax1.set_yscale('log')
#F_obs = 2*Shr1D_ave[scan_i]
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res0.x,args=(param_fix0,param_ind0,ku[ind],)),'-o') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0],args=(param_fix0,param_ind0,ku[ind],)),'-o') 
#ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0]+
#                prod0*pmcmc0[:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc0[:,0]-
#                prod0*pmcmc0[:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')


prod0 = np.array([-1,1,-1])
f, ax1 = plt.subplots()
prod0 = np.array([-1,1,-1])
ax1.set_xscale('log')
ax1.set_yscale('log')
#F_obs = 2*Shr1D_ave[scan_i]
ax1.plot(ku[ind],ku[ind]*F_obs[ind])

ax1.plot(ku[ind],ku[ind]*spectra_peltier2(res1.x,args=(param_fix1,param_ind1,ku[ind],)),'-o') 
ax1.plot(ku[ind],ku[ind]*spectra_peltier2(pmcmc1[:,0],args=(param_fix1,param_ind1,ku[ind],)),'-o') 
#ax1.fill_between(ku[ind],ku[ind]*spectra_peltier2(pmcmc0[:,0]+
#                prod0*pmcmc0[:,1],args=(param_fix0,param_ind0,ku[ind],)),ku[ind]*spectra_peltier2(pmcmc0[:,0]-
#                prod0*pmcmc0[:,2],args=(param_fix0,param_ind0,ku[ind],)), color="grey",alpha=0.5, edgecolor="")
ax1.set_xlabel('$k$')
ax1.set_ylabel('$kF_{UU}(k)$')


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# In[Mann model spectral fitting]
root = tkint.Tk()
file_mann_2D= tkint.filedialog.askdirectory(parent=root,title='Choose an table-lookup Mann 2D')
root.destroy()


k1L = np.r_[0,10**np.arange(-3,3.1,.1)]
k2L = np.r_[0,10**np.arange(-3,3.1,.1)]
gamma = np.arange(0,5.1,.1)

k1L_g, k2L_g, gamma_g = np.meshgrid(k1L, k2L, gamma[30])

spec_ij = np.fromfile(file_mann_2D+'/US2DSpec.bin', dtype=np.float64)

spec_ij = np.reshape(spec_ij,(-1,6,62,62)).T

plt.figure()
plt.contourf(np.log(spec_ij[:,:,5,30]),cmap='jet')
plt.colorbar()

###############################################################################
###############################################################################  
L=100
k1 = np.r_[0,10**np.arange(-4,2.1,.1)]
k2 = np.r_[0,10**np.arange(-4,2.1,.1)]
k1L, k2L, specij = mn.spec_2D_mann_lookup(.025, L, 3, ks = (k1,k2), path = file_mann_2D) 
k_int_grd = np.meshgrid(k1L/L, k2L/L)

plt.figure()
plt.contourf(k_int_grd[0], k_int_grd[1],np.log(specij[:,:,0]),cmap='jet')
plt.colorbar()

sc.plot_log2D(k_int_grd, specij[:,:,0], label_S = "$\log_{10}{S}$", 
              C =10**-2,fig_num='a',nl=30, minS = -1.5)

Dp, rp, Ds, theta = 35, 45, 3500*2*np.pi/180, np.pi/2
k_dot =  k_int_grd[0]*np.cos(theta) + k_int_grd[1]*np.sin(theta)
k_cross = -k_int_grd[0]*np.sin(theta) + k_int_grd[1]*np.cos(theta)     
Filter = (np.exp(-(k_dot*rp/2)**2)*np.sinc(Dp*k_dot/(2*rp))*np.sinc(Ds*k_cross))**2

plt.figure()
plt.contourf(k_int_grd[0], k_int_grd[1],np.log(Filter),cmap='jet')
plt.colorbar()

plt.figure()
plt.plot(k_int_grd[0][51,:],Filter[:,51]**2)

sc.plot_log2D(k_int_grd, Filter*specij[:,:,0], label_S = "$\log_{10}{S}$", 
              C =10**-2,fig_num='a',nl=30, minS = -1.5)
####################################################################################
#Model Fitting




res0 = spectra_fitting(F_obs[ind],mann_model,param_fix0,param_ind0,ku[ind],param_init = pmcmc0[:,0],bound= bound0)

