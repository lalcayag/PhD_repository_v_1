# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:27:46 2018

Module for spectral fitting based on:
    
    - Least-Squares with specific weights
    - Maximum Likelihood estimation
    - Markov-Chain Monte-Carlo

Autocorrelation is calculated in terms 

@author: lalc
"""
# In[Packages used]

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import os, sys
import re
import emcee
import corner
from scipy.stats import chi
from scipy.interpolate import RegularGridInterpolator
import mann_model.mann as mn

#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer
                           
#from sklearn.neighbors import KDTree
# In[Log-Likelihood, Log-prior, Log-posterior]
def LLH(param,args=()):
    #args = (spectra_error,F_model,param_fix,param_ind,k,path,F_obs)
    print(-1*sum(np.log(args[0](param,args=(args[1:])))))
    return -1*sum(np.log(args[0](param,args=(args[1:]))))
# log-Prior
def LPR(param,args=()):
    #un-informative, first try, uniform (are the parameters independent?)
    #args = bounds
        #un-informative, first try, uniform (are the parameters independent?)   
    bound = np.array(args)   
    #print(bound)     
    ind_0 = param > bound[:,0]
    ind_1 = param < bound[:,1]
    lpr = np.sum(np.log(1/np.diff(np.array(bound),axis=1)))      
    if (np.sum(ind_0)==len(ind_0)) & (np.sum(ind_1)==len(ind_1)):
        return lpr
    else:
        return -np.inf         
# log-Posterior
def LPST(param,args=()):
    #normal:
    #args = (fun,x,bounds) 
    lpr = LPR(param,args=args[-1])
    if not np.isfinite(lpr):
        return -np.inf
    else:
        return lpr - LLH(param, args=args[:-1])
    
# In[Filter]
def filter_H(param,args=()): 
    #param = [c1,c2,l,s]
    #args = (k)
    w,n= param  
    k_1 = args[0]
    return 1/(1+(k_1*w)**n)     
    
# In[Turbulence Spectra models]

#def spectra_peltier(param,args=()): 
#    # Peltier 1996, Horizontal wind speed spectrum model
#    #param = [c1,c2,l,s]
#    #args = (k)
#    # mixture of free conection and neutral conditions
#    c1_f,c2_f,l_f,s_f, c1_n,c2_n,l_n,s_n,w,n= param 
#    
#    k_1 = args[0]
#    E = lambda c1,c2,l,s,k: .71*c1*l**2*k*s**2/(c2+(k*l)**2)**(4/3)
#    F = lambda c1,c2,l,s,k: .71*c1*l*s**2/((c2+(k*l)**2)**(5/6))
#    E_m = E(c1_f,c2_f,l_f,s_f,k_1)+E(c1_n,c2_n,l_n,s_n,k_1)
#    F_m = F(c1_f,c2_f,l_f,s_f,k_1)+F(c1_n,c2_n,l_n,s_n,k_1)
#    
#    #H = (np.sin(k_1*(w*np.pi)/np.max(k_1))/(k_1*(w*np.pi)/np.max(k_1)))**n
#    H = 1/(1+(k_1*w)**n)
#    
#    #plt.plot(k_1,H)
#    
#    F_m = F_m*H
#    return F_m

# In[The one used here]
def spectra_peltier2(param,args=()): 
    # Peltier 1996, Horizontal wind speed spectrum model
    #args = (param_fix,param_ind,k)
    # mixture of free conection and neutral conditions    
    #print(args[0])
    param_fix = args[0]
    param_ind = args[1]
    k_1 = args[2]
    param_tot = np.zeros(10)
    param_set = np.arange(10)
    param_ind_not = [e for e in param_set if e not in set(param_ind)]
    param_tot[param_ind] = param
    param_tot[param_ind_not] = param_fix  
    l_f, s_f, c1_f, c2_f, l_n, s_n, c1_n, c2_n, w, n= param_tot 
    F = lambda l,s,c1,c2,k: .71*c1*l*s**2/((c2+(k*l/2/np.pi)**2)**(5/6))  
    F_m = F(l_f,s_f,c1_f,c2_f,k_1)+F(l_n,s_n,c1_n,c2_n,k_1)
    H = 1/(1+(k_1*w)**n)
    #print([param])
    F_m = F_m*H
    return F_m

# In[]
#def spectra_peltier3(param,args=()): 
#    # Peltier 1996, Horizontal wind speed spectrum model
#    #param = [c1,c2,l,s]
#    #args = (k)
#    # mixture of free conection and neutral conditions
#    l_f,L,s_n,c2_n,w,n= param 
#    
#    if np.abs(L) > 0.0:
#        z_L = (-l_f/L)
#        z_L = np.sign(z_L)*np.abs(z_L)**(1/3)
#        s_f = .7*s_n*z_L
#        #print('Value', s_f, s_n, l_f, L, z_L)
#    else:
#        s_f = 0       
#    c1_f, c2_f, c1_n,l_n = .85, 23, 1.6, 200    
#    k_1 = args[0]
#    E = lambda c1,c2,l,s,k: .71*c1*l**2*k*s**2/(c2+(k*l)**2)**(4/3)
#    F = lambda c1,c2,l,s,k: .71*c1*l*s**2/((c2+(k*l)**2)**(5/6))
#    E_m = E(c1_f,c2_f,l_f,s_f,k_1)+E(c1_n,c2_n,l_n,s_n,k_1)
#    F_m = F(c1_f,c2_f,l_f,s_f,k_1)+F(c1_n,c2_n,l_n,s_n,k_1)
#    H = 1/(1+(k_1*w)**n)   
#    F_m = F_m*H
#    return F_m       

# In[Theoretical Spectra]
def spectra_noise(param,args=()):
    #args = (k)
#    return param[0]*args[0]**param[1]
    return np.exp(param[0]*args[0]**param[1])

# In[Noise in spectra]
def spectra_theo(param,args=()):
    #args = (F_model,param_fix,param_ind,k,path)
    param_model = param
    F_model = args[0](param_model,args= args[1:])
    return (F_model)

# In[Error distribution: Chi squared]

#def spectra_error(param,args=()):
#    #args = (F_model,F_noise,k,F_obs)
#    args_theo = args[:-1]
#    F_obs = args[-1]
#    k_1 = args[-2]
#    #print(param)
#    param_chi=param[-1]
#    param_theo = param[:-1]
#    F_theo = spectra_theo(param_theo,args=args_theo)
#    #Chi-squared error
#    error = param_chi*F_obs/F_theo
#    return chi.pdf(error, param_chi) 

# In[Error distribution: exponential!!]
def spectra_error(param,args=()):
    #args = (F_model,param_fix,param_ind,k,path,F_obs)
    args_theo = args[:-1]
    F_obs = args[-1]
    param_theo = param
    F_theo = spectra_theo(param_theo,args = args_theo)
    if len(F_obs.shape)<2:
        err = np.exp(-F_obs/F_theo)/F_theo
    elif len(F_obs.shape)==2:
        err = []
        for i in range(F_obs.shape[0]):
            e = (np.exp(-F_obs[i,:]/F_theo[i,:])/F_theo[i,:]).flatten()
            err = np.hstack((err,e))
    else:
        err = []
        for i in range(F_obs.shape[-1]):
            e = (np.exp(-F_obs[:,:,i]/F_theo[:,:,i])/F_theo[:,:,i]).flatten()
            err = np.hstack((err,e))
    return err
   
# In[Spectral fitting]
    
def spectra_fitting(F_obs,model,param_fix,param_ind,k_1,param_init = [],bound= []):
    print(param_init)
    res = sp.optimize.minimize(LLH, param_init, args=((spectra_error,model,param_fix,param_ind,k_1,F_obs),),method='L-BFGS-B', bounds = bound)#,callback=callbackF, options={'disp': True})
    return res
    
# In[]    
def callbackF(Xi):
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f} {4: 3.6f}'.format(Xi[2], Xi[3], Xi[6], Xi[7], Xi[8]))  
    
# In[convergence]
def get_convergence_statistic(i, sampler_chain, nwalkers, convergence_length=10,
                              convergence_period=10):
    s = sampler_chain[:, i-convergence_length+1:i+1, :]
    within_std = np.mean(np.var(s, axis=1), axis=0)
    per_walker_mean = np.mean(s, axis=1)
    mean = np.mean(per_walker_mean, axis=0)
    between_std = np.sqrt(np.mean((per_walker_mean-mean)**2, axis=0))
    W = within_std
    B_over_n = between_std**2 / convergence_period
    Vhat = ((convergence_period-1.)/convergence_period * W
            + B_over_n + B_over_n / float(nwalkers))
    c = Vhat/W
    return i - convergence_period/2, c


# In[Markov Chain Montecarlo/emcee for all scales]
def spec_2D_mann_lookup(param,args = ()):
    # Mann 1994, 3D atmpspheric turbulence
    # args = (param_fix,param_ind,k, path)  
    # parameters = (alpha_ep, L, gamma, Dp, rp, Ds, theta)
    param_fix = args[0]
    param_ind = args[1]
    ks = args[2]
    path = args[3]
    param_tot = np.zeros(7)
    param_set = np.arange(7)
    param_ind_not = [e for e in param_set if e not in set(param_ind)]
    param_tot[param_ind] = param
    param_tot[param_ind_not] = param_fix  
    alpha_ep, L, gamma, Dp, rp, Ds, theta = param_tot
    n = 62
    m= 62
    comp = 6   
    spec_ij = np.fromfile(path+'/US2DSpec.bin', dtype=np.float64)
    spec_ij = np.reshape(spec_ij,(-1,comp,n,m)).T 
    # Grid 
    k1_g = np.r_[0,10**np.arange(-3,3.1,.1)]
    k2_g = np.r_[0,10**np.arange(-3,3.1,.1)]
    gamma_g = np.arange(0,5.1,.1)
    
    if len(ks) == 0:
        k1L = k1_g*L
        k2L = k2_g*L
    else:
        k1,k2 = ks
        k1L = k1*L
        k2L = k2*L   
    
    # Interpolator
    
    k1L = np.r_[-np.flip(k1L[1:]),k1L]
    k2L = np.r_[-np.flip(k2L[1:]),k2L]
    sign1 = np.sign(k1L)
    sign2 = np.sign(k2L)
    
    int_fun = np.zeros((k1L.shape[0],k2L.shape[0],comp))
    
    for i in range(comp):
        fun= RegularGridInterpolator((k1_g*L, k2_g*L, gamma_g), spec_ij[:,:,i,:])
        sign = np.meshgrid(sign1, sign2)
        sign = sign[0]*sign[1]
        k1L_g, k2L_g, G = np.meshgrid(k1L, k2L, gamma)
        k1k2gam = np.c_[np.abs(k1L_g).flatten(), np.abs(k2L_g).flatten(), G.flatten()]
        interp = L**(8/3)*alpha_ep*np.reshape(fun(k1k2gam),np.squeeze(k1L_g).shape)
        
        k_dot =  k1L_g*np.cos(theta) + k2L_g*np.sin(theta)
        k_cross = -k1L_g*np.sin(theta) + k2L_g*np.cos(theta) 
        Filter = np.exp(-(k_dot*rp/2)**2)*np.sinc(Dp*k_dot/(2*rp))*np.sinc(Ds*k_cross)
        
        if i in [1,2,4]:
            interp = sign*interp
        print(np.squeeze(k1L_g).shape,interp.shape,int_fun[:,:,i].shape)
        int_fun[:,:,i] = interp.T*Filter         
    return (int_fun)    
# In[]
def mann_table_lookup(param, args = ()):
    # Mann 1994, 3D atmpspheric turbulence
    # args = (param_fix, param_ind, kinput)  
    # parameters = (alpha_ep, L, gamma, Dp, rp, Ds, theta)    
    ks, gams, spec = mn.us_values_mann()    
    param_fix = args[0]
    param_ind = args[1]
    kinput = args[2]
    param_tot = np.zeros(5)
    param_set = np.arange(5)
    param_ind_not = [e for e in param_set if e not in set(param_ind)]
    param_tot[param_ind] = param
    param_tot[param_ind_not] = param_fix  
    alpha_ep, L, gamma, w, n = param_tot
    print(alpha_ep, L, gamma)    
    f1 = sp.interpolate.interp1d(gams, spec[0], axis=0)(gamma)
    f2 = sp.interpolate.interp1d(gams, spec[1], axis=0)(gamma)
    f3 = sp.interpolate.interp1d(gams, spec[2], axis=0)(gamma)
    f4 = sp.interpolate.interp1d(gams, spec[3], axis=0)(gamma)
    Psi0 = (L**(5/3))*alpha_ep*np.c_[f1,f2,f3,f4] 
    k0 = ks/L
    klog0 = np.log10(k0);
    klog = np.log10(kinput);
    f1 = sp.interpolate.interp1d(klog0, Psi0[:,0])(klog)
    f2 = sp.interpolate.interp1d(klog0, Psi0[:,1])(klog)
    f3 = sp.interpolate.interp1d(klog0, Psi0[:,2])(klog)
    f4 = sp.interpolate.interp1d(klog0, Psi0[:,3])(klog)
    H = 1/(1+(kinput*w)**n)
    Psi1 = np.c_[f1*H, f2*H, f3*H, f4*H]
    ind = ~np.isnan(Psi1[:,0])
    return (Psi1[ind,:].T)
   
# In[The one used here]
def spec_2D_mann_lookup_cont(param,args = ()):
    # Mann 1994, 3D atmpspheric turbulence
    # args = (param_fix,param_ind,k, path)  
    # parameters = (alpha_ep, L, gamma, Dp, rp, Ds, theta)
    param_fix = args[0]
    param_ind = args[1]
    ks = args[2]
    path = args[3]
    param_tot = np.zeros(7)
    param_set = np.arange(7)
    param_ind_not = [e for e in param_set if e not in set(param_ind)]
    param_tot[param_ind] = param
    param_tot[param_ind_not] = param_fix  
    alpha_ep, L, gamma, Dp, rp, Ds, theta1, theta2 = param_tot
    n = 62
    m= 62
    comp = 6   
    spec_ij = np.fromfile(path+'/US2DSpec.bin', dtype=np.float64)
    spec_ij = np.reshape(spec_ij,(-1,comp,n,m)).T 
    # Grid 
    k1_g = np.r_[0,10**np.arange(-3,3.1,.1)]
    k2_g = np.r_[0,10**np.arange(-3,3.1,.1)]
    gamma_g = np.arange(0,5.1,.1)
    
    if len(ks) == 0:
        k1L = k1_g*L
        k2L = k2_g*L
    else:
        k1,k2 = ks
        k1L = k1*L
        k2L = k2*L   
    
    # Interpolator
    
    k1L = np.r_[-np.flip(k1L[1:]),k1L]
    k2L = np.r_[-np.flip(k2L[1:]),k2L]
    sign1 = np.sign(k1L)
    sign2 = np.sign(k2L)
    
    int_fun = np.zeros((k1L.shape[0],k2L.shape[0],comp))
    
    for i in range(comp):
        fun= RegularGridInterpolator((k1_g*L, k2_g*L, gamma_g), spec_ij[:,:,i,:])
        sign = np.meshgrid(sign1, sign2)
        sign = sign[0]*sign[1]
        k1L_g, k2L_g, G = np.meshgrid(k1L, k2L, gamma)
        k1k2gam = np.c_[np.abs(k1L_g).flatten(), np.abs(k2L_g).flatten(), G.flatten()]
        interp = L**(8/3)*alpha_ep*np.reshape(fun(k1k2gam),np.squeeze(k1L_g).shape)
        
        k_dot =  k1L_g*np.cos(theta) + k2L_g*np.sin(theta)
        k_cross = -k1L_g*np.sin(theta) + k2L_g*np.cos(theta) 
        Filter = np.exp(-(k_dot*rp/2)**2)*np.sinc(Dp*k_dot/(2*rp))*np.sinc(Ds*k_cross)
        
        if i in [1,2,4]:
            interp = sign*interp
        print(np.squeeze(k1L_g).shape,interp.shape,int_fun[:,:,i].shape)
        int_fun[:,:,i] = interp.T*Filter         
    return (int_fun)

    