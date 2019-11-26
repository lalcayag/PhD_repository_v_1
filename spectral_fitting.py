
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:54:15 2018

@author: lalc
"""

import numpy as np
import pandas as pd
import pickle
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize, abspath
from os import listdir
import sys
import tkinter as tkint
import tkinter.filedialog
import spectralfitting.spectralfitting as sf


# In[Data loading]
# To do: it is necessary to link the mask file with the source file
root = tkint.Tk()
file_spec_path = tkint.filedialog.askopenfilenames(parent=root,title='Choose a Spectra file')
root.destroy()
root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output file')
root.destroy()

# In[Data loading]
with open(file_spec_path[0], 'rb') as S:
     Su_u,Sv_v,Su_v,k_1,k_2 = pickle.load(S)

# In[Spectra fitting]
scan_n = 13000
F_obs = .5*(Su_u[scan_n,:] + Sv_v[scan_n,:])
res = spectra_fitting(F_obs[k_1[scan_n,:]>0],spectra_peltier,spectra_noise,
                      k_1[scan_n,k_1[scan_n,:]>0]) 


# In[]

k = k_1[scan_n,k_1[scan_n,:]>0]*(2*np.pi)
S_obs = F_obs[k_1[scan_n,:]>0]
param = [.85,1,1000,3, 1.6,1,100,.36,2,4,0.001,1,2]
plt.plot(k/(2*np.pi),k*S_obs/(2*np.pi),'o')
plt.plot(k/(2*np.pi),k*(spectra_peltier(param[:10],args=(k,))+spectra_noise(param[11:13],args=(k,)))/(2*np.pi),'-',lw=2)
#plt.plot(k,(spectra_peltier(param[:8],args=(k,))+spectra_noise(param[9:11],args=(k,)))*(np.sin(k*(3*np.pi)/np.max(k))/(k*(3*np.pi)/np.max(k)))**3,'--',lw=2)
plt.xscale('log')
plt.yscale('log')

import scipy.optimize as optimization

def spec_slop(k,a,b):
    return a*(k**b)

limits = [10**-2,5*10**-2]

params = np.zeros((len(limits),2))

for i in range(len(limits)-1):
    print(i)
    ind = [(k>limits[i])&(k<limits[i+1])]
    params[i,:] = optimization.curve_fit(spec_slop, k[ind], S_obs[ind]*k[ind]**(-2), [-10,-2/3])[0]
    params[i,:] = optimization.curve_fit(spec_slop, k[ind], (spectra_peltier(param[:8],args=(k,))+spectra_noise(param[9:11],args=(k,)))[ind], [-10,-2/3])[0]
    plt.plot(k[ind],spec_slop(k[ind],params[i,0],params[i,1]))
