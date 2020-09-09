# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:01:59 2020

@author: lalc
"""

import numpy as np
import scipy as sp
import pandas as pd
import os
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectra_construction as sc
from sqlalchemy import create_engine
from datetime import datetime, timedelta
date = datetime(1904, 1, 1) 

# In[]
index = int(sys.argv[1])
suf = str(index).zfill(2) 

file_name = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_spec_phase2_d'+suf+'.h5'

L = pd.read_hdf(file_name, key='L')

name_min, hms_min = L.index.min()
name_max, hms_max = L.index.max()
t_init = datetime.strptime(name_min+hms_min, '%Y%m%d%H%M%S')
t_end = datetime.strptime(name_max+hms_max, '%Y%m%d%H%M%S')+timedelta(hours = 1)
name_id= np.unique(pd.date_range(t_init, t_end,freq='10T').strftime('%Y%m%d'))[0]
t_arrayhms = pd.date_range(t_init, t_end,freq='10T').strftime('%H%M%S')
N = 256

for i in range(len(t_arrayhms)-1):
    hms1 = t_arrayhms[i]
    hms2 = t_arrayhms[i+1]
    corr = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='corr')
    L = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='L')
    
    if corr.empty:
        print('empty df')
    else:

    reslist = [corr[['tau', 'eta', 'r_u', 'r_v', 'r_uv']].loc[corr.scan == s].values for s in L.scan.values]

    tau_list = [np.split(r,r.shape[1],axis=1)[0] for r in reslist]
    eta_list = [np.split(r,r.shape[1],axis=1)[1] for r in reslist]
    ru_list = [np.split(r,r.shape[1],axis=1)[2] for r in reslist]
    rv_list = [np.split(r,r.shape[1],axis=1)[3] for r in reslist]
    ruv_list = [np.split(r,r.shape[1],axis=1)[4] for r in reslist]
    resarray = np.vstack([r for r in reslist])
    tau_arr, eta_arr, ru_arr, rv_arr, ruv_arr = np.split(resarray,resarray.shape[1],axis=1)

    taumax = np.nanmax(np.abs(tau_arr))
    taui = np.linspace(0,taumax,int(N/2)+1)
    taui = np.r_[-np.flip(taui[1:]),taui]
    etamax = np.nanmax(np.abs(eta_arr))
    etai = np.linspace(0,etamax,int(N/2)+1)
    etai = np.r_[-np.flip(etai[1:]),etai]
    taui, etai = np.meshgrid(taui,etai)

    rui = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ru_list)])
    rvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, rv_list)])
    ruvi = np.vstack([sc.autocorr_interp_sq(r,t,e,tau_lin = taui, eta_lin = etai)[2].T for t,e,r in zip(tau_list, eta_list, ruv_list)])

    ru_mean = np.nanmean(rui.reshape((int(len(rui)/(N+1)),N+1,N+1)),axis=0)
    rv_mean = np.nanmean(rvi.reshape((int(len(rvi)/(N+1)),N+1,N+1)),axis=0)
    ruv_mean = np.nanmean(ruvi.reshape((int(len(ruvi)/(N+1)),N+1,N+1)),axis=0)

    k1, k2, Suu, Svv, Suv = sc.spectra_fft((taui,etai),ru_mean,rv_mean,ruv_mean)
    Sh = .5*(Suu+Svv)
    k1, k2 = np.meshgrid(k1, k2)
    
    data = np.c_[k1.flatten(), k2.flatten(), Suu.flatten(), Svv.flatten(), Suv.flatten(), Sh.flatten()]
    
    times = np.repeat(t_arrayhms[i],len(k1.flatten()))
    names_ids = np.repeat(name_id,len(k1.flatten()))

    columns = ['k1','k2','Suu','Svv','Suv', 'Sh']
    
    aux0 = pd.DataFrame(data, columns = columns)
    
    aux0['name'] = names_ids
    aux0['hms'] = times
    
    aux0.set_index(['name','hms'], inplace=True)
    
    cols_hdf5 = aux0.iloc[[0]].reset_index().columns.tolist()
 
    aux0.to_hdf(file_name,'Spec',mode='a', data_columns = ,
                format='table', append = True)




