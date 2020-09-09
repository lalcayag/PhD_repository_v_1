# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:00:28 2020

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
from scipy.spatial import Delaunay
import joblib
date = datetime(1904, 1, 1) 

# check if  lengths are right

# load

file_corr = '/mnt/mimer/lalc/results/correlations/west/phase_2/corr_uv_west_phase2_ind.db'
file_out_path = '/mnt/mimer/lalc/results/correlations/west/phase_2/'

csv_database_r = create_engine('sqlite:///'+file_corr)
dfL_phase2 = pd.read_sql_query("""select * from "L" """,csv_database_r)
joblib.dump(dfL_phase2, file_out_path+'dfL_phase2.pkl')


file_out_path_local = 'C:/Users/lalc/Documents/PhD/Python Code/repository_v1/data_process/results/'
dfL_phase2_test = joblib.load(file_out_path_local+'dfL_phase2.pkl')

with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)    


dfL_phase2 = dfL_phase2_test.copy()
    
dfL_phase2.dropna(inplace=True)
dfL_phase2.reset_index(inplace=True)
dfL_phase2.drop(['index'],axis=1,inplace=True)
dfL_phase2 = dfL_phase2.drop_duplicates(subset='time', keep='first')

heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
    
L_list_2 = np.hstack([L_smooth(np.array(res_flux_2)[:,i,:]) for i in range(len(heights))])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_2)[:,2,0]])
    
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]

cols = [item for sublist in cols for item in sublist]

stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2

dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph2.index = dfL_phase2.index
aux2 = dfL_phase2.time
dfL_phase2.time = pd.to_datetime(dfL_phase2.time)

for i in range(len(t2)-1):
    print(i,t2[i])
    ind = (dfL_phase2.time>=t2[i]) & (dfL_phase2.time<=t2[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values

dfL_phase2.time = aux2
dfL_phase2[cols] = Lph2
dfL_phase2.sort_values(by=['time'], inplace = True)
ind = drel_phase2.time0.isin(dfL_phase2.time)
timein = pd.concat([drel_phase2.time0.loc[ind],drel_phase2.time1.loc[ind],drel_phase2.relscan.loc[ind]],axis = 1)
timein.index = dfL_phase2.index
dfL_phase2[['time0','time1','rel']] = timein
dfL_phase2['date'] = pd.to_datetime(dfL_phase2.time.values)


plt.figure()
plt.scatter(dfL_phase2['$U_{175}$'], dfL_phase2['$L_{h,x}$'].values/dfL_phase2['$L_{h,y}$'].values)


