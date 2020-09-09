# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:39:47 2020

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
from os.path import isfile, join, getsize
from os import listdir
import ppiscanprocess.windfieldrec as wr
import joblib

date = datetime(1904, 1, 1) 

# In[Files]   
############### files and sql engines ########################################
file_name0 = '/mnt/mimer/lalc/db/scans/phase_2/west/raw_filt_0_phase2.db'
file_name1 = '/mnt/mimer/lalc/db/scans/phase_2/west/raw_filt_1_phase2.db'
file_in_path = '/mnt/mimer/lalc/results/correlations/west/phase_2'
L_file = '/mnt/mimer/lalc/results/correlations/west/L_ph2.pkl'
file_out = '/mnt/mimer/lalc/results/correlations/west/Table_phase2_L.h5'
filelist = [join(file_in_path,filename) for filename in listdir(file_in_path)]
csv_database_0_ind = create_engine('sqlite:///'+file_name0)
csv_database_1_ind = create_engine('sqlite:///'+file_name1)

# In[Functions]
def mov_con(x,N):
    return np.convolve(x, np.ones((N,))/N,mode='same')
def L_smooth(flux, N=6):    
    U = mov_con(flux[:,1], N)
    T = mov_con(flux[:,4], N)
    T = T+273.15
    uw = mov_con(flux[:,7], N)
    vw = mov_con(flux[:,9], N)
    wth = mov_con(flux[:,11], N)
    u_fric = (uw**2+vw**2)**.25
    k = .4
    g = 9.806
    L = -u_fric**3*T/(k*g*wth)
    #L[np.abs(L)>=500] = 500*np.sign(L[np.abs(L)>=500])
    stab = np.ones(L.shape)    
    indL = (L>0) & (L<100)
    stab[indL] = 0
    indL = (L>100) & (L<500)
    stab[indL] = 1
    indL = np.abs(L)>=500
    stab[indL] = 2
    indL = (L>-500) & (L<-100)
    stab[indL] = 3
    indL = (L>-100) & (L<0)
    stab[indL] = 4   
    return np.c_[u_fric, L, stab, U]

def synch_df(df0,df1,dtscan=45):
    s0 = df0.scan.unique()
    s1 = df1.scan.unique()
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s0])
    t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s1])
    dt = [(t0[i]-t1[j]).total_seconds() for i in range(len(t0)) for j in range(len(t1))]
    s = np.array([[s_0,s_1] for s_0 in s0 for s_1 in s1])
    t = np.array([[t_0,t_1] for t_0 in t0 for t_1 in t1])
    ind_synch = np.abs(dt)<dtscan
    if np.sum(ind_synch)==0:
       sync = []
       time = []
       off0 = s0[-1]
       off1 = s1[-1]
    else:
       sync = s[ind_synch,:]
       time = t[ind_synch,:]
       # Complete scans   
       n_scan_0 = np.array([(df0.scan==s).sum() for s in sync[:,0]]) 
       n_scan_1 = np.array([(df1.scan==s).sum() for s in sync[:,1]])
       sync = sync[(n_scan_0==45) & (n_scan_1==45),:]
       time = time[(n_scan_0==45) & (n_scan_1==45),:]
       off0 = sync[-1,0]+1
       off1 = sync[-1,1]+1 
    return (sync,time, off0*45,off1*45)

# In[Recover Integral length scale and Obukhov length]
#############################labels###########################################

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
labels_new = iden_lab
#Labels for range gates and speed
labels_mask = []
labels_ws = []
labels_rg = []
labels_CNR = []
labels_Sb = []
for i in np.arange(198):
    vel_lab = np.array(['range_gate_'+str(i),'ws_'+str(i),'CNR_'+str(i),'Sb_'+str(i)])
    mask_lab = np.array(['ws_mask'+str(i)])
    labels_new = np.concatenate((labels_new,vel_lab))
    labels = np.concatenate((labels,np.array(['range_gate','ws','CNR','Sb'])))
    labels_mask = np.concatenate((labels_mask,mask_lab))
    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
    labels_Sb = np.concatenate((labels_Sb,np.array(['Sb_'+str(i)])))
labels_new = np.concatenate((labels_new,np.array(['scan'])))    
labels_short = np.array([ 'stop_time', 'azim'])
for w,r in zip(labels_ws,labels_rg):
    labels_short = np.concatenate((labels_short,np.array(['ws','range_gate', 'CNR', 'Sb'])))
labels_short = np.concatenate((labels_short,np.array(['scan'])))   
lim = [-8,-24]
i=0
col = 'SELECT '
col_raw = 'SELECT '
for w,r,c, s in zip(labels_ws,labels_rg,labels_CNR, labels_Sb):
    if i == 0:
        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', ' + s + ', '
        col_raw = col_raw  +  w  +  ', '
    elif (i == len(labels_ws)-1):
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', scan'
        col_raw = col_raw + ' ' + w
    else:
        col = col + ' ' + w + ', ' + r + ', ' + c + ', ' + s + ', ' 
        col_raw = col_raw + ' ' + w + ', '
    i+=1
selec_fil = col + ' FROM "table_fil"'

###################data frame master###########################################
Lcorr  = pd.DataFrame()
L10min = pd.DataFrame() 
#########################daily loops###########################################

for file_name in filelist:
    ###################daily dataframe 10min###################################
    print(file_name)
    L = pd.read_hdf(file_name, key='L_10min')
    Lscan = pd.read_hdf(file_name, key='L')
    ###################daily times#############################################
    name_min, hms_min = Lscan.index.min()
    name_max, hms_max = Lscan.index.max()
    Lscan = None
    t_init = datetime.strptime(name_min+hms_min, '%Y%m%d%H%M%S')
    t_end = datetime.strptime(name_max+hms_max, '%Y%m%d%H%M%S')+timedelta(hours = 1)
    name_id= np.unique(pd.date_range(t_init, t_end,freq='10T').strftime('%Y%m%d'))[0]
    t_arrayhms = pd.date_range(t_init, t_end,freq='10T').strftime('%H%M%S')
    ########################queries and reliable scans#############################
    query_fil_0 = selec_fil+ ' where name = ' + name_min 
    query_fil_1 = selec_fil+ ' where name = ' + name_min
    # First database loading
    print('reading df0')
    df_0 = pd.read_sql_query(query_fil_0, csv_database_0_ind)
    df_0.columns = labels_short
    # Second database loading
    print('reading df1')
    df_1 = pd.read_sql_query(query_fil_1, csv_database_1_ind)
    df_1.columns = labels_short
    s_syn,t_scan,_,_ = synch_df(df_0,df_1,dtscan=45/2)
    ind_df0 = df_0.scan.isin(s_syn[:,0])
    df0 = df_0.loc[ind_df0]
    ind_df1 = df_1.scan.isin(s_syn[:,1])
    df1 = df_1.loc[ind_df1] 
    chunk = len(df0.azim.unique())    
    rt = df0.loc[df0.scan==s_syn[0,0]].CNR.values.flatten().shape[0]    
    ind_cnr0 = np.sum(((df0.CNR >-24) & (df0.CNR < -8)).values, axis = 1)
    ind_cnr1 = np.sum(((df1.CNR >-24) & (df1.CNR < -8)).values, axis = 1)
    rel0 = ind_cnr0.reshape((int(len(ind_cnr0)/chunk),chunk)).sum(axis=1)/rt
    rel1 = ind_cnr1.reshape((int(len(ind_cnr1)/chunk),chunk)).sum(axis=1)/rt
    t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in s_syn[:,0]])
    t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in s_syn[:,1]])
    rel = np.array([np.min([r0,r1]) for r0,r1 in zip(rel0,rel1)])
    rel = np.c_[s_syn[:,0],t0,rel0,s_syn[:,1],t1,rel1,rel]
    
    ###############################################################################
    loc0 = np.array([0,6322832.3])
    loc1 = np.array([0,6327082.4])
    d = loc1-loc0  
    phi0 = df0.azim.unique()
    phi1 = df1.azim.unique()               
    r0 = df0.range_gate.iloc[0].values
    r1 = df1.range_gate.iloc[0].values                
    r_0, phi_0 = np.meshgrid(r0, np.pi/2-np.radians(phi0)) # meshgrid
    r_1, phi_1 = np.meshgrid(r0, np.pi/2-np.radians(phi1)) # meshgrid                
    tree,tri, w0, neigh0, index0, w1, neigh1, index1 =  wr.grid_over2((r_0, phi_0),(r_1,phi_1), -d)
    chunk = 256
    _, _, grdu, so = wr.direct_wf_rec((df0.loc[df0.scan == s_syn[0,0]]).astype(np.float32),
                                      (df1.loc[df1.scan == s_syn[0,1]]).astype(np.float32),
                                      tri, d, N_grid = chunk)
    find = np.reshape(tri.get_trifinder()(grdu[0].flatten(), grdu[1].flatten()),grdu[0].shape)
    c = len(grdu[0].flatten())/np.sum(find!=-1)
    
    ################## 10min periods and averages #############################    
    for i in range(len(t_arrayhms)-1):
        hms1 = t_arrayhms[i]
        hms2 = t_arrayhms[i+1]
        L_scan = pd.read_hdf(file_name,where = 'name == name_id & hms >= hms1 & hms < hms2', key='L')        
        if L_scan.empty:
            print('empty df')
        else:
            ind_rel = np.isin(pd.to_datetime(rel[:,1]),L_scan.time.values)
            L_scan['relscan'] = rel[ind_rel,-1]
            L_scan['area_frac'] = L_scan['area_frac'].values*c 
            L.loc[(name_id,hms1),'area_frac'] = np.nanmedian(L_scan['area_frac'].values.astype(float))
            L.loc[(name_id,hms1),'relscan'] = np.nanmedian(L_scan['relscan'].values.astype(float))
            Lcorr = pd.concat([Lcorr,L_scan],axis=0)
    L10min = pd.concat([L10min,L],axis=0)


#####################save just in case#########################################        
joblib.dump((Lcorr,L10min),'/mnt/mimer/lalc/results/correlations/west/storeLph2.pkl')

##################### Obukhov Length ##########################################

res_flux = joblib.load(L_file)
heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
L_list_2 = np.hstack([L_smooth(np.array(res_flux)[:,i,:]) for i in range(len(heights))])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux)[:,2,0]])
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]
cols = [item for sublist in cols for item in sublist]
stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2

############################# 10 min ##########################################
L10min = L10min[~L10min.index.duplicated(keep='first')]
L10min.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in L10min.index.values]])
L10min[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
print(len(Lph2.index), len(L10min.index), len(tL))
Lph2.index = L10min.index
L10min['time'] = tL 
for i in range(len(t2)-1):
    ind = (L10min.time>=t2[i]) & (L10min.time<=t2[i+1])
    if ind.sum()>0:
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values

L10min[cols] = Lph2
L10min.sort_values(by=['time'], inplace = True)

############################# Instantaneous ###################################

Lcorr = Lcorr[~Lcorr.index.duplicated(keep='first')]
Lcorr.dropna(inplace=True)
tL = pd.to_datetime([datetime.strptime(d,'%Y%m%d%H%M%S') for d in [''.join(l) for l in Lcorr.index.values]])
Lcorr[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(tL),len(cols))))
Lph2.index = Lcorr.index
Lcorr['time'] = tL 
for i in range(len(t2)-1):
    ind = (Lcorr.time>=t2[i]) & (Lcorr.time<=t2[i+1])
    if ind.sum()>0:
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values
Lcorr[cols] = Lph2
Lcorr.sort_values(by=['time'], inplace = True)

#####################save just in case#########################################        
joblib.dump((Lcorr,L10min),'/mnt/mimer/lalc/results/correlations/west/storeLph2.pkl')

columnshdf5 = Lcorr.iloc[[0]].reset_index().columns.tolist()
Lcorr.to_hdf(file_out,'L',mode='a', data_columns = columnshdf5,format='table', append = True)
columnshdf5 = L10min.iloc[[0]].reset_index().columns.tolist()
L10min.to_hdf(file_out,'L10min',mode='a', data_columns = columnshdf5,format='table', append = True)

