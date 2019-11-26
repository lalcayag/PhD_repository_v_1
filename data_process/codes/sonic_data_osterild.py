# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:32:44 2019

@author: lalc
"""
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import tkinter as tkint
import tkinter.filedialog
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import datetime
from datetime import datetime, timedelta
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[FUnctions for stability]
# Richarson number
# Bulk Richardson number
# Ri(zm) = ((g/theta)*(Delta z_m)/(Delta U)**2)ln(z_2/z_1)
    
def stability_grad(t0, t1, x0, x1, y0, y1, z0, z1, p0, p1):
    u0 = np.sqrt(x0**2+y0**2)
    u1 = np.sqrt(x1**2+y1**2)
    theta0 = t0*(p0/1000)**.286
    theta1 = t1*(p1/1000)**.286
    dtheta = theta1 - theta0
    du = u1 - u0
    Ri = np.zeros(t0.shape)*np.nan
    L = np.zeros(t0.shape)*np.nan
    stab = np.zeros(t0.shape)*np.nan
    Ri = 9.806*dtheta*.5*(z1+z0)*np.log(z1/z0)/du**2/theta0
    Ri[Ri>0.2] = 0.2
    indRi = Ri <= 0
    L[indRi] = .5*(z1+z0)/Ri[indRi]
    L[L<-500] = -500
    indRi = (Ri > 0) & (Ri < .2)
    L[indRi] = .5*(z1+z0)*(1-5*Ri[indRi])/Ri[indRi]
    L[L>500] = 500
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
    return np.c_[Ri , L, stab]
 
def stability_bulk(t0, t1, x1, y1, z0, z1, p0, p1, zf = .1):    
    u1 = np.sqrt(x1**2+y1**2)
    theta0 = t0*(p0/1000)**.286
    theta1 = t1*(p1/1000)**.286
    dtheta = theta1 - theta0
    Ri = np.zeros(t0.shape)*np.nan
    # Initial guess
    L = np.zeros(t0.shape)*np.nan
    stab = np.zeros(t0.shape)*np.nan
    Ri = (z1*9.806*dtheta/theta1)/u1**2
    L = np.ones(Ri.shape)*z1
    from scipy.optimize import fsolve 
    for i in range(len(L)):
        func = lambda l: np.abs(l-z1/(Ri[i]*F_G(l,z1,z0,zf=.1)))
        L[i] =  fsolve(func, L[i])
    L[np.abs(L)>=500] = 500    
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
    return np.c_[Ri, L, stab] 

def F_G(L,z1,z0,zf):
    ef = (1-16*zf/L)**.25
    e1 = (1-16*z1/L)**.25
    l0 = (1-16*z0/L)**.5
    l1 = (1-16*z1/L)**.5
    indL = L<0
    F = np.zeros(L.shape)*np.nan
    F[indL] = (np.log((z1/zf)*((ef[indL]**2+1)/(e1[indL]**2+1))*((ef[indL]+1)/(e1[indL]+1))**2)+
                     2*np.arctan((ef[indL]-e1[indL])/(1+ef[indL]*e1[indL])))
    F[~indL] = (np.log((z1/zf))+5*z1/L[~indL])
    indL = L<=0
    G = np.zeros(L.shape)*np.nan
    G[indL] = (np.log((z1/z0)*((l0[indL]+1)/(l1[indL]+1))**2)) 
    G[~indL] = (np.log(z1/z0) + 5*(z1-z0)/L[~indL])
    return F**2/G        

def stability_flux(t0,t1, x0, y0, x1, y1, w1, z0, z1, p0, p1):
    #this will 10 Hz data
    theta_mean = .5*(np.nanmean(t0*(p0/1000)**.286)+np.nanmean(t1*(p1/1000)**.286))
    theta1 = t1*(p1/1000)**.286
    u0 = np.sqrt(x0**2+y0**2)
    U0 = np.nanmean(u0)
    w1 = w1-np.nanmean(w1)
    u1 = np.sqrt(x1**2+y1**2)
    U1 = np.nanmean(u1)
    zm = .5*(z1+z0)
    dudz = (U1-U0)/zm/np.log(z1/z0)
    wth = np.nanmean(w1*(theta1-np.nanmean(theta1)))
    uw = np.nanmean((u1-U1)*w1)
    Ri = 9.806*wth/(theta_mean*uw*dudz)
    L = zm/Ri
    stab = np.nan
    if (L>0) & (L<=100):
        stab = 0
    if (L>100) & (L<500):
        stab = 1
    if np.abs(L)>500:
        stab = 2
    if (L>-500) & (L<-100):
        stab = 3
    if (L>-100) & (L<0):
        stab = 4  
    return np.c_[Ri, L, stab,theta_mean,dudz,uw,wth] 

def valid_col(df_stab,cols):
    indextot=[]
    for j, c in enumerate(cols):
        index = []
        for i,cc in enumerate(c):
            value = df_stab[cc].values
            if np.sum(value == None)==0:
                index.append(i)
        indextot.append(index)    
    return indextot

def obukhov_L(t, x, y, w, z, p, name):
    theta = t*(p/1000)**.286
    X = np.nanmean(x)
    Y = np.nanmean(y)
    angle = np.arctan2(Y,X)
    # Components in matrix of coefficients
    S11 = np.cos(angle)
    S12 = np.sin(angle)
    R = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[x.flatten(),y.flatten()]).T
    vel = np.dot(R,vel)
    u = vel[0,:]
    v = vel[1,:]
    w = w-np.nanmean(w)
    u_fric = (np.nanmean(u*w)**2+np.nanmean(v*w)**2)**.25
    wth = np.nanmean(w*(theta-np.nanmean(theta)))
    k = .4
    g = 9.806
    L = -u_fric**3*np.nanmean(theta)/(k*g*wth)
    return [int(name),np.nanmean(u),u_fric,wth,L,np.nanmean(theta),z]

def fluxes(t, x, y, w, z, p, name):
    th = (t+273.15)*(p/1000)**.286
    X = np.nanmean(x)
    Y = np.nanmean(y)
    angle = np.arctan2(Y,X)
    # Components in matrix of coefficients
    S11 = np.cos(angle)
    S12 = np.sin(angle)
    R = np.array([[S11,S12], [-S12,S11]])
    vel = np.array(np.c_[x.flatten(),y.flatten()]).T
    vel = np.dot(R,vel)
    u = vel[0,:]
    v = vel[1,:]
    U = np.nanmean(u)
    V = np.nanmean(v)
    W = np.nanmean(w)
    T = np.nanmean(th)   
    u  = u-U
    v  = v-V
    w  = w-W
    th = th-T    
    uu  = np.nanmean(u*u)
    uv  = np.nanmean(u*v)
    uw  = np.nanmean(u*w)
    vv  = np.nanmean(v*v)
    vw  = np.nanmean(v*w)
    ww  = np.nanmean(w*w)
    wth = np.nanmean(w*th)   
    return [int(name), U, V, W, T, uu, uv, uw, vv, vw, ww, wth, z]

def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)

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
    L = -u_fric**3*T/(k*g*wth)
    L[np.abs(L)>=500] = 500*np.sign(L[np.abs(L)>=500])
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
def datetimeDF(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

# In[Directory of the input and output data]

root = tkint.Tk()
file_in_path_db_phase1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir phase 1, corr db')
root.destroy()

root = tkint.Tk()
file_in_path_db_phase2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir phase 2, corr db')
root.destroy()

# Filtered scans
root = tkint.Tk()
file_in_path_0_ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir filtered scan, phase 1')
root.destroy()
root = tkint.Tk()
file_in_path_1_ph1 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir filtered scan, phase 1')
root.destroy()

root = tkint.Tk()
file_in_path_0_ph2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir filtered scan, phase 2')
root.destroy()
root = tkint.Tk()
file_in_path_1_ph2 = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir filtered scan, phase 2')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

root = tkint.Tk()
file_out_path_df = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir DataFrame')
root.destroy()

root = tkint.Tk()
file_out_figures = tkint.filedialog.askdirectory(parent=root,title='Choose an Output, figures')
root.destroy()

# In[MySQL Østerild sonic and lidar data]
osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')

# In[Lidar data] 
csv_database_r1 = create_engine('sqlite:///'+file_in_path_db_phase1+'/corr_uv_west_phase1_ind.db')
csv_database_r2 = create_engine('sqlite:///'+file_in_path_db_phase2+'/corr_uv_west_phase2_ind.db')

drel_phase1 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r1)
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r2)
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)

# In[Sonic data from data base]
times_phase1 = pd.to_datetime(dfL_phase1.time.values)
times_phase2 = pd.to_datetime(dfL_phase2.time.values).sort_values()

Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']

T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
X_1 = ['X_241m_LMN','X_175m_LMN','X_103m_LMN','X_37m_LMN','X_7m_LMN']
Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
Y_1 = ['Y_241m_LMN','Y_175m_LMN','Y_103m_LMN','Y_37m_LMN','Y_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']
P_0 = ['Press_241m_LMS','Press_7m_LMS']
P_1 = ['Press_241m_LMN','Press_7m_LMN']

times_stamp_ph1 = times_phase1.strftime("%Y%m%d%H%M").values
times_stamp_ph2 = times_phase2.strftime("%Y%m%d%H%M").values

# List of dates with pandas

init = times_phase1.strftime("%Y%m%d").values[0]
end = (times_phase1+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph1 = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='10T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())

init = times_phase2.strftime("%Y%m%d").values[0]
end = (times_phase2+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph2 = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='10T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())

######################
# The df at once
stab_phase1 = pd.DataFrame()
stab_phase2 = pd.DataFrame()
table = ' from calmeans '
where = 'where name >= (select max(name)' + table + 'where name < ' + t_ph1[0] + ') and ' + 'name <= (select min(name)' + table + 'where name > ' + t_ph1[-1] + ')'
sql_query_1 = 'select name, ' + ", ".join(T_0+T_1+X_0+X_1+Y_0+Y_1+Z_0+Z_1+P_0+P_1+Tabs_0+Tabs_1+Sspeed_0+Sspeed_1) + table +  where
stab_phase1 = pd.read_sql_query(sql_query_1, osterild_database)

where = 'where name >= (select max(name)' + table + 'where name < ' + t_ph2[0] + ') and ' + 'name <= (select min(name)' + table + 'where name > ' + t_ph2[-1] + ')'
sql_query_2 = 'select name, ' + ", ".join(T_0+T_1+X_0+X_1+Y_0+Y_1+Z_0+Z_1+P_0+P_1+Tabs_0+Tabs_1+Sspeed_0+Sspeed_1) + table +  where
stab_phase2 = pd.read_sql_query(sql_query_2,osterild_database)
    
######################    
    
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/stab_phase1.pkl', 'wb') as writer:
    pickle.dump(stab_phase1,writer)
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/stab_phase2.pkl', 'wb') as writer:
    pickle.dump(stab_phase2,writer)      
      
# In[Stability]
#Phase 1, South
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/stab_phase1.pkl', 'rb') as reader:
    stab_phase1 = pickle.load(reader)
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/stab_phase2.pkl', 'rb') as reader:
    stab_phase2 = pickle.load(reader) 
    
i, j = 3, 2
#Delta = j-i    
heights = [241, 175, 103, 37, 7]
t0 = stab_phase2[T_0[i]].values
t1 = stab_phase2[T_0[j]].values
x0 = stab_phase2[X_0[i]].values
x1 = stab_phase2[X_0[j]].values
y0 = stab_phase2[Y_0[i]].values
y1 = stab_phase2[Y_0[j]].values
z0 = heights[i]
z1 = heights[j]
p0 = stab_phase2[P_0[1]].values
p1 = stab_phase2[P_0[1]].values
name = stab_phase2.name.values
res_grad2 = stability_grad(t0, t1, x0, x1, y0, y1, z0, z1, p0, p1) 
res_bulk2 = stability_bulk(t0, t1, x1, y1, z0, z1, p0, p1)
cols = ['name','$Ri_{grad}$','$L_{grad}$','$stab_{grad}$','$Ri_{bulk}$','$L_{bulk}$','$stab_{bulk}$']
stab_phase2_df = pd.DataFrame(columns = cols, data = np.c_[name, res_grad2, res_bulk2])

t0 = stab_phase1[T_0[i]].values
t1 = stab_phase1[T_0[j]].values

tabs0 = stab_phase1[Tabs_0[i]].values
tabs1 = stab_phase1[Tabs_0[j]].values

x0 = stab_phase1[X_0[i]].values
x1 = stab_phase1[X_0[j]].values
y0 = stab_phase1[Y_0[i]].values
y1 = stab_phase1[Y_0[j]].values
z0 = heights[i]
z1 = heights[j]
p0 = stab_phase1[P_0[1]].values
p1 = stab_phase1[P_0[1]].values
name = stab_phase1.name.values
res_grad1 = stability_grad(t0, t1, x0, x1, y0, y1, z0, z1, p0, p1) 
res_bulk1 = stability_bulk(t0, t1, x1, y1, z0, z1, p0, p1)
cols = ['name','$Ri_{grad}$','$L_{grad}$','$stab_{grad}$','$Ri_{bulk}$','$L_{bulk}$','$stab_{bulk}$']
stab_phase1_df = pd.DataFrame(columns = cols, data = np.c_[name, res_grad1, res_bulk1])

# In[This]
osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')
times_phase1 = pd.to_datetime(dfL_phase1.time.values)
times_phase2 = pd.to_datetime(dfL_phase2.time.values).sort_values()

times_stamp_ph1 = times_phase1.strftime("%Y%m%d%H%M").values
times_stamp_ph2 = times_phase2.strftime("%Y%m%d%H%M").values

# List of dates with pandas

init = times_phase1.strftime("%Y%m%d").values[0]
end = (times_phase1+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph1 = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='10T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())

init = times_phase2.strftime("%Y%m%d").values[0]
end = (times_phase2+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph2 = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='10T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())   

Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
T_0 = ['T_241m_LSM','T_175m_LSM','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
X_0 = ['X_241m_LMS','X_175m_LMS','X_103m_LMS','X_37m_LMS','X_7m_LMS']
X_1 = ['X_241m_LMN','X_175m_LMN','X_103m_LMN','X_37m_LMN','X_7m_LMN']
Y_0 = ['Y_241m_LMS','Y_175m_LMS','Y_103m_LMS','Y_37m_LMS','Y_7m_LMS']
Y_1 = ['Y_241m_LMN','Y_175m_LMN','Y_103m_LMN','Y_37m_LMN','Y_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']
P_0 = ['Press_241m_LMS','Press_7m_LMS']
P_1 = ['Press_241m_LMN','Press_7m_LMN']

res_flux_1=[]
for j,stamp in enumerate(t_ph1):
    print(stamp,j)    
    table_20Hz = ' from caldata_'+stamp[:4]+'_'+stamp[4:6]+'_20hz '
    where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stamp + ')'
    sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0) + table_20Hz +  where
    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
    
    stab_20hz.fillna(value=pd.np.nan, inplace=True)
    
    table = ' from calmeans '
    where = 'where name = (select max(name)' + table + 'where name < ' + stamp + ')'
    sql_query = 'select name, ' + ", ".join(P_0) + table +  where
    stab_press = pd.read_sql_query(sql_query,osterild_database) 
   
    if len(stab_20hz)>0:
        L=[]
        for i in range(len(T_0)):
            t0 = stab_20hz[T_0[i]].values
            x0 = stab_20hz[X_0[i]].values
            y0 = stab_20hz[Y_0[i]].values
            z0 = heights[i]
            w0 = stab_20hz[Z_0[i]].values
            p1, p0 = stab_press.values[0,1:3]
            if len(t0)>0:
                L.append(fluxes(t0, x0, y0, w0, z0, p0, stamp))
        res_flux_1.append(L)    
    
res_flux_2=[]
T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
for j, stamp in enumerate(t_ph2[4752:]):
    print(stamp,j+4752)
    table_20Hz = ' from caldata_'+stamp[:4]+'_'+stamp[4:6]+'_20hz '
    where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stamp + ')'
    sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0) + table_20Hz +  where
    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
    
    stab_20hz.fillna(value=pd.np.nan, inplace=True)
    
    table = ' from calmeans '
    where = 'where name = (select max(name)' + table + 'where name < ' + stamp + ')'
    sql_query = 'select name, ' + ", ".join(P_0) + table +  where
    stab_press = pd.read_sql_query(sql_query,osterild_database) 
   
    if len(stab_20hz)>0:
        L=[]
        for i in range(len(T_0)):
            t0 = stab_20hz[T_0[i]].values
            x0 = stab_20hz[X_0[i]].values
            y0 = stab_20hz[Y_0[i]].values
            z0 = heights[i]
            w0 = stab_20hz[Z_0[i]].values
            p1, p0 = stab_press.values[0,1:3]
            if len(t0)>0:
                L.append(fluxes(t0, x0, y0, w0, z0, p0, stamp))
        res_flux_2.append(L)         
   
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph1.pkl', 'wb') as writer:
#    pickle.dump(res_flux_1,writer) 
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'wb') as writer:
#    pickle.dump(res_flux_2,writer) 
    
# In[]

with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph1.pkl', 'rb') as reader:
    res_flux_1 = pickle.load(reader)   
with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/L_ph2.pkl', 'rb') as reader:
    res_flux_2 = pickle.load(reader)      

drel_phase1 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r1)
dfL_phase1 = pd.read_sql_query("""select * from "L" """,csv_database_r1)
dfL_phase1.dropna(inplace=True)
dfL_phase1.reset_index(inplace=True)
dfL_phase1.drop(['index'],axis=1,inplace=True)
dfL_phase1 = dfL_phase1.drop_duplicates(subset='time', keep='first')


drel_phase2 = pd.read_sql_query("""select * from "reliable_scan" """,csv_database_r2)
dfL_phase2 = pd.read_sql_query("""select * from "Lcorrected" """,csv_database_r2)
dfL_phase2.dropna(inplace=True)
dfL_phase2.reset_index(inplace=True)
dfL_phase2.drop(['index'],axis=1,inplace=True)
dfL_phase2 = dfL_phase2.drop_duplicates(subset='time', keep='first')

heights = [241, 175, 103, 37, 7]
k = .4
g = 9.806
    
L_list_1 = np.hstack([L_smooth(np.array(res_flux_1)[:,i,:]) for i in range(len(heights))])
L_list_2 = np.hstack([L_smooth(np.array(res_flux_2)[:,i,:]) for i in range(len(heights))])
t1 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_1)[:,2,0]])
t2 = pd.to_datetime([str(int(n)) for n in np.array(res_flux_2)[:,2,0]])
    
cols = [['$u_{star,'+str(int(h))+'}$', '$L_{flux,'+str(int(h))+'}$',
         '$stab_{flux,'+str(int(h))+'}$', '$U_{'+str(int(h))+'}$'] for h in heights]

cols = [item for sublist in cols for item in sublist]

stab_phase1_df = pd.DataFrame(columns = cols, data = L_list_1)
stab_phase1_df['time'] = t1

stab_phase2_df = pd.DataFrame(columns = cols, data = L_list_2)
stab_phase2_df['time'] = t2

dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
dfL_phase1[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))

Lph1 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))
Lph2 = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
Lph1.index = dfL_phase1.index
Lph2.index = dfL_phase2.index

aux1 = dfL_phase1.time
aux2 = dfL_phase2.time

dfL_phase1.time = pd.to_datetime(dfL_phase1.time)
dfL_phase2.time = pd.to_datetime(dfL_phase2.time)

for i in range(len(t1)-1):
    print(i,t1[i])
    ind = (dfL_phase1.time>=t1[i]) & (dfL_phase1.time<=t1[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase1_df[cols].loc[stab_phase1_df.time==t1[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph1.loc[ind] = aux.values
    
for i in range(len(t2)-1):
    print(i,t2[i])
    ind = (dfL_phase2.time>=t2[i]) & (dfL_phase2.time<=t2[i+1])
    if ind.sum()>0:
        print(ind.sum())
        aux = stab_phase2_df[cols].loc[stab_phase2_df.time==t2[i]]
        aux = pd.concat([aux]*ind.sum(), ignore_index=True)
        Lph2.loc[ind] = aux.values

dfL_phase1.time = aux1 
dfL_phase2.time = aux2

# In[]
dfL_phase1[cols] = Lph1
dfL_phase2[cols] = Lph2
dfL_phase1.sort_values(by=['time'], inplace = True)
dfL_phase2.sort_values(by=['time'], inplace = True)

ind1 = drel_phase1.time0.isin(dfL_phase1.time)
timein = pd.concat([drel_phase1.time0.loc[ind1],drel_phase1.relscan.loc[ind1]],axis = 1)
timein.index = dfL_phase1.index
dfL_phase1[['time0','rel']] = timein

ind1 = drel_phase2.time0.isin(dfL_phase2.time)
timein = pd.concat([drel_phase2.time0.loc[ind1],drel_phase2.relscan.loc[ind1]],axis = 1)
timein.index = dfL_phase2.index
dfL_phase2[['time0','rel']] = timein

# In[Plots]
features1 = dfL_phase1.columns
dfL_phase1_L = dfL_phase1.copy()

dfL_phase1_L['sign'] = np.sign(dfL_phase1[features1[26]].values)
features1 = dfL_phase1_L.columns[[-2,2,3,4,5,6,25,-1]]#[[2, 3, 4, 5, -6, -5]]

#ind, labels1 = clust_stab(dfL_phase1_L,features1,nn=5,plot=True)

#ind, labels1 = clust_stab_optics(dfL_phase1_L,features1,nn=5,plot=True)

dfL_phase1_L['label'] = np.ones(len(dfL_phase1_L)) 
dfL_phase1_L['label'].loc[ind] = labels1 
elem = [np.sum(labels1==l) for l in np.unique(labels1)]
clust1 = np.unique(labels1)[np.flip(np.argsort(elem))][:2]
###############################################################################
# In[]
features2 = dfL_phase2.columns
dfL_phase2_L = dfL_phase2.copy()

dfL_phase2_L['sign'] = np.sign(dfL_phase2[features2[26]].values)
features2 = dfL_phase2_L.columns[[-2,2,3,4,5,6,25,-1]]

#ind, labels2 = clust_stab(dfL_phase2_L,features2,nn=5,plot=True)

#ind, labels2 = clust_stab_optics(dfL_phase2_L,features2,nn=5,plot=True)

dfL_phase2_L['label'] = np.ones(len(dfL_phase2_L)) 
dfL_phase2_L['label'].loc[ind] = labels2 
elem = [np.sum(labels2==l) for l in np.unique(labels2)]
clust2 = np.unique(labels2)[np.flip(np.argsort(elem))][:2]
###############################################################################
# In[]
#rel1_0 = ((dfL_phase1_L.area_frac>.5*dfL_phase1_L.area_frac.max()) & (dfL_phase1_L['label']==clust1[0])) & dfL_phase1_L['rel']>.25
#rel1_1 = ((dfL_phase1_L.area_frac>.5*dfL_phase1_L.area_frac.max()) & (dfL_phase1_L['label']==clust1[1])) & dfL_phase1_L['rel']>.25
#
#rel2_0 = ((dfL_phase2_L.area_frac>.5*dfL_phase2_L.area_frac.max()) & (dfL_phase2_L['label']==clust2[0])) & dfL_phase2_L['rel']>.25
#rel2_1 = ((dfL_phase2_L.area_frac>.5*dfL_phase2_L.area_frac.max()) & (dfL_phase2_L['label']==clust2[1])) & dfL_phase2_L['rel']>.25
colL = '$L_{flux,103}$'
rel1_0 = (dfL_phase1_L['rel']>.25) & ((dfL_phase1_L[colL]>=0) & (dfL_phase1_L[colL]<500))
rel1_1 = (dfL_phase1_L['rel']>.25) & ((dfL_phase1_L[colL]<=0) & (dfL_phase1_L[colL]>-500))
rel1_2 = (dfL_phase1_L['rel']>.25) & (dfL_phase1_L[colL].abs()>=500)

rel2_0 = (dfL_phase2_L['rel']>.25) & ((dfL_phase2_L[colL]>=0) & (dfL_phase2_L[colL]<500))
rel2_1 = (dfL_phase2_L['rel']>.25) & ((dfL_phase2_L[colL]<=0) & (dfL_phase2_L[colL]>-500))
rel2_2 = (dfL_phase2_L['rel']>.25) & (dfL_phase2_L[colL].abs()>=500)

xlim = 1800
ylim = 1200
######################################################################################
#Phase 1

g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase1.loc[rel1_0])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:stable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_stable_phase_1.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase1.loc[rel1_1])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:unstable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_unstable_phase_1.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase1.loc[rel1_2])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:neutral$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_neutral_phase_1.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase1.loc[rel1_0])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:stable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_stable_phase_1.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase1.loc[rel1_1])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:unstable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_unstable_phase_1.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase1.loc[rel1_2])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:1,\:neutral$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_neutral_phase_1.png')
######################################################################################
# Phase 2

g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase2.loc[rel2_0])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:stable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_stable_phase_2.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase2.loc[rel2_1])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:unstable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_unstable_phase_2.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{u,y}$', data=dfL_phase2.loc[rel2_2])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{u,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:neutral$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lu_y_neutral_phase_2.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase2.loc[rel2_0])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:stable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_stable_phase_2.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase2.loc[rel2_1])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:unstable$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_unstable_phase_2.png')
######################################################################################
g = sns.JointGrid(x='$L_{u,x}$', y='$L_{v,y}$', data=dfL_phase2.loc[rel2_2])
g = g.plot_joint(sns.kdeplot, kernel='epa', shade=True, levels=10, gridsize=100,
             clip = [(0,xlim), (0,ylim)], cmap="Greys")
plt.xlabel('$L_{u,x}$', fontsize=18)
plt.ylabel('$L_{v,y}$', fontsize=18)
plt.xlim(0,xlim)
plt.ylim(0,ylim)
g = g.plot_marginals(sns.distplot, kde=True, color="grey")
g.fig.suptitle('$Phase\:2,\:neutral$',fontsize=16, x= .7)
plt.tight_layout()
plt.savefig(file_out_figures+'/Lu_x_Lv_y_neutral_phase_2.png')

# In[]
T_0 = ['T_241m_LMS','T_175m_LSM','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']

i, j = 2, 1

tabs0_0 = stab_phase2[Tabs_0[i]].values
tabs1_0 = stab_phase2[Tabs_0[j]].values
 
t0_0 = stab_phase2[T_0[i]].values
t1_0 = stab_phase2[T_0[j]].values
x0_0 = stab_phase2[X_0[i]].values
x1_0 = stab_phase2[X_0[j]].values
y0_0 = stab_phase2[Y_0[i]].values
y1_0 = stab_phase2[Y_0[j]].values
s0_0 = stab_phase2[Sspeed_0[i]].values 
s1_0 = stab_phase2[Sspeed_0[j]].values
z0_0 = heights[i]
z1_0 = heights[j]
p0_0 = stab_phase2[P_0[1]].values
p1_0 = stab_phase2[P_0[1]].values 
w0_0 = stab_phase2[Z_0[i]].values 
w1_0 = stab_phase2[Z_0[j]].values

u0_0 = np.sqrt(x0_0**2+y0_0**2)
u1_0 = np.sqrt(x1_0**2+y1_0**2)
theta0_0 = t0_0*(p0_0/1000)**.286
theta1_0 = t1_0*(p1_0/1000)**.286
theta0_a_0 = tabs0_0*(p0_0/1000)**.286
theta1_a_0 = tabs1_0*(p1_0/1000)**.286
dtheta_0 = theta1_0 - theta0_0
dtheta_a_0 = theta1_a_0 - theta0_a_0

res_grad2_0 = stability_grad(t0_0, t1_0, x0_0, x1_0, y0_0, y1_0, z0_0, z1_0, p0_0, p1_0) 
res_bulk2_0 = stability_bulk(t0_0, t1_0, x1_0, y1_0, z0_0, z1_0, p0_0, p1_0)
res_grad2_0_a = stability_grad(tabs0_0, tabs1_0, x0_0, x1_0, y0_0, y1_0, z0_0, z1_0, p0_0, p1_0) 
res_bulk2_0_a = stability_bulk(tabs0_0, tabs1_0, x1_0, y1_0, z0_0, z1_0, p0_0, p1_0)

tabs0_1 = stab_phase1[Tabs_1[i]].values
tabs1_1 = stab_phase1[Tabs_1[j]].values
    
t0_1 = stab_phase2[T_1[i]].values
t1_1 = stab_phase2[T_1[j]].values
x0_1 = stab_phase2[X_1[i]].values
x1_1 = stab_phase2[X_1[j]].values
y0_1 = stab_phase2[Y_1[i]].values
y1_1 = stab_phase2[Y_1[j]].values
s0_1 = stab_phase2[Sspeed_1[i]].values 
s1_1 = stab_phase2[Sspeed_1[j]].values
z0_1 = heights[i]
z1_1 = heights[j]
p0_1 = stab_phase2[P_1[1]].values
p1_1 = stab_phase2[P_1[1]].values
w0_1 = stab_phase2[Z_0[i]].values 
w1_1 = stab_phase2[Z_0[j]].values    

u0_1 = np.sqrt(x0_1**2+y0_1**2)
u1_1 = np.sqrt(x1_1**2+y1_1**2)
theta0_1 = t0_1*(p0_1/1000)**.286
theta1_1 = t1_1*(p1_1/1000)**.286
dtheta_1 = theta1_1 - theta0_1

lab_t_0 = ['$\theta_{241m,\:South}$','$\theta_{175m,\:South}$','$\theta_{103m,\:South}$',
           '$\theta_{37m,\:South}$','$\theta_{7m,\:South}$']
lab_t_1 = ['$\theta_{241m,\:North}$','$\theta_{175m,\:North}$','$\theta_{103m,\:North}$',
           '$\theta_{37m,\:North}$','$\theta_{7m,\:North}$']

lab_u_0 = ['$u_{241m,\:South}$','$u_{175m,\:South}$','$u_{103m,\:South}$',
           '$u_{37m,\:South}$','$u_{7m,\:South}$']
lab_u_1 = ['$u_{241m,\:North}$','$u_{175m,\:North}$','$u_{103m,\:North}$',
           '$u_{37m,\:North}$','$u_{7m,\:North}$']

lab_a_0 = ['$tabs_{241m,\:South}$','$tabs_{175m,\:South}$','$tabs_{103m,\:South}$',
           '$tabs_{37m,\:South}$','$tabs_{7m,\:South}$']
lab_a_1 = ['$tabs_{241m,\:North}$','$tabs_{175m,\:North}$','$tabs_{103m,\:North}$',
           '$tabs_{37m,\:North}$','$tabs_{7m,\:North}$']

lab_s_0 = ['$s_{241m,\:South}$','$s_{175m,\:South}$','$s_{103m,\:South}$',
           '$s_{37m,\:South}$','$s_{7m,\:South}$']
lab_s_1 = ['$s_{241m,\:North}$','$s_{175m,\:North}$','$s_{103m,\:North}$',
           '$s_{37m,\:North}$','$s_{7m,\:North}$']

plt.figure()
plt.plot(theta0_0,'--', label = lab_t_0[i])
plt.plot(theta0_1,label = lab_t_1[i])
plt.plot(theta1_0,'--', label = lab_t_0[j])
plt.plot(theta1_1,label = lab_t_1[j])
plt.legend()

plt.figure()
plt.plot(tabs0_0,'--', label = lab_a_0[i])
plt.plot(tabs0_1,label = lab_a_1[i])
plt.plot(tabs1_0,'--', label = lab_a_0[j])
plt.plot(tabs1_1,label = lab_a_1[j])
plt.legend()

plt.figure()
plt.plot(u0_0,'--', label = lab_u_0[i])
plt.plot(u0_1,label = lab_u_1[i])
plt.plot(u1_0,'--', label = lab_u_0[j])
plt.plot(u1_1,label = lab_u_1[j])
plt.legend()

plt.figure()
plt.plot(s0_0,'--', label = lab_s_0[i])
plt.plot(s0_1,label = lab_s_1[i])
plt.plot(s1_0,'--', label = lab_s_0[j])
plt.plot(s1_1,label = lab_s_1[j])
plt.legend()

plt.figure()
plt.plot(u0_0,'--', label = lab_u_0[i])
plt.plot(s0_0,label = lab_s_0[i])
plt.legend()

plt.figure()
plt.plot(np.abs(w0_0),'--', label = 'abs w')
plt.plot(w0_0,'--', label = 'w')
plt.plot(s0_0-u0_0,label = 's-u')
plt.plot(np.sqrt(s0_0**2-u0_0**2),label = 'sqrt(s**2+u**2)')
plt.legend()

plt.figure()
plt.plot(res_grad2_0[:,1], label = 'Lgrad')
plt.plot(res_grad2_0_a[:,1],label = 'Lgrada')
plt.legend()
    
plt.figure()
plt.plot(res_bulk2_0[:,1], label = 'Lbulk')
plt.plot(res_bulk2_0_a[:,1],label = 'Lbulk')
plt.legend()
    
    
#####################

i0 = 0
T_0 = ['T_241m_LSM','T_175m_LSM','T_103m_LMS','T_37m_LMS','T_7m_LMS']
T_1 = ['T_241m_LMN','T_175m_LMN','T_103m_LMN','T_37m_LMN','T_7m_LMN']
Z_0 = ['Z_241m_LMS','Z_175m_LMS','Z_103m_LMS','Z_37m_LMS','Z_7m_LMS']
Z_1 = ['Z_241m_LMN','Z_175m_LMN','Z_103m_LMN','Z_37m_LMN','Z_7m_LMN']

k = .4
g = 9.806

N = 6
nz = 4
t = pd.to_datetime([str(int(n)) for n in np.array(res_flux_1)[:,nz,0] ]).values
U = mov_con(np.array(res_flux_1)[:,nz,1], N)
V = mov_con(np.array(res_flux_1)[:,nz,2], N)
W = mov_con(np.array(res_flux_1)[:,nz,3], N)
T = mov_con(np.array(res_flux_1)[:,nz,4], N)
uu = mov_con(np.array(res_flux_1)[:,nz,5], N)
uv = mov_con(np.array(res_flux_1)[:,nz,6], N)
uw = mov_con(np.array(res_flux_1)[:,nz,7], N)
vv = mov_con(np.array(res_flux_1)[:,nz,8], N)
vw = mov_con(np.array(res_flux_1)[:,nz,9], N)
ww = mov_con(np.array(res_flux_1)[:,nz,10], N)
wth = mov_con(np.array(res_flux_1)[:,nz,11], N)

u_fric0 = (np.array(res_flux_1)[:,nz,7]**2+np.array(res_flux_1)[:,nz,9]**2)**.25
u_fric = (uw**2+vw**2)**.25

wth0 = np.array(res_flux_1)[:,nz,11]
T0 = np.array(res_flux_1)[:,nz,4]

L0 = -u_fric0**3*T0/(k*g*wth0)
L = -u_fric**3*T/(k*g*wth)
L0[L0>500] = 500
L0[L0<-500] = -500
L[L>500] = 500
L[L<-500] = -500
plt.figure()
plt.plot(t,L0)
plt.plot(t,np.sign(L))

plt.hist(L,bins=500)


#fluxes = [int(name),U, V, W, T, uu, uv, uw, vv, vw, ww, wth, z]
#u_fric = (np.nanmean(u*w)**2+np.nanmean(v*w)**2)**.25

indz = [0,1,2,4]
zf=3
u_profile = stab_20hz[Sspeed_0].mean().values[indz]
alpha = .2
z = np.array(heights)[indz]
zref = np.array(heights)[0]
func0 = lambda alpha: np.abs(np.sum(u_profile - u_profile[0]*(z/zref)**alpha))
func1 = lambda us: np.abs(np.sum(u_profile-(us/.4)*np.log(z/zf)))
from scipy.optimize import fsolve
alpha =  fsolve(func0, alpha)
us = fsolve(func1, .36)

plt.plot(u_profile,z,'o', color='r')
zt = np.linspace(z[0],z[-1],10)
plt.plot(u_profile[0]*(zt/zref)**alpha,zt)
plt.plot((us/.4)*np.log(zt/zf),zt)

# In[]




















# In[Clustering]
from sklearn.neighbors import KDTree
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D

def clust_stab(df,features,nn=5,plot=False):
    # Initialization of input array of high-dimensional datapoints for DBSCAN
    a = df[features[0]].values.shape 
    X = np.empty((a[0],len(features)))   
    # Data preparation
    # If input has nan values
    ind = np.ones(a)==1
    print(ind.shape)
    for f in features:
        ind = ind & ~np.isnan(df[f].values.flatten())
        
    X = np.empty((sum(ind),len(features)))
    # DBSCAN input filling with data form df
    for i,f in enumerate(features):
            X[:,i] = df[f].values.flatten()[ind]
    # After input creation, it is re-scaled to non-dimensional space.
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
    # Tree creation for k-distance graph
    tree_X = KDTree(X)
    # first k distance estimation
    d,i = tree_X.query(tree_X.data,k = nn)  
    #k-nearest distance
    d=d[:,-1]
    #non-zero values
    d = d[d>0]
    # log transformation to level-up values and easier identification of "knees"
    # d is an array with k-distances sorted in increasing value.
    d = np.log(np.sort(d))
    # x axis (point label)
    l = np.arange(0,len(d)) 
    rsmpl = 200
    # Down sampling to speed up calculations
    # same with point lables
    l_resample = np.unique(np.r_[l[::int(len(l)/rsmpl)],l[-1]])
#    print(l_resample,l_resample.shape)
    d_resample = d[l_resample]
    # Cubic spline of resampled k-distances, lower memory usage and higher calculation speed.
    #spl = UnivariateSpline(l_resample, d_resample,s=0.5)
    std=.001*np.ones_like(d_resample)
    # Changes in slope in the sorted, log transformed, k-distance graph
    t = np.arange(l_resample.shape[0])    
    fx = UnivariateSpline(t, l_resample/(-l_resample[0]+l_resample[-1]), k=4, w=1/std)
    fy = UnivariateSpline(t, d_resample/(-d_resample[0]+d_resample[-1]), k=4, w=1/std)
    x_1prime = fx.derivative(1)(t)
    x_2prime = fx.derivative(2)(t)
    y_1prime = fy.derivative(1)(t)
    y_2prime = fy.derivative(2)(t) 
    kappa = (x_1prime* y_2prime - y_1prime* x_2prime) / np.power(x_1prime**2 + y_1prime**2, 1.5)
    # location of knee (first point with k-distance above 1 std of k-distance mean)
    #ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
    ind_kappa, _ = find_peaks(kappa,prominence=1) 
    # Just after half of the graph
    ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
    # The first knee...
    l1 = l_resample[ind_kappa][0]
    # the corresponding eps distance
    eps = np.exp(d[l1])
    clf = []
    clf = DBSCAN(eps=eps, min_samples=nn)
    clf.fit(X)
    # Cluster-core samples and boundary samples identification
    core_samples_mask = np.zeros_like(clf.labels_, dtype=bool)
    core_samples_mask[clf.core_sample_indices_] = True
    # Array with labels identifying the different clusters in the data
    labels = clf.labels_
    # Number of identified clusters (excluding noise)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
    # Informative plots after clustering:
    #       - k-distance plot with eps value using both criterias, and the average value.
    #       - Scatter plot in 3 dimensions (more features are not shown)  
    if plot:        
        fig, ax = plt.subplots(figsize = (8,8))
        ax.plot(d,label = '$Data$', c = 'k')
        ax.plot([0,X.shape[0]],[np.log(eps),np.log(eps)],'--', c = 'red',
                label = r'$\log(\varepsilon_{knee})$')
        ax.set_xlabel('$Data\:point$', fontsize=16, weight='bold')
        ax.set_ylabel('$log(k-dist)$', fontsize=16, weight='bold')
        ax.set_xlim([-X.shape[0]*.03,X.shape[0]*1.03])
        ax.legend(loc=(.15,.8),fontsize=16)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
        ax.annotate('$Knee$', xy=(l1, d[l1]), xytext=(10000, 0), fontsize = 16,
            arrowprops=dict(facecolor='grey',arrowstyle = "->"),)
        ax.tick_params(labelsize = 16)
        fig.tight_layout()

        for i,f in enumerate(features):
            X[:,i] = df[f].values.flatten()[ind]
        unique_labels = set(labels)
        colors = [plt.cm.Set1(each)
              for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure(figsize = (16,16))
        f = 16
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(30, 50)
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.set_xlabel(features[0], fontsize=f, weight='bold')
        ax.set_ylabel(features[1], fontsize=f, weight='bold')
        ax.set_zlabel(features[2], fontsize=f, weight='bold')
        ax.text(-20, 7000,250, '(b)', transform=ax.transAxes, fontsize=32,
        verticalalignment='top')
        for k, col in zip(unique_labels, colors):
                al = 1
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    al = .3
                class_member_mask = (labels == k)
                xy = X[class_member_mask & core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                           s=20, alpha=al)
                xy = X[class_member_mask & ~core_samples_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                           s=6, alpha = al)  
    # First reliable value and the corresponding cluster to which it belongs
    # 2-D array
    labels = labels
    return (ind,labels)
# In[]
from sklearn.cluster import OPTICS, cluster_optics_dbscan
    
def clust_stab_optics(df,features,nn=5,plot=False):
    # Initialization of input array of high-dimensional datapoints for DBSCAN
    a = df[features[0]].values.shape 
    X = np.empty((a[0],len(features)))   
    # Data preparation
    # If input has nan values
    ind = np.ones(a)==1
    print(ind.shape)
    for f in features:
        ind = ind & ~np.isnan(df[f].values.flatten())
        
    X = np.empty((sum(ind),len(features)))
    # DBSCAN input filling with data form df
    for i,f in enumerate(features):
            X[:,i] = df[f].values.flatten()[ind]
    # After input creation, it is re-scaled to non-dimensional space.
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
    # Tree creation for k-distance graph
    tree_X = KDTree(X)
    # first k distance estimation
    d,i = tree_X.query(tree_X.data,k = nn)  
    #k-nearest distance
    d=d[:,-1]
    #non-zero values
    d = d[d>0]
    # log transformation to level-up values and easier identification of "knees"
    # d is an array with k-distances sorted in increasing value.
    d = np.log(np.sort(d))
    # x axis (point label)
    l = np.arange(0,len(d)) 
    rsmpl = 200
    # Down sampling to speed up calculations
    # same with point lables
    l_resample = np.unique(np.r_[l[::int(len(l)/rsmpl)],l[-1]])
#    print(l_resample,l_resample.shape)
    d_resample = d[l_resample]
    # Cubic spline of resampled k-distances, lower memory usage and higher calculation speed.
    #spl = UnivariateSpline(l_resample, d_resample,s=0.5)
    std=.001*np.ones_like(d_resample)
    # Changes in slope in the sorted, log transformed, k-distance graph
    t = np.arange(l_resample.shape[0])    
    fx = UnivariateSpline(t, l_resample/(-l_resample[0]+l_resample[-1]), k=4, w=1/std)
    fy = UnivariateSpline(t, d_resample/(-d_resample[0]+d_resample[-1]), k=4, w=1/std)
    x_1prime = fx.derivative(1)(t)
    x_2prime = fx.derivative(2)(t)
    y_1prime = fy.derivative(1)(t)
    y_2prime = fy.derivative(2)(t) 
    kappa = (x_1prime* y_2prime - y_1prime* x_2prime) / np.power(x_1prime**2 + y_1prime**2, 1.5)
    # location of knee (first point with k-distance above 1 std of k-distance mean)
    #ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
    ind_kappa, _ = find_peaks(kappa,prominence=1) 
    # Just after half of the graph
    ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
    # The first knee...
    l1 = l_resample[ind_kappa][0]
    # the corresponding eps distance
    eps = np.exp(d[l1])
    clf = []
    
    clf = OPTICS(min_samples=5, xi=.05, max_eps = eps, min_cluster_size=.05)
    clf.fit(X)
    # Array with labels identifying the different clusters in the data
    labels = clf.labels_
    # Number of identified clusters (excluding noise)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
    # Informative plots after clustering:
    #       - k-distance plot with eps value using both criterias, and the average value.
    #       - Scatter plot in 3 dimensions (more features are not shown)  
    if plot:        
        fig, ax = plt.subplots(figsize = (8,8))
        ax.plot(d,label = '$Data$', c = 'k')
        ax.plot([0,X.shape[0]],[np.log(eps),np.log(eps)],'--', c = 'red',
                label = r'$\log(\varepsilon_{knee})$')
        ax.set_xlabel('$Data\:point$', fontsize=16, weight='bold')
        ax.set_ylabel('$log(k-dist)$', fontsize=16, weight='bold')
        ax.set_xlim([-X.shape[0]*.03,X.shape[0]*1.03])
        ax.legend(loc=(.15,.8),fontsize=16)
        ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
        ax.annotate('$Knee$', xy=(l1, d[l1]), xytext=(10000, 0), fontsize = 16,
            arrowprops=dict(facecolor='grey',arrowstyle = "->"),)
        ax.tick_params(labelsize = 16)
        fig.tight_layout()

        for i,f in enumerate(features):
            X[:,i] = df[f].values.flatten()[ind]
        unique_labels = set(labels)
        colors = [plt.cm.Set1(each)
              for each in np.linspace(0, 1, len(unique_labels))]
        fig = plt.figure(figsize = (16,16))
        f = 16
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(30, 50)
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.set_xlabel(features[0], fontsize=f, weight='bold')
        ax.set_ylabel(features[1], fontsize=f, weight='bold')
        ax.set_zlabel(features[2], fontsize=f, weight='bold')
        ax.text(-20, 7000,250, '(b)', transform=ax.transAxes, fontsize=32,
        verticalalignment='top')
        for k, col in zip(unique_labels, colors):
                al = 1
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    al = .3
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                           s=20, alpha=al)
    # First reliable value and the corresponding cluster to which it belongs
    # 2-D array
    labels = labels
    return (ind,labels)


#np.c_[Ri, L, stab,theta_mean,dudz,uw,wth]
# In[Good quality scans]
#csv_database_0_ph1 = create_engine('sqlite:///'+file_in_path_0_ph1+'/raw_filt_0.db')
#csv_database_1_ph1 = create_engine('sqlite:///'+file_in_path_1_ph1+'/raw_filt_1.db')
#csv_database_0_ph2 = create_engine('sqlite:///'+file_in_path_0_ph2+'/raw_filt_0.db')
#csv_database_1_ph2 = create_engine('sqlite:///'+file_in_path_1_ph2+'/raw_filt_1.db')
#
#iden_lab = np.array(['stop_time','azim'])
#labels = iden_lab
#labels_ws = []
#labels_rg = []
#labels_CNR = []
#for i in np.arange(198):
#    labels = np.concatenate((labels,np.array(['ws','range_gate','CNR'])))
#    labels_ws = np.concatenate((labels_ws,np.array(['ws_'+str(i)])))
#    labels_rg = np.concatenate((labels_rg,np.array(['range_gate_'+str(i)])))
#    labels_CNR = np.concatenate((labels_CNR,np.array(['CNR_'+str(i)])))
#labels = np.concatenate((labels,np.array(['scan'])))
#
#col = 'SELECT '
#i = 0
#for w,r,c in zip(labels_ws,labels_rg,labels_CNR):
#    if i == 0:
#        col = col + 'stop_time,' + ' azim, ' + w + ', ' + r + ', ' + c + ', '
#    elif (i == len(labels_ws)-1):
#        col = col + ' ' + w + ', ' + r + ', ' + c + ', scan'
#    else:
#        col = col + ' ' + w + ', ' + r + ',' + c + ', '        
#    i+=1
#
#rel_scan_0_ph1 = []
#rel_scan_1_ph1 = []
#rel_scan_0_ph2 = []
#rel_scan_1_ph2 = []
#
#scann = int(62*3600/45)
#chunksize = 45*scann
#tot_scans = 55000
#tot_iter = int(tot_scans*45/chunksize)
#off = 0
#lim = [-10,-24]
#selec_fil = col + ' FROM "table_fil"'
#
#for i in range(tot_iter):
#    print(off/45,'Phase 1')
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_0_ph1)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_cnr = np.sum(((df.CNR >-24) & (df.CNR < -8)).values, axis = 1) 
#    rel_scan = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
#    times = np.array([date+timedelta(seconds = df.loc[df.scan==s].stop_time.min()) for s in df.scan.unique()])
#    rel_scan_0_ph1.append(np.c_[rel_scan,scans,times])
#    df = []
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_1_ph1)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_cnr = np.sum(((df.CNR >-24) & (df.CNR < -8)).values, axis = 1) 
#    rel_scan = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
#    times = np.array([date+timedelta(seconds = df.loc[df.scan==s].stop_time.min()) for s in df.scan.unique()])
#    rel_scan_1_ph1.append(np.c_[rel_scan,scans,times])    
#    df = []
#    
#    print(off/45,'Phase 2')   
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_0_ph2)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_cnr = np.sum(((df.CNR >-24) & (df.CNR < -8)).values, axis = 1) 
#    rel_scan = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
#    times = np.array([date+timedelta(seconds = df.loc[df.scan==s].stop_time.min()) for s in df.scan.unique()])
#    rel_scan_0_ph2.append(np.c_[rel_scan,scans,times])  
#    df = []
#    
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_1_ph2)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_cnr = np.sum(((df.CNR >-24) & (df.CNR < -8)).values, axis = 1) 
#    rel_scan = ind_cnr.reshape((int(len(ind_cnr)/chunk),chunk)).sum(axis=1)/rt
#    times = np.array([date+timedelta(seconds = df.loc[df.scan==s].stop_time.min()) for s in df.scan.unique()])
#    rel_scan_1_ph2.append(np.c_[rel_scan,scans,times])    
#    df = []
#    
#    off+=chunksize 

#with open(file_in_path_0_ph1+'/rel_scan_0_ph1.pkl', 'wb') as writer:
#    pickle.dump(rel_scan_0_ph1,writer)
#with open(file_in_path_1_ph1+'/rel_scan_1_ph1.pkl', 'wb') as writer:
#    pickle.dump(rel_scan_1_ph1,writer)   
#with open(file_in_path_0_ph2+'/rel_scan_0_ph2.pkl', 'wb') as writer:
#    pickle.dump(rel_scan_0_ph2,writer)
#with open(file_in_path_1_ph2+'/rel_scan_1_ph2.pkl', 'wb') as writer:
#    pickle.dump(rel_scan_1_ph2,writer)

# In[]
#df1 = pd.read_sql_query(selec_fil+' LIMIT 45*4 offset 30000*45', csv_database_1_ph1)
#df0 = pd.read_sql_query(selec_fil+' LIMIT 45*4 offset 30000*45', csv_database_0_ph1)
#t0 = np.array([date+timedelta(seconds = df0.loc[df0.scan==s].stop_time.min()) for s in df0.scan.unique()])
#t1 = np.array([date+timedelta(seconds = df1.loc[df1.scan==s].stop_time.min()) for s in df1.scan.unique()])    
#print(t1[0]-t0[0],t0[0]-t1[0])
#df0.columns = labels_short
#df1.columns = labels_short
#U_out, V_out, grd, _ = wr.direct_wf_rec(df0.astype(np.float32), df1.astype(np.float32), tri, d,N_grid = 512)
#for i in range(len(U_out)):
#    plt.figure()
#    plt.contourf(U_out[i],cmap='jet')
#from datetime import datetime, timedelta    
#date = datetime(1904, 1, 1)       
#scann = int(62*3600/45)
#chunksize = 45*scann
#tot_scans = 25000
#tot_iter = int(tot_scans*45/chunksize)
#off = 0
#lim = [-10,-24]
#selec_fil = col + ' FROM "table_fil"'
#
#fil_scan_0_ph1 = []
#fil_scan_1_ph1 = []
#fil_scan_0_ph2 = []
#fil_scan_1_ph2 = []
#
#for i in range(tot_iter):
#    print(off/45,'Phase 1')
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_0_ph1)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    times = np.array([date+timedelta(seconds = df.loc[df.scan==s].stop_time.min()) for s in df0.scan.unique()])
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_fil = np.sum(np.isnan(df.ws.values), axis = 1) 
#    
#    fil_scan = ind_fil.reshape((int(len(ind_fil)/chunk),chunk)).sum(axis=1)/rt
#    fil_scan_0_ph1.append(np.c_[fil_scan,scans,times])
#    df = []
#
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_1_ph1)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_fil = np.sum(np.isnan(df.ws.values), axis = 1) 
#    fil_scan = ind_fil.reshape((int(len(ind_fil)/chunk),chunk)).sum(axis=1)/rt
#    fil_scan_1_ph1.append(np.c_[fil_scan,scans])
#    df = []
#    
#    print(off/45,'Phase 2')
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_0_ph2)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_fil = np.sum(np.isnan(df.ws.values), axis = 1) 
#    fil_scan = ind_fil.reshape((int(len(ind_fil)/chunk),chunk)).sum(axis=1)/rt
#    fil_scan_0_ph2.append(np.c_[fil_scan,scans])
#    df = []
#    
#    df = pd.read_sql_query(selec_fil+' LIMIT ' + str(int(chunksize)) +
#                           ' OFFSET ' + str(int(off)), csv_database_1_ph2)
#    df.columns = labels 
#    chunk = len(df.azim.unique())
#    scans = df.scan.unique()
#    rt = df.loc[df.scan==off/45].ws.values.flatten().shape[0]
#    ind_fil = np.sum(np.isnan(df.ws.values), axis = 1) 
#    fil_scan = ind_fil.reshape((int(len(ind_fil)/chunk),chunk)).sum(axis=1)/rt
#    fil_scan_1_ph2.append(np.c_[fil_scan,scans])
#    df = []
#    
#with open(file_in_path_0_ph1+'/fil_scan_0_ph1.pkl', 'wb') as writer:
#    pickle.dump(fil_scan_0_ph1,writer)
#with open(file_in_path_1_ph1+'/fil_scan_1_ph1.pkl', 'wb') as writer:
#    pickle.dump(fil_scan_1_ph1,writer)   
#with open(file_in_path_0_ph2+'/fil_scan_0_ph2.pkl', 'wb') as writer:
#    pickle.dump(fil_scan_0_ph2,writer)
#with open(file_in_path_1_ph2+'/fil_scan_1_ph2.pkl', 'wb') as writer:
#    pickle.dump(fil_scan_1_ph2,writer)

#    
## In[] 
#    
#with open(file_in_path_0_ph1+'/rel_scan_0_ph1.pkl', 'rb') as reader:
#    rel_scan_0_ph1 = pickle.load(reader)
#with open(file_in_path_1_ph1+'/rel_scan_1_ph1.pkl', 'rb') as reader:
#    rel_scan_1_ph1 = pickle.load(reader)   
#with open(file_in_path_0_ph2+'/rel_scan_0_ph2.pkl', 'rb') as reader:
#    rel_scan_0_ph2 = pickle.load(reader)
#with open(file_in_path_1_ph2+'/rel_scan_1_ph2.pkl', 'rb') as reader:
#    rel_scan_1_ph2 = pickle.load(reader)    
#        
#rel_scan_0_ph1 = np.vstack(rel_scan_0_ph1)[:10000,:]
#rel_scan_1_ph1 = np.vstack(rel_scan_1_ph1)[:10000,:]
#rel_scan_0_ph2 = np.vstack(rel_scan_0_ph2)[:10000,:]
#rel_scan_1_ph2 = np.vstack(rel_scan_1_ph2)[:10000,:]
#
## In[]
#
#with open(file_in_path_0_ph1+'/fil_scan_0_ph1.pkl', 'rb') as reader:
#    fil_scan_0_ph1 = pickle.load(reader)
#with open(file_in_path_1_ph1+'/fil_scan_1_ph1.pkl', 'rb') as reader:
#    fil_scan_0_ph1 = pickle.load(reader)   
#with open(file_in_path_0_ph2+'/fil_scan_0_ph2.pkl', 'rb') as reader:
#    fil_scan_0_ph2 = pickle.load(reader)
#with open(file_in_path_1_ph2+'/fil_scan_1_ph2.pkl', 'rb') as reader:
#    fil_scan_1_ph2 = pickle.load(reader)    
#        
#fil_scan_0_ph1 = np.vstack(fil_scan_0_ph1)
#fil_scan_1_ph1 = np.vstack(fil_scan_1_ph1)
#fil_scan_0_ph2 = np.vstack(fil_scan_0_ph2)
#fil_scan_1_ph2 = np.vstack(fil_scan_1_ph2)

#stab_phase1 = pd.DataFrame()
#stab_phase2 = pd.DataFrame()
#for stmp_1 in t_ph1:
#    print(stmp_1)
#    table = ' from calmeans '
#    where = 'where name = (select max(name)' + table + 'where name < ' + stmp_1 + ')'
#    sql_query_1 = 'select name, ' + ", ".join(T_0+T_1+X_0+X_1+Y_0+Y_1+P_0+P_1+Tabs_0+Tabs_1+Sspeed_0+Sspeed_1) + table +  where
#    stab_phase1 = pd.concat([stab_phase1,pd.read_sql_query(sql_query_1,
#                                                           osterild_database)],axis = 0)
#
#for stmp_2 in t_ph2:
#    print(stmp_2)
#    table = ' from calmeans '
#    where = 'where name = (select max(name)' + table + 'where name < ' + stmp_2 + ')'
#    sql_query_2 = 'select name, ' + ", ".join(T_0+T_1+X_0+X_1+Y_0+Y_1+P_0+P_1+Tabs_0+Tabs_1+Sspeed_0+Sspeed_1) + table +  where
#    stab_phase2 = pd.concat([stab_phase2,pd.read_sql_query(sql_query_2,
#                                                           osterild_database)],axis = 0) 

# In[]
#cols = ['name','$Ri_{grad}$','$L_{grad}$','$stab_{grad}$',
#               '$Ri_{bulk}$','$L_{bulk}$','$stab_{bulk}$',
#               '$Ri_{flux}$','$L_{flux}$','$stab_{flux}$']
#
#name = stab_phase1.name.values
#stab_phase1_df = pd.DataFrame(columns = cols,
#                              data = np.c_[name, res_grad1, res_bulk1, np.squeeze(np.array(res_flux_ph1))[:,:3]])
#name = stab_phase2.name.values
#stab_phase2_df = pd.DataFrame(columns = cols,
#                              data = np.c_[name, res_grad2, res_bulk2, np.squeeze(np.array(res_flux_ph2))[:,:3]])
#
#dfL_phase2[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase2.time.values),len(cols))))
#dfL_phase1[cols] = pd.DataFrame(columns=cols, data = np.nan*np.zeros((len(dfL_phase1.time.values),len(cols))))
#
#tph1 = pd.to_datetime(stab_phase1_df.name).values
#tph2 = pd.to_datetime(stab_phase2_df.name).sort_values().values
#
#for i in range(len(tph1)-1):
#    print(i,tph1[i])
#    ind = (pd.to_datetime(dfL_phase1.time)>=tph1[i]) & (pd.to_datetime(dfL_phase1.time)<=tph1[i+1])
#    dfL_phase1[cols].loc[ind] = stab_phase1_df[cols].loc[stab_phase1_df.name==tph1[i]]
#for i in range(len(tph2)-1):
#    print(i,tph2[i])
#    ind = (pd.to_datetime(dfL_phase2.time)>=tph2[i]) & (pd.to_datetime(dfL_phase2.time)<=tph2[i+1])
#    dfL_phase2[cols].loc[ind] = stab_phase2_df[cols].loc[stab_phase2_df.name==tph2[i]]

# In[]   
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/res_flux_phase1.pkl', 'rb') as reader:
#    res_flux_ph1 = pickle.load(reader)
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/res_flux_ph2.pkl', 'rb') as reader:
#    res_flux_ph2 = pickle.load(reader)
#
#stab_flux_ph1 = np.c_[stab_phase1.name.values,np.squeeze(np.array(res_flux_ph1))]
#stab_flux_ph2 = np.c_[stab_phase2.name.values,np.squeeze(np.array(res_flux_ph2))]   
#
#stab_phase1_df_flux = pd.DataFrame(columns = ['name','$Ri_{flux}$','$L_{flux}$','$stab_{flux}$'],
#                                   data = stab_flux_ph1[:,:4])
#stab_phase2_df_flux = pd.DataFrame(columns = ['name','$Ri_{flux}$','$L_{flux}$','$stab_{flux}$'],
#                                   data = stab_flux_ph2[:,:4])  
#
#ind = stab_phase1_df_flux['$L_{flux}$'].abs()>=500
#neutral = 500*stab_phase1_df_flux['$L_{flux}$'].loc[ind]/stab_phase1_df_flux['$L_{flux}$'].loc[ind].abs()
#stab_phase1_df_flux['$L_{flux}$'].loc[stab_phase1_df_flux['$L_{flux}$'].abs()>=500] = neutral
# In[]
#i0 = 0
#T_0 = ['T_241m_LSM','T_175m_LSM','T_103m_LMS','T_37m_LMS','T_7m_LMS']
#res_flux=[]
#for stmp_1 in t_ph1:
#    print(stmp_1,i0)
#    table_20Hz = ' from caldata_'+stmp_1[:4]+'_'+stmp_1[4:6]+'_20hz '
#    where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stmp_1 + ')'
#    sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0) + table_20Hz +  where
#    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
#    
#    table = ' from calmeans '
#    where = 'where name = (select max(name)' + table + 'where name < ' + stmp_1 + ')'
#    sql_query = 'select name, ' + ", ".join(P_0+P_1) + table +  where
#    stab_press = pd.read_sql_query(sql_query,osterild_database) 
#    
#    if len(stab_20hz)>0:
#        val_ind = valid_col(stab_20hz,[T_0,X_0,Y_0])
#        t0 = stab_20hz[T_0[val_ind[0][-2]]].values
#        t1 = stab_20hz[T_0[val_ind[0][0]]].values
#        x0 = stab_20hz[X_0[val_ind[1][-2]]].values
#        x1 = stab_20hz[X_0[val_ind[1][-3]]].values
#        y0 = stab_20hz[Y_0[val_ind[2][-2]]].values
#        y1 = stab_20hz[Y_0[val_ind[2][-3]]].values
#        z0 = heights[val_ind[2][-2]]
#        z1 = heights[val_ind[2][-3]]
#        w0 = stab_20hz[Z_0[val_ind[2][-3]]].values
#        p0, p1 = stab_press.values[0,1:3]
#    if len(t0)>0:
#        res_flux.append([stmp_1], stability_flux(t0,t1, x0, y0, x1, y1, w0, z0, z1, p0,p1))
#    i0 = i0+1
#    
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/res_flux_ph1.pkl', 'wb') as writer:
#    pickle.dump(res_flux,writer)
#
#i0 = 0
#T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']
#res_flux=[]    
#for stmp_2 in t_ph2[5112:]:
#    print(stmp_2,i0)
#    table_20Hz = ' from caldata_'+stmp_2[:4]+'_'+stmp_2[4:6]+'_20hz '
#    where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stmp_2 + ')'
#    sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0) + table_20Hz +  where
#    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
#    
#    table = ' from calmeans '
#    where = 'where name = (select max(name)' + table + 'where name < ' + stmp_2 + ')'
#    sql_query = 'select name, ' + ", ".join(P_0+P_1) + table +  where
#    stab_press = pd.read_sql_query(sql_query,osterild_database) 
#    
#    if len(stab_20hz)>0:
#        val_ind = valid_col(stab_20hz,[T_0,X_0,Y_0])
#        t0 = stab_20hz[T_0[val_ind[0][-2]]].values
#        t1 = stab_20hz[T_0[val_ind[0][0]]].values
#        x0 = stab_20hz[X_0[val_ind[1][-2]]].values
#        x1 = stab_20hz[X_0[val_ind[1][-3]]].values
#        y0 = stab_20hz[Y_0[val_ind[2][-2]]].values
#        y1 = stab_20hz[Y_0[val_ind[2][-3]]].values
#        z0 = heights[val_ind[2][-2]]
#        z1 = heights[val_ind[2][-3]]
#        w0 = stab_20hz[Z_0[val_ind[2][-3]]].values
#        p0, p1 = stab_press.values[0,1:3]
#    if len(t0)>0:
#        res_flux.append(stability_flux(t0,t1, x0, y0, x1, y1, w0, z0, z1, p0,p1))
#    i0 = i0+1
#    
#with open('E:/PhD/Python Code/Balcony/data_process/results/correlations'+'/res_flux_ph2.pkl', 'wb') as writer:
#    pickle.dump(res_flux,writer)     






    