# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:22:44 2020

@author: lalc
"""
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import scipy as sp
import joblib
import datetime
from datetime import datetime, timedelta

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
def datetimeDF(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

# In[This]
L_file = '/mnt/mimer/lalc/results/correlations/west/L_ph2_30min_2.pkl'

Lcorr, _ = joblib.load('/mnt/mimer/lalc/results/correlations/west/storeLph2.pkl')

osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')
times_phase = pd.to_datetime(Lcorr.time.values).sort_values()

times_stamp_ph = times_phase.strftime("%Y%m%d%H%M").values


# List of dates with pandas

init = times_phase.strftime("%Y%m%d").values[0]
end = (times_phase+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='30T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())

Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
Sdir_1 = ['Sdir_241m_LMN','Sdir_175m_LMN','Sdir_103m_LMN','Sdir_37m_LMN','Sdir_7m_LMN']
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

res_flux=[]


heights = [241, 175, 103, 37, 7]

#
for j,stamp in enumerate(t_ph):
    print(stamp,j)  
    if stamp>='201608010000':
        T_0 = ['T_241m_LMS','T_175m_LMS','T_103m_LMS','T_37m_LMS','T_7m_LMS']   
    table_20Hz = ' from caldata_'+stamp[:4]+'_'+stamp[4:6]+'_20hz '
    where = 'where name = (select max(name)' + table_20Hz + 'where name < ' + stamp + ')'
    sql_query = 'select ' + ", ".join(T_0+X_0+Y_0+Z_0, Sspeed_0, Sdir_0) + table_20Hz +  where
    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
    
    stab_20hz.fillna(value=pd.np.nan, inplace=True)
    
    table = ' from calmeans '
    where = 'where name = (select max(name)' + table + 'where name < ' + stamp + ')'
    sql_query = 'select name, ' + ", ".join(P_0) + table +  where
    stab_press = pd.read_sql_query(sql_query,osterild_database) 
   
    if len(stab_20hz)>0:
        L=[]
        for i in range(len(T_0)):
            t0 = stab_20hz[T_1[i]].values
            x0 = stab_20hz[X_1[i]].values
            y0 = stab_20hz[Y_1[i]].values
            z0 = heights[i]
            w0 = stab_20hz[Z_1[i]].values
            p1, p0 = stab_press.values[0,1:3]
            if len(t0)>0:
                L.append(fluxes(t0, x0, y0, w0, z0, p0, stamp))
        res_flux.append(L)    
    
joblib.dump(res_flux, L_file)
