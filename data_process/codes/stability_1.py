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
from scipy.interpolate import UnivariateSpline

# In[FUnctions for stability]
# Richarson number
# Bulk Richardson number
# Ri(zm) = ((g/theta)*(Delta z_m)/(Delta U)**2)ln(z_2/z_1)

# In[]
def fluxtot(resflux, resflux10 =[], heights =[], heights10=[], name =[], j=-2,N=6000):  
    heights=np.array(heights)
    #Data filtering anc correcting for 37m
    from sklearn import mixture
    # There are three posibilities:
        # 1 No offset
        # 2 Offset in the whole series
        # 3 Offset on part of the data    
    flag = np.nanmedian(resflux[:,:,-1].T,axis=0)
    x = resflux[j,:,1].T.flatten()
    y = resflux[j,:,2].T.flatten()
    w = resflux[j,:,3].T.flatten()
    T = resflux[j,:,0].T.flatten()
    components = np.array([1,2])
    gic = []
    ind0=[]
    ind1=[]
    comp=2
    for i in components:
        gmm = mixture.GaussianMixture(n_components=i, covariance_type='full')
        gmm.fit(np.c_[x,y,w,T])
        gic.append(gmm.bic(np.c_[x,y,w,T]))
    gic = gic
    labels=gmm.predict(np.c_[x,y,w,T])
    if np.diff(gic)/np.abs(gic[0])>-.15:
        wmean = np.nanmean(w)
        if np.abs(wmean)>1:
            stat = 2
            ind0 = np.ones(w.shape).astype(bool)
        else:
            stat = 1
            ind0 = np.ones(w.shape).astype(bool)
    
    else:       
        stat = 3
        for l in np.unique(labels):
            ind = labels==l            
            if np.abs(np.nanmean(w[ind]))>1:
                ind0 = ind
        if len(ind0)==0:
            stat=2
            ind0 = np.ones(w.shape).astype(bool)
    #fluctuations horizontal, vertical and temperature   
    if stat != 1:
        wprime = w.copy()
        wprime[ind0] = w[ind0]-np.nanmean(w[ind0])
        wprime[~ind0] = w[~ind0]-np.nanmean(w[~ind0])
        tprime = T.copy()               
        tprime[ind0] = T[ind0]-np.nanmean(T[ind0])
        tprime[~ind0] = T[~ind0]-np.nanmean(T[~ind0])                
        xprime = x.copy()
        xprime[ind0] = x[ind0]-np.nanmean(x[ind0])
        xprime[~ind0] = x[~ind0]-np.nanmean(x[~ind0])
        yprime = y.copy()       
        yprime[ind0] = y[ind0]-np.nanmean(y[ind0])
        yprime[~ind0] = y[~ind0]-np.nanmean(y[~ind0])       
        #Outliers
        ind_out = np.abs(wprime)>3*np.nanstd(wprime)
        if (np.sum(ind_out)/len(w))>0.0027:
            ind_out = np.abs(wprime)>3.5*np.nanstd(wprime)
            wprime[ind_out] = 0
            tprime[ind_out] = 0
            xprime[ind_out] = 0
            yprime[ind_out] = 0
            wprime[ind0] = wprime[ind0]-np.nanmean(wprime[ind0])
            wprime[~ind0] = wprime[~ind0]-np.nanmean(wprime[~ind0])
            tprime[ind0] = tprime[ind0]-np.nanmean(tprime[ind0])
            tprime[~ind0] = tprime[~ind0]-np.nanmean(tprime[~ind0])   
            xprime[ind0] = xprime[ind0]-np.nanmean(xprime[ind0])
            xprime[~ind0] = xprime[~ind0]-np.nanmean(xprime[~ind0])  
            yprime[ind0] = yprime[ind0]-np.nanmean(yprime[ind0])
            yprime[~ind0] = yprime[~ind0]-np.nanmean(yprime[~ind0])          
        if stat == 2:
            inds = heights!= heights[j]
        if stat == 3:
            inds = np.ones(heights.shape).astype(bool)
        S = np.nanmean(resflux10,axis=0)
        
        D = np.zeros(heights.shape)
        T = np.zeros(heights.shape)
        W = np.zeros(heights.shape)
        
        for i in range(len(D)):
            if i==3:
                d = (270-resflux[i,~ind0,7].T) % 360
                Dc = np.nanmean(np.cos(d*np.pi/180),axis=0)
                Ds = np.nanmean(np.sin(d*np.pi/180),axis=0)
                D[i] = np.arctan2(Ds,Dc)
                T[i] = np.nanmean(resflux[i,~ind0,0].T,axis=0)
                W[i] = np.nanmean(resflux[i,~ind0,3].T,axis=0)
            else:
                d = (270-resflux[i,:,7].T) % 360
                Dc = np.nanmean(np.cos(d*np.pi/180),axis=0)
                Ds = np.nanmean(np.sin(d*np.pi/180),axis=0)
                D[i] = np.arctan2(Ds,Dc)
                T[i] = np.nanmean(resflux[i,:,0].T,axis=0)
                W[i] = np.nanmean(resflux[i,:,3].T,axis=0)                
        
        # Dc = np.nanmean(np.cos(((270-resflux[:,~ind0,7].T) % 360)*np.pi/180),axis=0)
        # Ds = np.nanmean(np.sin(((270-resflux[:,~ind0,7].T) % 360)*np.pi/180),axis=0) 
        # D = np.arctan2(Ds,Dc)
        
        S = UnivariateSpline(heights10[::-1],S[::-1],k=3)(heights)
        D = UnivariateSpline(heights[inds][::-1],D[inds][::-1],k=3)(heights)  
        T = UnivariateSpline(heights[inds][::-1],T[inds][::-1],k=3)(heights)
        W = UnivariateSpline(heights[inds][::-1],W[inds][::-1],k=3)(heights)
        
        Y = S*np.cos(D)
        X = S*np.sin(D)        
        xj = X[j]+xprime
        yj = Y[j]+yprime
        wj = W[j]+wprime              
        tj = T[j]+tprime        
    x = resflux[:,:,1].T.copy()
    y = resflux[:,:,2].T.copy()
    w = resflux[:,:,3].T.copy()
    t = resflux[:,:,0].T.copy()
    if stat != 1:
        x[:,j] = xj
        y[:,j] = yj
        w[:,j] = wj
        t[:,j] = tj 
    
    th = (t+273.15)*(p0/1000)**.286
    TH = np.nanmean(th, axis=0)    
    X = np.nanmean(x, axis=0)
    Y = np.nanmean(y, axis=0)
    u = x.copy()
    v = x.copy()
    U = X.copy()
    V = X.copy()
    W = X.copy()
    if stat == 1:
        D = (270-resflux[:,~ind0,7].T) % 360
        Dc = np.nanmean(np.cos(D*np.pi/180),axis=0)
        Ds = np.nanmean(np.sin(D*np.pi/180),axis=0)
        D = np.arctan2(Ds,Dc)     
    angle = np.arctan2(X,Y)
    # Components in matrix of coefficients
    for i,a in enumerate(angle):
        S11 = np.cos(a)
        S12 = np.sin(a)
        R = np.array([[S11,S12], [-S12,S11]])
        vel = np.array(np.c_[y[:,i],x[:,i]]).T
        vel = np.dot(R,vel)
        u[:,i] = vel[0,:]
        v[:,i] = vel[1,:]
        U[i] = np.nanmean(u[:,i], axis=0)
        V[i] = np.nanmean(v[:,i], axis=0)
        W[i] = np.nanmean(w[:,i], axis=0)
        u[:,i] = u[:,i]-U[i]
        v[:,i] = v[:,i]-V[i]
        th[:,i] = th[:,i]-TH[i]
        w[:,i] = w[:,i]-W[i]       
    ####################filter large fluctuations#############################
    time = np.arange(u.shape[0])
    for j in range(u.shape[1]):
        aux = u[:,j]  
        u[:,j] = u[:,j]-mov_con(np.r_[np.ones(N)*aux[0],aux,np.ones(N)*aux[-1]],N)[N:-N]
        aux = v[:,j]  
        v[:,j] = v[:,j]-mov_con(np.r_[np.ones(N)*aux[0],aux,np.ones(N)*aux[-1]],N)[N:-N]
        aux = w[:,j]  
        w[:,j] = w[:,j]-mov_con(np.r_[np.ones(N)*aux[0],aux,np.ones(N)*aux[-1]],N)[N:-N]
    uu  = np.nanmean(u*u, axis=0)
    uv  = np.nanmean(u*v, axis=0)
    uw  = np.nanmean(u*w, axis=0)
    vv  = np.nanmean(v*v, axis=0)
    vw  = np.nanmean(v*w, axis=0)
    ww  = np.nanmean(w*w, axis=0)
    wth = np.nanmean(w*th, axis=0)  
    stat = np.ones(U.shape)*stat    
    u_fric = (uw**2+vw**2)**.25
    k = .4
    g = 9.806
    L = -u_fric**3*TH/(k*g*wth)
    namei = np.ones(U.shape)*int(name)
    flags=np.median
    return np.c_[namei, U, V, W, TH, uu, uv, uw, vv, vw, ww, wth, L, D, angle, stat, flag]
###################################################################################################

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
L_file = '/mnt/mimer/lalc/results/correlations/west/L_ph1_60m.pkl'
Lcorr, _ = joblib.load('/mnt/mimer/lalc/results/correlations/west/storeL.pkl')

osterild_database = create_engine('mysql+mysqldb://lalc:La469lc@ri-veadbs04:3306/oesterild_light_masts')
times_phase = pd.to_datetime(Lcorr.time.values).sort_values()
times_stamp_ph = times_phase.strftime("%Y%m%d%H%M").values
# List of dates with pandas
init = times_phase.strftime("%Y%m%d").values[0]
end = (times_phase+timedelta(days=1)).strftime("%Y%m%d").values[-1]

t_ph = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range(init, end,
                                      freq='60T'))
       .index.strftime('%Y%m%d%H%M')
       .tolist())
Tabs_0 = ['Tabs_241m_LMS','Tabs_175m_LMS','Tabs_103m_LMS','Tabs_37m_LMS','Tabs_7m_LMS']
Tabs_1 = ['Tabs_241m_LMN','Tabs_175m_LMN','Tabs_103m_LMN','Tabs_37m_LMN','Tabs_7m_LMN']
Sspeed_0 = ['Sspeed_241m_LMS','Sspeed_175m_LMS','Sspeed_103m_LMS','Sspeed_37m_LMS','Sspeed_7m_LMS']
Sspeed_1 = ['Sspeed_241m_LMN','Sspeed_175m_LMN','Sspeed_103m_LMN','Sspeed_37m_LMN','Sspeed_7m_LMN']
Sdir_0 = ['Sdir_241m_LMS','Sdir_175m_LMS','Sdir_103m_LMS','Sdir_37m_LMS','Sdir_7m_LMS']
Sstat_0 = ['Sstat_241m_LMS','Sstat_175m_LMS','Sstat_103m_LMS','Sstat_37m_LMS','Sstat_7m_LMS']
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
Sspeed10_0 = ['wsp_244m_LMS', 'wsp_210m_LMS', 'wsp_178m_LMS', 'wsp_140m_LMS',
            'wsp_106m_LMS', 'wsp_70m_LMS', 'wsp_40m_LMS', 'wsp_10m_LMS']
Sdir10_0 = ['Wdir_244m_LMS', 'Wdir_40m_LMS']

heights = [241, 175, 103, 37, 7]
heights10 = [244, 210, 178, 140, 106, 70, 40, 10]
stab_20hz = pd.DataFrame()
stab_10hz = pd.DataFrame()
stab_press = pd.DataFrame()

res = []
for j in range(len(t_ph)-1):
    stampi = t_ph[j]
    stampe = t_ph[j+1]
    print(stampi, stampe, j)    
    table_20Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_20hz '
    table_10Hz = ' from caldata_'+stampi[:4]+'_'+stampi[4:6]+'_10hz '
    where_i = ' name <= (select max(name)' + table_20Hz + 'where name < ' + stampe + ')'
    where_e = ' name >= (select min(name)' + table_20Hz + 'where name >= ' + stampi + ')'
    where = 'where '+ where_i +' and '+where_e
    sql_query = 'select ' + ", ".join(['name']+T_0+X_0+Y_0+Z_0+Sspeed_0+Sdir_0+Sstat_0) + table_20Hz +  where
    stab_20hz = pd.read_sql_query(sql_query,osterild_database)
    where_i = ' name <= (select max(name)' + table_10Hz + 'where name < ' + stampe + ')'
    where_e = ' name >= (select min(name)' + table_10Hz + 'where name >= ' + stampi + ')'
    where = 'where '+ where_i +' and '+where_e
    sql_query = 'select ' + ", ".join(['name']+Sspeed10_0+Sdir10_0) + table_10Hz +  where
    stab_10hz = pd.read_sql_query(sql_query,osterild_database)  
    stab_20hz.fillna(value=pd.np.nan, inplace=True)
    stab_10hz.fillna(value=pd.np.nan, inplace=True)  
    table = ' from calmeans '
    where = 'where name = (select max(name)' + table + 'where name < ' + stampi + ')'
    sql_query = 'select name, ' + ", ".join(P_0) + table +  where
    stab_press = pd.read_sql_query(sql_query,osterild_database)    
    if len(stab_20hz)>0:
        resflux = []
        for i in range(len(T_0)):
            t0 = stab_20hz[T_0[i]].values
            x0 = stab_20hz[X_0[i]].values
            y0 = stab_20hz[Y_0[i]].values
            z0 = heights[i]
            w0 = stab_20hz[Z_0[i]].values
            s0 = stab_20hz[Sspeed_0[i]].values
            st0 = stab_20hz[Sstat_0[i]].values
            d0 = stab_20hz[Sdir_0[i]].values
            p1, p0 = stab_press.values[0,1:3]
            X = np.nanmean(x0)
            Y = np.nanmean(y0)
            angle = np.arctan2(Y,X)
            S11 = np.cos(angle)
            S12 = np.sin(angle)
            R = np.array([[S11,S12], [-S12,S11]])
            vel = np.array(np.c_[x0.flatten(),y0.flatten()]).T
            vel = np.dot(R,vel)
            u0 = vel[0,:]
            v0 = vel[1,:]
            resflux.append(np.c_[t0,x0,y0,w0,u0,v0,s0,d0,st0])
        resflux = np.array(resflux)
        resflux10 = stab_10hz[Sspeed10_0].values
        resfluxdir = stab_10hz[Sdir10_0].values
        res.append(fluxtot(resflux, resflux10 = resflux10, 
                                    heights = heights, heights10 = heights10,
                                    name = stampi, j=-2))   
joblib.dump(res, L_file)
