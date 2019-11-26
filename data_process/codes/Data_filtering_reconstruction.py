# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:50:05 2018

@author: lalc
"""
# In[Packages to use]
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn import cluster
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy.sparse import csr_matrix

from scipy.spatial import Delaunay

import pickle

from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator

from skimage import measure
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.ticker as ticker



############# [Functions] #################

# In[DataFrame manipulation]

def df_ws_grad(df):
    #r = np.unique(df.range_gate.values)
    phi = df.azim.unique()
    #r,phi = np.meshgrid(r,phi)
    #period = len(df.azim.unique())
    dvdr1 = df.ws.diff(axis=1)
    dvdr2 = df.ws.diff(axis=1,periods=-1)
    dvdr3 = df.ws.diff(axis=0)
    dvdr3.loc[df.azim==phi[0]] = np.nan
    dvdr4 = df.ws.diff(axis=0,periods=-1)
    dvdr4.loc[df.azim==phi[-1]] = np.nan
    
    dvdr = pd.concat([dvdr1, dvdr2, dvdr3, dvdr4]).groupby(level=0).median() # Average over diferences
    dvdr.columns = ['dvdr']*(df['ws'].shape[1])
    
    dvdt = df.ws.diff(axis=0,periods=45).fillna(value=0.0)  
    dvdt.columns = ['dvdt']*(df['ws'].shape[1])
    dv = pd.concat([dvdr,dvdt], axis=1)

    return(pd.concat([df,dv], axis=1))

# In[DataFrame manipulation]

def df_mov_med(df,col,n):
    #r = np.unique(df.range_gate.values)

    movmedian = df[col].rolling(n,axis=1,min_periods=1).median()
    
    movmedian = df[col].sub(movmedian).abs()
    
    movmedian.columns = ['movmed']*(df[col].shape[1])
    
    return(pd.concat([df,movmedian], axis=1))


# In[un-structured grid generation]

def translationpolargrid(mgrid,h):  
    # Linear tranaslation from (r,theta) = (x,y) -> (x0,y0) = (x+h[0],y+h[1])
    # mgrid = (rho,phi)
    rho = mgrid[0]
    phi = mgrid[1]
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    x0 = x - h[0]
    y0 = y - h[1]
    rho_prime = np.sqrt(x0**2+y0**2)
    phi_prime = np.arctan2(y0,x0)
    return(rho_prime, phi_prime)

def nearestpoint(mg0,mg1,dr,dp):
    #mg1 = (rgrid1,thetagrid1)
    #mg2 = (rgrid2,thetagrid2)
    #R = circle of radius R for neighbouring points scaled
    r0 = mg0[0].flatten()
    r1 = mg1[0].flatten()
    p0 = mg0[1].flatten()
    p1 = mg1[1].flatten()
    raux = []
    paux = []
    iaux = []
    
    for i in range(len(r1)):
        
        dist1 = np.sqrt((r0-r1[i])**2)
        dist2 = np.sqrt((p0-p1[i])**2)
        
        ind = ((dist1<=dr) & (dist2<=dp)).nonzero()[0] 
              
        raux.append(r0[ind])
        paux.append(p0[ind])
        iaux.append(ind)        
    r_flat= [item for sublist in raux for item in sublist]
    p_flat= [item for sublist in paux for item in sublist]
    i_flat= [item for sublist in iaux for item in sublist]
    polar = zip(r_flat, p_flat, i_flat)
    unique = [list(t) for t in zip(*list(set(polar)))]   
    return (np.array(unique[0]),np.array(unique[1]),np.array(unique[2]))

# In[Overlapping grid]

def grid_over2(mg0, mg1, d):
    #grid on the overlaping area
    #mg0     = (r0,phi0) non translated
    #mg1     = (r1,phi1) non translated 
    dr = min(np.diff(np.unique(mg0[0].flatten())))/2
    dp = min(np.diff(np.unique(mg0[1].flatten())))/2   
    
    # Translation of grids
    r0, p0 = translationpolargrid(mg0,-d/2)
    r1, p1 = translationpolargrid(mg1,d/2) 
    
    # Overlapping points
    r_o_0, p_o_0, i_o_0 = nearestpoint((r0,p0),(r1,p1),dr,dp)
    r_o_1, p_o_1, i_o_1 = nearestpoint((r1,p1),(r0,p0),dr,dp)
    
    #Cartesian trees
    pos0 = np.c_[r_o_0*np.cos(p_o_0),r_o_0*np.sin(p_o_0)]    
    tree_0 = KDTree(pos0)
    pos1 = np.c_[r_o_1*np.cos(p_o_1),r_o_1*np.sin(p_o_1)]    
    tree_1 = KDTree(pos1)  
    
    # Intersections, frist iteration
    ind0,dist0 = tree_0.query_radius(tree_1.data, r=3*dr/2,return_distance = True,sort_results=True) #check r!
    ind1,dist1 = tree_1.query_radius(tree_0.data, r=3*dr/2,return_distance = True,sort_results=True) #check r! 
    ind00 = []
    ind01 = []
    for i,j in zip(ind0,range(len(ind0))):
        if i.size > 0:
            #indices
            ind00.append(np.asscalar(i[0]))
            ind01.append(j)           
    ind10 = []
    ind11 = []
    for i,j in zip(ind1,range(len(ind1))):
        if i.size > 0:
            #indices
            ind11.append(np.asscalar(i[0]))
            ind10.append(j)
            #dismax11.append(np.asscalar(dist1[-1]))        

    posg0=np.c_[0.5*(pos0[:,0][ind00]+pos1[:,0][ind01]),0.5*(pos0[:,1][ind00]+pos1[:,1][ind01])] 
    posg1=np.c_[0.5*(pos0[:,0][ind10]+pos1[:,0][ind11]),0.5*(pos0[:,1][ind10]+pos1[:,1][ind11])]  
    posg = np.vstack((posg0,posg1))
    unique = [list(t) for t in zip(*list(set(zip(posg[:,0], posg[:,1]))))] 
    posg = np.c_[unique[0],unique[1]]
    tree_g = KDTree(posg)    
    indg, distg = tree_g.query_radius(tree_g.data, r=2*dr, return_distance = True, sort_results=True)
    S = sorted(set((tuple(sorted(tuple(i))) for i in indg if len(tuple(i))>1)))
    nonS = [np.asscalar(i) for i in indg if len(tuple(i))==1]
    temp = [set(u) for u in S]
    S = []
    for ti in temp:
        aux = [t for t in temp if t!=ti]
        if not any(ti <= u for u in aux):
            S.append(list(ti))
  
    aux=np.array([np.mean(posg[list(p),:],axis=0) for p in S])  
    posg = np.vstack((aux,posg[nonS])) 
    tree_g = KDTree(posg)
    
    d0,n0 = tree_g.query(tree_0.data, return_distance = True)
    d1,n1 = tree_g.query(tree_1.data, return_distance = True)
    
    w0 = np.squeeze(d0)**-1
    w1 = np.squeeze(d1)**-1
    n0 = np.squeeze(n0)
    n1 = np.squeeze(n1)

    tri = Triangulation(posg[:,0], posg[:,1])
     
    return (tree_g, tri, w0, n0, i_o_0, w1, n1, i_o_1)


# In[Wind field reconstruction]

def windfieldrec(Vr0,phi0,Vr1,phi1):
    #Cartesian wind field [U,V] from two lidar measurements
    #Trec = np.array([[np.cos(phi0),np.sin(phi0)],[np.cos(phi1),np.sin(phi1)]])
    phi0 = np.pi-phi0
    phi1 = np.pi-phi1
    
#    U = (Vr0*np.cos(phi1)-Vr1*np.cos(phi0))/(np.sin(phi0-phi1))
#    V = (Vr0*np.sin(phi1)-Vr1*np.sin(phi0))/(np.sin(phi0-phi1))
    
#    U = (Vr0*np.cos(phi1)-Vr1*np.cos(phi0))/(np.sin(phi0-phi1))
#    V = (Vr0*np.sin(phi1)-Vr1*np.sin(phi0))/(np.sin(phi0-phi1))
    #return np.dot(np.linalg.inv(Trec),np.array([Vr0,Vr1]))
    a = np.array([[np.sin(phi0),np.cos(phi0)], [np.sin(phi1),np.cos(phi1)]])
    b = np.array([Vr0,Vr1])
    x = np.linalg.solve(a, b)
    U = x[0]
    V = x[1]
    
    return (U,V)

def wind_field_rec(Lidar0, Lidar1, tree, triangle, d): #at time t, old coordinate sistem

    #Lidar0: (vr0,r0,phi0,w0,neigh0,index0) 
    #Lidar1: (vr1,r1,phi1,w1,neigh1,index1)
    
    vr0,phi0_old,w0,neigh0,index0 = Lidar0
    vr1,phi1_old,w1,neigh1,index1 = Lidar1
    
    vr0 = vr0.values.flatten()[index0]
    vr1 = vr1.values.flatten()[index1]
    
    # r and phi is the new field, phi
#    _,phi0_old = translationpolargrid((r0,phi0),-d/2)
#    _,phi1_old = translationpolargrid((r1,phi1),d/2)
    
    phi0_old = phi0_old.flatten()[index0]
    phi1_old = phi1_old.flatten()[index1]
    
    U = np.ones(len(tree.data))
    V = np.ones(len(tree.data))
    
    U[U==1] = np.nan
    V[V==1] = np.nan
    
    dphi0= np.ones(len(tree.data))
    dphi0[dphi0==1] = np.nan
    
    dphi1 = np.ones(len(tree.data))
    dphi1[dphi1==1] = np.nan
    
    for i in range(len(tree.data)):
        ind0 = (neigh0==i).nonzero()[0]
        ind1 = (neigh1==i).nonzero()[0]
        ind00 = (~np.isnan(vr0[ind0])).nonzero()[0]
        ind11 = (~np.isnan(vr1[ind1])).nonzero()[0]
        
        vr_0 = vr0[ind0][ind00]
        vr_1 = vr1[ind1][ind11]
        
        w_0  = w0[ind0][ind00]
        w_1  = w1[ind1][ind11]
        
        sin0 = np.sin(phi0_old[ind0][ind00])
        cos0 = np.cos(phi0_old[ind0][ind00])
        
        sin1 = np.sin(phi1_old[ind1][ind11])
        cos1 = np.cos(phi1_old[ind1][ind11])
        
        # Averages in each grid point
        
        if (w_0.size > 0) and (w_1.size > 0):
                        
#            vr0_ave = np.average(vr_0, weights = w_0/np.sum(w_0))             
#            vr1_ave = np.average(vr_1, weights = w_1/np.sum(w_1))
#            
#            sin0_ave = np.average(sin0, weights = w_0/np.sum(w_0))
#            cos0_ave = np.average(cos0, weights = w_0/np.sum(w_0))
#            
#            sin1_ave = np.average(sin1, weights = w_1/np.sum(w_0))
#            cos1_ave = np.average(cos1, weights = w_1/np.sum(w_1))
#            
#            phi_0 = np.arctan2(sin0_ave,cos0_ave)
#            phi_1 = np.arctan2(sin1_ave,cos1_ave)
#        
#            dphi0[i] = phi_0
#            dphi1[i] = phi_1
#            
#            U[i], V[i] = windfieldrec(vr0_ave, phi_0,vr1_ave, phi_1)
            
             alpha_i = np.r_[sin0*w_0,sin1*w_1]
             beta_i = np.r_[cos0*w_0,cos1*w_1]
             V_i = np.r_[vr_0*w_0,vr_1*w_1]
             
             S11 = np.nansum(alpha_i**2)
             S12 = np.nansum(alpha_i*beta_i)
             S22 = np.nansum(beta_i**2)
             V11 = np.nansum(alpha_i*V_i)
             V22 = np.nansum(beta_i*V_i)
             
             a = np.array([[S11,S12], [S12,S22]])
             b = np.array([V11,V22])
             x = np.linalg.solve(a, b)
             U[i] = x[0]
             V[i] = x[1]
     
        else:
            U[i], V[i] = np.nan, np.nan
    
    return (U, V, dphi0,dphi1)

# In[Filtering]
    
# CNR limit
def filt_CNR(group,value):
        vel = group.ws
        mask_cnr = (group.CNR<=value[0]) | (group.CNR>=value[1])
        mask_cnr.columns = vel.columns
        group.ws = vel.mask(mask_cnr)
        return group.ws
    
def filt_stat(df_ini,col,P,ngrid,g_by,N,bw=0.0,init=0):
    
    df = df_ws_grad(df_ini)
    
    var_g,var_g_s,grid_s,grid_shape = kernel_grid(df,ngrid,col)
    bw = 0.014210526315789474 # !!!! Change this!
    df=df.set_index(g_by)
    values = []
    out = pd.DataFrame()
    Ztot = []
    for i in range(init,N):
        #print(i)
        if i == init: 
            Z,var, var_s = kernel_scan(df.loc[i],col,var_g,var_g_s,grid_s,grid_shape,bandwidth=bw)
            Z_old = Z
            #bwv = np.array(list(bw.values()))
            
        else:
            Z,var, var_s = kernel_scan(df.loc[i],col,var_g,var_g_s,grid_s,grid_shape, bandwidth=bw, Z_old = Z_old,l = 0.7)
            Z_old = Z
        
        Ztot.append(Z)
        
        value = total_pdf(Z,var_g_s,P) 
        values.append(value)
        print(i,value)
        points = grid_s[Z.flatten()>value,:]
        
        hull = Delaunay(points,qhull_options = 'QJ')
        invalid = np.reshape(hull.find_simplex(var_s)<0,df.loc[i,'ws'].values.shape)  
        #print(len(grid_s),len(points),len(var_s), len(var_s[hull.find_simplex(var_s)<0]))
        aux = df.loc[i,'ws'].values
        aux[invalid] = np.nan
        aux2 = df.loc[i,'ws'] 
        aux2.iloc[:,:]= aux     
        out = pd.concat([out,aux2])
    return(out,values,Ztot)    

# In[Other filters]
# In[]

def data_filt_median(df_ini,col='ws',lim_m=6,lim_g=9,n=10): 
    df = df_ini.copy()
    # filter in r
    mmr = df[col].rolling(n,axis=1,min_periods=1).median()
    gm = df[col].sub(df[col].median(axis=1),axis=0)
    # filter in phi
    mmp = df.groupby('scan')[col].rolling(10,axis=0,min_periods=1).median().reset_index(level='scan')[col]
    mmp = df[col].sub(mmp)
    mmr = df[col].sub(mmr)
    gm = df[col].sub(gm)  
    mask = []
    mask = (mmr.abs()>lim_m) | (gm.abs()>lim_g) | (mmp.abs()>lim_m)   
    #r =  []
    #r = df.copy().ws.mask(mask)    
    #r['scan'] = df.scan   
    df.ws = df.copy().ws.mask(mask) 
    return df

def filt_CNR(group,value):
    vel = group.ws
    mask_cnr = (group.CNR<=value[0]) | (group.CNR>=value[1])
    mask_cnr.columns = vel.columns
    group.ws = vel.mask(mask_cnr)
    return group.ws

# In[]
def runningmedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array([np.median(c) for c in b])

# In[]
    
def data_filt_DBSCAN(df_ini,features,nn=5,eps=0.3,plot=False):

    df = df_ws_grad(df_ini)
    if 'movmed' in features:
        df = df_mov_med(df,'ws',10)

    a,b = df.CNR.values.shape 
    X = np.empty((a*b,len(features)))
    # Data preparation
    ind = ~np.isnan(df.ws.values.flatten())
    
    
    X = np.empty((sum(ind),len(features)))
    
    for i,f in enumerate(features):
        if (f =='azim') | (f =='scan') :
            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
            

        else:
            X[:,i] = df[f].values.flatten()[ind]
    
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
    
    tree_X = KDTree(X)
    
    d,i = tree_X.query(tree_X.data,k = nn)
    
    d=d[:,-1]
    
    d = np.log(np.sort(d))
    
    l = np.arange(0,len(d))  
    
    d_resample = np.array(d[::int(len(d)/300)])
    
    l_resample = l[::int(len(d)/300)]
    
    spl = UnivariateSpline(l_resample, d_resample,s=0.1)
    
    kappa = np.log(np.abs(spl.derivative(n=1)(l_resample)))
    
    ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa))))
    
    ind_kappa = ind_kappa[ind_kappa>int(.5*len(kappa))]
    
    l1 = l_resample[ind_kappa][0]
    
    eps0 = np.exp(d[l1])
    
    x = np.sum(df.CNR.values>=-24)/len(df.CNR.values.flatten())
    
    c = .075#np.exp(np.min(d))
    
    m = .45#np.exp(np.max(d))-c
    
    eps = m*x+c   
    eps1 = eps0
    
    if x<.45:
        eps1=eps
    
    clf = []

    clf = DBSCAN(eps=(eps1+eps)*.5, min_samples=nn)
    
    clf.fit(X)

    core_samples_mask = np.zeros_like(clf.labels_, dtype=bool)
    core_samples_mask[clf.core_sample_indices_] = True
    
    labels = clf.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    p = set(labels)
    
    #print(metrics.silhouette_score(X, labels))
    
    if plot:
        
#        plt.figure()
#        #plt.plot(l,d)
#        #plt.plot(l_resample,d_resample)
#        plt.plot(l,spl(l))
#        plt.xlabel('Data point')
#        plt.ylabel('k-nearest distance [-]')
        
#        plt.figure()
#        plt.plot(l_resample,kappa)
#        plt.plot([l_resample[0],l_resample[-1]],[np.mean(kappa),np.mean(kappa)])
#        plt.plot([l_resample[0],l_resample[-1]],[np.mean(kappa)+1*np.std(kappa),np.mean(kappa)+1*np.std(kappa)])
#        plt.scatter(l1,kappa[ind_kappa][0])
        
        plt.figure()
        plt.plot(d)
        plt.plot([0,X.shape[0]],[np.log(eps),np.log(eps)])
        plt.plot([0,X.shape[0]],[np.log(eps0),np.log(eps0)])
        plt.plot([0,X.shape[0]],[np.log(.5*(eps+eps0)),np.log(.5*(eps+eps0))])
        plt.xlabel('Data point', fontsize=12, weight='bold')
        plt.ylabel('k-nearest distance [-]', fontsize=12, weight='bold')
        plt.grid()
        plt.legend(['Noise level','Knee','Average'])
    
        for i,f in enumerate(features):
            if (f =='azim') | (f =='scan') :
                X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
            else:
                X[:,i] = df[f].values.flatten()[ind]
        
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.set_xlabel('$V_{LOS} [m/s]$', fontsize=12, weight='bold')
        ax.set_ylabel('Range gate $[m]$', fontsize=12, weight='bold')
        ax.set_zlabel('CNR', fontsize=12, weight='bold')
        for k, col in zip(unique_labels, colors):
    
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = X[class_member_mask & core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k', s=20)
        
            xy = X[class_member_mask & ~core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k', s=6)  
    
    lab = labels[df.CNR.values.flatten()>-24][0]
    
    labels = np.reshape(labels,(a,b))

    mask = pd.DataFrame(labels!=lab)
#
    mask.columns = ['ws']*b
    mask=mask.set_index(df.ws.index)

    return mask

# In[Interpolation]    
    
def data_interp_kdtree(df_ini,r,phi,dt): #I can work inside the dataframe
    
    df = df_ini.copy()
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30, n_jobs=1)
    
    #dt = np.max(np.meshgrid(r,phi)[0]*np.cos(np.meshgrid(r,phi)[1]))
    
    rg,phig,tg = np.meshgrid(r,phi,np.array([-dt,0,dt]))
    ind_slice = np.isnan(rg)
    ind_slice[:,:,1] = True
    ws_f = []
    
    for scan in np.unique(df.index.values):
        print(scan)
        if (scan == 0) | (scan == np.unique(df.index.values)[-1]):
            rg0,phig0 = np.meshgrid(r,phi)
            rp = np.c_[rg0.flatten(),phig0.flatten()]  
            ws = df.ws.loc[scan].values.flatten()
            ind = np.isnan(ws)
            if sum(ind)>0:
                #print(rp.shape,ws.shape,ind.shape)
                neigh.fit(rp[~ind,:], ws[~ind])
                ws[ind] = neigh.predict(rp[ind,:])
            ws_f.append(np.reshape(ws,rg0.shape))
                
        else:
 
            ws = np.dstack((df.ws.loc[scan-1].values, df.ws.loc[scan].values,df.ws.loc[scan+1].values))
            
            temp = ws.flatten()*tg.flatten()
            
            temp[np.isnan(temp)]=tg.flatten()[np.isnan(temp)]*1000
            
            rpt = np.c_[rg.flatten()*np.cos(phig.flatten()),rg.flatten()*np.cos(phig.flatten()),temp]
            
            ind = np.isnan(ws).flatten()
            ind_int = (ind) & (ind_slice.flatten())
            if sum(ind_int)>0:
                ws = ws.flatten()
                neigh.fit(rpt[~ind,:], ws[~ind])
                ws[ind_int] = neigh.predict(rpt[ind_int,:])
            ws_f.append(np.reshape(ws,rg.shape)[:,:,1])  
    df.ws= pd.DataFrame(data = np.vstack(ws_f), index = df.ws.index, columns = df.ws.columns)     
    return (df) 

# In[]
    
def data_interp_triang(U,V,x,y,dt): #I can work inside the dataframe
    
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30, n_jobs=1)
    
    U_int = []
    V_int = []
    
    it = range(len(U))
    
    for scan in it:
        
        print(scan)
        
        if scan == it[0]:
            
            xj = x-U[scan+1]*dt
            yj = y-V[scan+1]*dt
            
            X = np.c_[np.r_[x,xj],np.r_[y,yj]]
            
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()

            ind0 = np.isnan(np.r_[U[scan],U[scan+1]])
            ind1 = np.isnan(U[scan])
            
            if (sum(ind1)>0) & (sum(~ind0)>26):
                
                neigh.fit(X[~ind0,:], np.r_[U[scan],U[scan+1]][~ind0])
                
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                neigh.fit(X[~ind0,:], np.r_[V[scan],V[scan+1]][~ind0])
                
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                U_int.append(U_aux)
                V_int.append(V_aux)
                
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                
                
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux)                

        if scan == it[-1]:
            
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt
            
            X = np.c_[np.r_[xj,x],np.r_[yj,y]]
            
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()

            ind0 = np.isnan(np.r_[U[scan-1],U[scan]])
            ind1 = np.isnan(U[scan])
            
            if (sum(ind1)>0) & (sum(~ind0)>26):
                
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan]][~ind0])
                
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan]][~ind0])
                
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                U_int.append(U_aux)
                V_int.append(V_aux)
                
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                
                
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
                
        else:
 
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt
            
            xk = x-U[scan+1]*dt
            yk = y-V[scan+1]*dt
            
            X = np.c_[np.r_[xj,x,xk],np.r_[yj,y,yk]]
            
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()

            ind0 = np.isnan(np.r_[U[scan-1],U[scan],U[scan+1]])
            ind1 = np.isnan(U[scan])
            
            if (sum(ind1)>0) & (sum(~ind0)>26):
                
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan],U[scan+1]][~ind0])
                
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan],V[scan+1]][~ind0])
                
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                
                U_int.append(U_aux)
                V_int.append(V_aux)
                
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                
                
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
    
    return (U_int,V_int)

# In[Kernels and pdf's]
     
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)

def kernel_scan(df,col,var_g,var_g_s,grid_s,grid_shape,kernel='gaussian',median = False, bandwidth=0.0, Z_old = 0,l = 0):
    
    N = len(df[col[0]].values.flatten())
    var      = np.empty((N,len(col)))
    var_s    = np.empty((N,len(col)))

    for c,i in zip(col,range(len(col))):
        
        if median:
            var[:,i] = df[c].sub(df[c].median(axis=1),axis=0).values.flatten()
        else:
            if c == 'azim':
                var[:,i] = np.tile(df[c].values.flatten().transpose(), (1, int(N/len(df[c].values.flatten())))).flatten()
            else:
                var[:,i] = df[c].values.flatten()
          
        a = 1.0/(max(var_g[:,i])-min(var_g[:,i]))
        b = min(var_g[:,i])        
        #print(a,b)
        # scaling  
        var_s[:,i] = (var[:,i]-b)*a
  
    # bandwidth, cross-validation
    if bandwidth == 0.0:
        gridh = GridSearchCV(KernelDensity(kernel=kernel),{'bandwidth': np.linspace(0.01, .05, 20)},cv=150)
        gridh.fit(var_s)
        #print(var_s)
        bandwidth=gridh.best_params_
        #print(gridh.best_params_)
        kde = gridh.best_estimator_
        Z = np.exp(kde.score_samples(grid_s))
        Z = np.reshape(Z, grid_shape)
        #rescaling 
        IC = simps_rec(Z,var_g_s,list(range(var_g.shape[1])))
        Z = Z/IC
        #print(IC,simps_rec(Z,var_g,list(range(var_g.shape[1]))))
        #if not Z_old:
        #    Z_old = np.zeros(Z.shape)
        return ((1-l)*Z+l*Z_old,var,var_s,bandwidth)
    
    else:
        Z = kde_sklearn(var_s, grid_s, bandwidth = bandwidth,rtol=1e-1,kernel=kernel,leaf_size=150,algorithm='kd_tree',metric='manhattan')
        Z = np.reshape(Z, grid_shape)
        #rescaling 
        IC = simps_rec(Z,var_g_s,list(range(var_g.shape[1])))
        #print(IC)
        Z = Z/IC
        #if not Z_old:
        #    Z_old = np.zeros(Z.shape)
        return ((1-l)*Z+l*Z_old,var,var_s)
    
def pdf_marg(Z,axes):
    f_marg = []
    F_marg = []
    CL = []
    for i in range(axes.shape[1]):
        ind = list(range(axes.shape[1]))
        if i > 0:
            ind = list(range(axes.shape[1]))
            #print(len(axes),i)
            ind.insert(0,ind.pop(i))
        #print(ind)   
        f_marg.append(simps_rec(Z,axes[:,ind[1:]],ind[1:])) 
        F_marg.append(sp.integrate.cumtrapz(f_marg[-1],axes[:,ind[0]],initial = 0.0))
        
        F_marg[-1] = F_marg[-1]/F_marg[-1][np.argmax(F_marg[-1])]
        plt.plot(axes[:,ind[0]],F_marg[-1])
        #F_marg[-1][0]=0.0
        F_marg_int = sp.interpolate.splrep(F_marg[-1],axes[:,ind[0]], k=3)
        CL.append(np.array([sp.interpolate.splev(.025, F_marg_int, der=0),sp.interpolate.splev(.975, F_marg_int, der=0)]))
        
    return(f_marg,F_marg,CL)
        
def simps_rec(Z,axes,naxis):# recursive estimation of integral
    #print(naxis,len(naxis))
    if len(naxis)==1:
        #print(naxis[0])
        return sp.integrate.simps(Z,axes[:,0],axis=naxis[0])
    else:
        #print(naxis[0],naxis[1:])
        #print(naxis[0],simps_rec(Z,axes[:,1:],naxis[1:]).shape)
        return sp.integrate.simps(simps_rec(Z,axes[:,1:],naxis[1:]),axes[:,0],axis=naxis[0]) 


def kernel_grid(df,ngrid,col,median = False):
    var_g    = np.empty((ngrid,len(col)))
    var_g_s    = np.empty((ngrid,len(col)))
    a = np.empty(len(col))
    b = np.empty(len(col))
    
    for c,i in zip(col,range(len(col))):
        if median:
            g_max =df[c].sub(df[c].median(axis=1),axis=0).values.flatten().max()
            g_min =df[c].sub(df[c].median(axis=1),axis=0).values.flatten().min()
        else:
            g_max =df[c].values.flatten().max()
            g_min =df[c].values.flatten().min()
        a[i] = 1.0/(g_max-g_min)
        b[i] = g_min
        var_g[:,i] = np.linspace(g_min,g_max,ngrid)
        if i == 0:
            var_g_s[:,i]      = (var_g[:,i]-b[i])*a[i]
            h = min(np.diff(var_g_s[:,i]))
            C = (ngrid-1)*h
            print(C)
        else:
            var_g_s[:,i]    = C*(var_g[:,i]-b[i])*a[i]
    
    v_mesh_scaled = np.meshgrid(*var_g_s.T)
    
    grid_shape = v_mesh_scaled[0].shape
    
    grid_s  = np.array([g.flatten() for g in v_mesh_scaled]).T
    
    return (var_g,var_g_s,grid_s,grid_shape)
        
def total_pdf(Z,var_g,P):
    N = 20
    aux = np.ones(Z.shape)
    values = np.linspace(0,np.max(Z),N)
    P_values = np.zeros(values.shape)
    for value,i in zip(values,range(len(values))):

        aux[Z < value] = 0.0

        P_values[i] = simps_rec(Z*aux,var_g,list(range(var_g.shape[1])))
        #print(value,simps_rec(Z*aux,var_g,list(range(var_g.shape[1]))))
        if P_values[i] < P:
            #print(i,P_values)
            break
    f = sp.interpolate.interp1d(P_values[0:i+1], values[0:i+1])
    #plt.plot(P_values,values)
    aux = np.ones(Z.shape)
    aux[Z < f(P)] = 0.0

    return(f(P))

# In[Autocorrelation for non-structured grid]  

def spatial_autocorr(tri,U,V,N,alpha):
    
#    tau = .05*1.1**(np.linspace(0,N,N))
#    tau = tau/np.max(tau)
#    #print(N,len(tau),len(np.linspace(0,N)))
#    eta = tau
#    tau = tau*alpha*(np.max(tri.x)-np.min(tri.x))    
#    eta = eta*alpha*(np.max(tri.y)-np.min(tri.y))
#    print(N,len(tau))
#    
##    tau = np.exp(np.linspace(np.log(0.1),np.log(alpha*(np.max(tri.x)-np.min(tri.x))),int(N/2)))
##    eta = np.exp(np.linspace(np.log(0.1),np.log(alpha*(np.max(tri.y)-np.min(tri.y))),int(N/2)))
#    
#    tau = np.r_[-tau,0,tau]
#    eta = np.r_[-eta,0,eta]
    
    tau = np.linspace(-alpha*(np.max(tri.x)-np.min(tri.x)),alpha*(np.max(tri.x)-np.min(tri.x)),N)
    eta = np.linspace(-alpha*(np.max(tri.y)-np.min(tri.y)),alpha*(np.max(tri.y)-np.min(tri.y)),N)
    
    tau,eta = np.meshgrid(tau,eta)
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    U_int = LinearTriInterpolator(tri, U)
    V_int = LinearTriInterpolator(tri, V)

    if len(U[~np.isnan(U)])>0:
        r_u = [autocorr(tri,U,U_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
        r_v = [autocorr(tri,V,V_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(r_u,r_v)


def spatial_autocorr_tree(tri,U,V,N,alpha):
 
    tau = np.linspace(-alpha*0*(np.max(tri.x)-np.min(tri.x)),alpha*(np.max(tri.x)-np.min(tri.x)),N)
    eta = np.linspace(-alpha*0*(np.max(tri.y)-np.min(tri.y)),alpha*(np.max(tri.y)-np.min(tri.y)),N)
    tau,eta = np.meshgrid(tau,eta)
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    U_int = LinearTriInterpolator(tri, U)
    V_int = LinearTriInterpolator(tri, V)

    if len(U[~np.isnan(U)])>0:
        r_u = [autocorr(tri,U,U_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
        r_v = [autocorr(tri,V,V_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(r_u,r_v)
    

    
def autocorr(tri,U,Uint,t,e):
    ind = ~np.isnan(U)
    U_delta = Uint(tri.x[ind]+t,tri.y[ind]+e)
    if len(U_delta.data[~U_delta.mask]) == 0:
        r = np.nan
    else:
        r = np.corrcoef(U_delta.data[~U_delta.mask],U[ind][~U_delta.mask],rowvar=False)[0,1]
    return r
    
def autofourier(tri,U,Uint,t,e):
    ind = ~np.isnan(U)
    U_delta = Uint(tri.x[ind]+t,tri.y[ind]+e)
    if len(U_delta.data[~U_delta.mask]) == 0:
        r = np.nan
    else:
        r = np.corrcoef(U_delta.data[~U_delta.mask],U[ind][~U_delta.mask],rowvar=False)[0,1]
    #r = np.nansum(U*LinearTriInterpolator(tri, U)(tri.x+t,tri.y+e))/len(U[~np.isnan(LinearTriInterpolator(tri, U)(tri.x+t,tri.y+e))])
    return r


def spatial_autocorr_fft(tri,grid,U,V,NN,N):
    

    #U = U-np.nanmean(U)
    #V = V-np.nanmean(V)
    
    U_int = np.reshape(LinearTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data,(N,N))
    
    V_int = np.reshape(LinearTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data,(N,N))
    
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    
    # zero padding
    
    padding = np.zeros((grid[0].shape[0]+10,grid[0].shape[1]+10))
    
    print(padding.shape,U_int.shape)
    
    ind = 5#int(np.round(grid[0].shape[0]))
    print(ind)
    
    padding[ind:ind+grid[0].shape[0],ind:ind+grid[0].shape[0]] = U_int
    
    U_int = padding
    
    padding = np.zeros((grid[0].shape[0]+10,grid[0].shape[1]+10))
    
    padding[ind:ind+grid[0].shape[0],ind:ind+grid[0].shape[0]] = V_int
    
    V_int = padding
    
    plt.contourf(U_int,1000)
    fftU = np.fft.fft2(U_int, [NN,NN])
    fftV = np.fft.fft2(V_int, [NN,NN])
    
    #auxU = fftU*np.conj(fftU)
    r_u = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftU)**2)))/len(padding.flatten())#np.real(np.fft.ifft2(auxU))/len(grid[0].flatten())
    
    #auxV = fftV*np.conj(fftV)
    r_v = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftV)**2)))/len(padding.flatten())#np.real(np.fft.ifft2(auxV))/len(grid[0].flatten())
    
    return(r_u,r_v)
# In[]
def spectra_fft(tri,grid,U,V,NN,N):
    

    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]
    
    U_int = np.reshape(LinearTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data,(N,N))    
    V_int = np.reshape(LinearTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data,(N,N))
    
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    
    # zero padding
    
    padding = np.zeros((grid[0].shape[0]+2,grid[0].shape[1]+2))
    
   # print(padding.shape,U_int.shape)
    
    ind = 1#int(np.round(grid[0].shape[0]))
    #print(ind)
    
    padding[ind:ind+grid[0].shape[0],ind:ind+grid[0].shape[0]] = U_int
    
    U_int = padding
    
    padding = np.zeros((grid[0].shape[0]+2,grid[0].shape[1]+2))
    
    padding[ind:ind+grid[0].shape[0],ind:ind+grid[0].shape[0]] = V_int
    
    V_int = padding
    plt.figure()
    plt.contourf(U_int,200,cmap='rainbow')
    
    fftU = np.fft.fftshift(np.fft.fft2(U_int))
    
    fftV = np.fft.fftshift(np.fft.fft2(V_int))
    
    fftUV = fftU*np.conj(fftV)
    
    print(np.max(np.abs(fftV)**2))
    
    #auxU = fftU*np.conj(fftU)
    #r_u = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftU)**2)))/len(padding.flatten())#np.real(np.fft.ifft2(auxU))/len(grid[0].flatten())
    
    #auxV = fftV*np.conj(fftV)
    #r_v = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftV)**2)))/len(padding.flatten())#np.real(np.fft.ifft2(auxV))/len(grid[0].flatten())
    
    return(2*(np.abs(fftU)**2)/(n*m*dx*dy),2*(np.abs(fftV)**2)/(n*m*dx*dy),2*np.real(fftUV)/(n*m*dx*dy))
    
# In[Statistics/Energy and Enstrophy flux]
    
#def FST():
    

# In[Data loading]
#West scanning

sirocco_w_path = 'Data/Data_Phase_2/SiroccoWest/20160629133556_PPI1_merged_fixed.txt'
sirocco_w_df = pd.read_csv(sirocco_w_path,sep=";", header=None) 

vara_w_path = 'Data/Data_Phase_2/VaraWest/20160629133556_PPI1_merged_fixed.txt'
vara_w_df = pd.read_csv(vara_w_path,sep=";", header=None) 


# In[Data handling]

#LiDAR location

#Labels for identification of runs and beams

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])


labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(198):

    labels = np.concatenate((labels,vel_lab))
    
# In[West]

sirocco_w_df.columns = labels
vara_w_df.columns = labels

sirocco_loc = np.array([6322832.3,0])
vara_loc = np.array([6327082.4,0])
d = vara_loc-sirocco_loc


phi0w = vara_w_df.azim.unique()
phi1w = sirocco_w_df.azim.unique()
r0w = np.array(vara_w_df.iloc[(vara_w_df.azim==min(phi0w)).nonzero()[0][0]].range_gate)
r1w = np.array(sirocco_w_df.iloc[(sirocco_w_df.azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_vaw, phi_vaw = np.meshgrid(r0w, np.radians(phi0w)) # meshgrid
r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

treew,triw, wvaw, neighvaw, indexvaw, wsiw, neighsiw, indexsiw = grid_over2((r_vaw, phi_vaw),(r_siw, phi_siw),d)

sirocco_w_df['scan'] = sirocco_w_df.groupby('azim').cumcount()
vara_w_df['scan'] = vara_w_df.groupby('azim').cumcount()

# In[]
#
#feat = ['ws','range_gate','CNR','azim','dvdr']
#
#with open('mask_clust_v_200.pkl', 'rb') as m_200_int:
#    mask = pickle.load(m_200_int)
#    
#df_clust = vara_w_df.copy()
#
#df_clust.ws = df_clust.ws.mask(mask)

# In[]

filt_med_vara_w = data_filt_median(vara_w_df,lim_m=6,lim_g=20,n=20)
filt_med_sirocco_w = data_filt_median(sirocco_w_df,lim_m=6,lim_g=20,n=20)

# In[Interpolation]

filt_med_vara_w_int = data_interp_kdtree(filt_med_vara_w.copy().set_index('scan'),r0w,phi0w,45)
filt_med_sirocco_w_int = data_interp_kdtree(filt_med_sirocco_w.copy().set_index('scan'),r1w,phi1w,45)

# In[DBSCAN filtering]  
    
mask=pd.DataFrame()

feat = ['ws','range_gate','CNR','azim','dvdr']

df_clust = vara_w_df.copy()

t_step = 2

ind=np.unique(vara_w_df.scan.values)%t_step==0

times= np.unique(np.append(np.unique(vara_w_df.scan.values)[ind], vara_w_df.scan.values[-1]))

for i in range(len(times)-1):
    print(times[i])
    
    loc = (vara_w_df.scan>=times[i]) & (vara_w_df.scan<times[i+1])

    mask = pd.concat([mask,data_filt_DBSCAN(vara_w_df.loc[loc],feat)])
    
    if i == range(len(times)-1):
        
        loc = vara_w_df.scan == times[i+1]
    
        mask = pd.concat([mask,data_filt_DBSCAN(vara_w_df.loc[loc],feat)])
    
df_clust.ws = df_clust.ws.mask(mask)

with open('mask_clust_v_200f.pkl', 'wb') as mask_clust_200:
    pickle.dump(mask, mask_clust_200)

with open('df_clust_v_200f.pkl', 'wb') as clust_200:
    pickle.dump(df_clust, clust_200)

df_clust2 = sirocco_w_df.copy()

mask_s=pd.DataFrame()

ind=np.unique(sirocco_w_df.scan.values)%t_step==0

times= np.unique(np.append(np.unique(sirocco_w_df.scan.values)[ind], sirocco_w_df.scan.values[-1]))

for i in range(len(times)-1):
    print(times[i])
    
    loc = (sirocco_w_df.scan>=times[i]) & (sirocco_w_df.scan<times[i+1])

    mask_s = pd.concat([mask_s,data_filt_DBSCAN(sirocco_w_df.loc[loc],feat)])
    
    if i == range(len(times)-1):
        
        loc = sirocco_w_df.scan == times[i+1]
    
        mask_s = pd.concat([mask_s,data_filt_DBSCAN(sirocco_w_df.loc[loc],feat)])
    
df_clust2.ws = df_clust2.ws.mask(mask_s)

with open('df_clust_s_200f.pkl', 'wb') as clust_200:
    pickle.dump(df_clust2, clust_200)
    
# In[Recosntruction from clustering filtering]
    
with open('df_clust_s_200f.pkl', 'rb') as clust_200:
    df_clust2 = pickle.load(clust_200)
    
with open('df_clust_v_200f.pkl', 'rb') as clust_200:
    df_clust = pickle.load(clust_200)

U200_clust1 = []
V200_clust1 = []
#dphi200_clust0 = []
#dphi200_clust1 = []

for scan_n in range(0,1410):
    print(scan_n)
    Lidar_sir = (df_clust2.ws.loc[df_clust2.scan==scan_n],phi_siw,wsiw,neighsiw,indexsiw) 
    Lidar_var = (df_clust.ws.loc[df_clust.scan==scan_n],phi_vaw,wvaw,neighvaw,indexvaw)

    auxU, auxV= wind_field_rec(Lidar_var, Lidar_sir, treew, triw, d)
    
    U200_clust1.append(auxU) 
    V200_clust1.append(auxV)
#    dphi200_clust0.append(auxphi0)
#    dphi200_clust1.append(auxphi1)

with open('U200_clust.pkl', 'wb') as U_200_int:
    pickle.dump(U200_clust1, U_200_int)
    
with open('V200_clust.pkl', 'wb') as V_200_int:
    pickle.dump(V200_clust1, V_200_int)

# In[interpolation]

#U200_clust_int, V200_clust_int = data_interp_triang(U200_clust,V200_clust,triw.x,triw.y,45)

U200_clust_int1, V200_clust_int1 = data_interp_triang(U200_clust1,V200_clust1,triw.x,triw.y,45)

with open('U200_clust_int.pkl', 'wb') as U_200_int:
    pickle.dump(U200_clust_int1, U_200_int)
    
with open('V200_clust_int.pkl', 'wb') as V_200_int:
    pickle.dump(V200_clust_int1, V_200_int)

# In[triangulazation]
    
xmid = triw.x[triw.triangles].mean(axis=1)
ymid = triw.y[triw.triangles].mean(axis=1)

dUdx = []
dUdy = []
dVdx = []
dVdy = []

Ut = []
Vt = []

masks = []

for i in range(0,1409):
    print(i)
    if len(U200_clust_int[i])>0:
        tciU = CubicTriInterpolator(triw, U200_clust_int1[i],kind = 'geom')
        tciV = CubicTriInterpolator(triw, V200_clust_int1[i],kind = 'geom')
    
        Ut.append(tciU(xmid,ymid).data)
        Vt.append(tciV(xmid,ymid).data)
        
        dUdx.append(tciU.gradient(triw.x, triw.y)[0])
        dUdy.append(tciU.gradient(triw.x, triw.y)[1])
        dVdx.append(tciV.gradient(triw.x, triw.y)[0])
        dVdy.append(tciV.gradient(triw.x, triw.y)[1])  
    else: 
        Ut.append(np.empty(xmid.shape).fill(np.nan))
        Vt.append(np.empty(xmid.shape).fill(np.nan))
        
        dUdx.append(np.empty(triw.x.shape).fill(np.nan))
        dUdy.append(np.empty(triw.x.shape).fill(np.nan))
        dVdx.append(np.empty(triw.x.shape).fill(np.nan))
        dVdy.append(np.empty(triw.x.shape).fill(np.nan))  
    
#    masks.append(np.isnan(tciU(xmid,ymid).data))

with open('dUdx_clust.pkl', 'wb') as dU_dx:
    pickle.dump(dUdx, dU_dx)
    
with open('dUdy_clust.pkl', 'wb') as dU_dy:
    pickle.dump(dUdy, dU_dy)

with open('dVdx_clust.pkl', 'wb') as dV_dx:
    pickle.dump(dVdx, dV_dx)
    
with open('dVdy_clust.pkl', 'wb') as dV_dy:
    pickle.dump(dVdy, dV_dy)
    
with open('Ut_clust.pkl', 'wb') as U_t:
    pickle.dump(Ut, U_t)

with open('Vt_clust.pkl', 'wb') as V_t:
    pickle.dump(Vt, V_t)
    
#with open('masks_clust.pkl', 'wb') as mask_s:
#    pickle.dump(masks, mask_s)

# In[]
    
with open('U200_clust.pkl', 'rb') as U_200_clust:
    U200_clust = pickle.load(U_200_clust)
    
with open('V200_clust.pkl', 'rb') as V_200_clust:
    V200_clust = pickle.load(V_200_clust)

with open('dUdx_clust.pkl', 'rb') as dU_dx:
    dUdx = pickle.load(dU_dx)
    
with open('dUdy_clust.pkl', 'rb') as dU_dy:
    dUdy = pickle.load(dU_dy)

with open('dVdx_clust.pkl', 'rb') as dV_dx:
    dVdx = pickle.load(dV_dx)
    
with open('dVdy_clust.pkl', 'rb') as dV_dy:
    dVdy = pickle.load(dV_dy)

with open('Ut_clust.pkl', 'rb') as U_t:
    Ut = pickle.load(U_t)

with open('Vt_clust.pkl', 'rb') as V_t:
    Vt = pickle.load(V_t)
    
with open('masks_clust.pkl', 'rb') as mask_s:
    masks = pickle.load(mask_s)
    
with open('U200_int_med.pkl', 'rb') as U_200_int:
    U200_int_med = pickle.load(U_200_int)
    
with open('V200_int_med.pkl', 'rb') as V_200_int:
    V200_int_med = pickle.load(V_200_int)

# In[]
    
scan_n = 300
#levels1 = np.linspace(-20,20,100)
#levels2 = np.linspace(0,5,100)
   
fig, (ax1, ax2) = plt.subplots(ncols=2)    

ax1.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax1.use_sticky_edges = False
ax1.margins(0.07)
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax2.use_sticky_edges = False
ax2.margins(0.07)


ax1.triplot(triw, color='black',lw=.1)



ax2.triplot(triw, color='black',lw=.1)
ax2.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax2.set_xlabel(r'North-South [m]', fontsize=12, weight='bold')

im1 = ax1.tricontourf(triw,U200_clust_int1[scan_n],300,cmap='rainbow')
#im1 = ax1.tricontourf(triw,zero*180/np.pi,300,cmap='rainbow')
ax1.set_title('Scan = %scan_n' %scan_n)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.set_ylabel("U")



#im1 = ax1.tricontourf(triw,dphi200_clust1[scan_n],300,cmap='rainbow')
#ax1.set_title('Scan = %scan_n' %scan_n)
#divider1 = make_axes_locatable(ax1)
#cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#cbar1 = fig.colorbar(im1, cax=cax1)
#cbar1.ax.set_ylabel("dphi")

im2 = ax2.tricontourf(triw,V200_clust_int1[scan_n],300,cmap='rainbow')
#im2 = ax2.tricontourf(triw,one*180/np.pi,300,cmap='rainbow')
ax2.set_title('Scan = %scan_n' %scan_n)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.set_ylabel("V")


# In[]  
    
    
# In[]        
for i in range(301,800):
    #print(i)    
    
    if len(plt.gcf().axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts1 = fig.axes[0].get_position().get_points()
        pts2 = fig.axes[1].get_position().get_points()
        pts3 = fig.axes[2].get_position().get_points()
        pts4 = fig.axes[3].get_position().get_points()
        # and its label
#        label1 = fig.axes[2].get_ylabel()
#        label2 = fig.axes[3].get_ylabel()
        # and then remove the axes
        fig.axes[0].remove()
        fig.axes[0].remove()
        fig.axes[0].remove()
        fig.axes[0].remove()
        
        cax1= fig.add_axes([pts1[0][0],pts1[0][1],pts1[1][0]-pts1[0][0],pts1[1][1]-pts1[0][1]  ])
        cax2= fig.add_axes([pts2[0][0],pts2[0][1],pts2[1][0]-pts2[0][0],pts2[1][1]-pts2[0][1]  ])
        cax3= fig.add_axes([pts3[0][0],pts3[0][1],pts3[1][0]-pts3[0][0],pts3[1][1]-pts3[0][1]  ])
        cax4= fig.add_axes([pts4[0][0],pts4[0][1],pts4[1][0]-pts4[0][0],pts4[1][1]-pts4[0][1]  ])
        
        cax1.triplot(triw, color='black',lw=.1)
        cax2.triplot(triw, color='black',lw=.1)
        
        im1 = cax1.tricontourf(triw,U200_clust_int1[i],300,cmap='rainbow')
        im2 = cax2.tricontourf(triw,V200_clust_int1[i],300,cmap='rainbow')

#        # then we draw a new axes a the extents of the old one
#        ax1.set_aspect('equal')
#        # Enforce the margins, and enlarge them to give room for the vectors.
#        ax1.use_sticky_edges = False
#        ax1.margins(0.07)
#        
#        ax1.triplot(triw, color='black',lw=.5)
#        
#        im1 = cax1.tricontourf(triw,dVdx[i]-dUdy[i],cmap='rainbow')
#        
#        cax1= fig.add_axes([pts1[0][0],pts1[0][1],pts1[1][0]-pts1[0][0],pts1[1][1]-pts1[0][1]  ])
        # and add a colorbar to it
        cbar1 = fig.colorbar(im1, cax=cax3)
        cbar2 = fig.colorbar(im2, cax=cax4)
        cbar1.ax.set_ylabel("U")
        cbar2.ax.set_ylabel("V")
#        cbar1.ax.set_ylabel(label1)
#        cbar2.ax.set_ylabel(label2)
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
        #print(fig.axes)
    else:
        plt.colorbar(im)
    
    
    plt.pause(.1)

# In[]


U200_int_med = []
V200_int_med = []

for scan_n in range(0,1409):
    print(scan_n)

    Lidar_sir2 = (filt_med_sirocco_w_int.loc[scan_n].ws, phi_siw,wsiw,neighsiw,indexsiw)
    
    Lidar_var2 = (filt_med_vara_w_int.loc[scan_n].ws, phi_vaw,wvaw,neighvaw,indexvaw)

    auxU2, auxV2,_,_ = wind_field_rec(Lidar_var2, Lidar_sir2, treew, triw, d)
    
    U200_int_med.append(auxU2) 
    V200_int_med.append(auxV2)

with open('U200_int_med.pkl', 'wb') as U_200_int:
    pickle.dump(U200_int_med, U_200_int)
    
with open('V200_int_med.pkl', 'wb') as V_200_int:
    pickle.dump(V200_int_med, V_200_int)
    
# In[]
    
xmid = triw.x[triw.triangles].mean(axis=1)
ymid = triw.y[triw.triangles].mean(axis=1)

#dUdx = []
#dUdy = []
#dVdx = []
#dVdy = []

Ut_med = []
Vt_med = []


for i in range(0,1409):
    print(i)
  
    tciU = CubicTriInterpolator(triw, U200_int_med[i],kind = 'geom')
    tciV = CubicTriInterpolator(triw, V200_int_med[i],kind = 'geom')
    
    Ut_med.append(tciU(xmid,ymid).data)
    Vt_med.append(tciV(xmid,ymid).data)
    
#    dUdx.append(tciU2.gradient(triw.x, triw.y)[0])
#    dUdy.append(tciU2.gradient(triw.x, triw.y)[1])
#    dVdx.append(tciV2.gradient(triw.x, triw.y)[0])
#    dVdy.append(tciV2.gradient(triw.x, triw.y)[1])  
#
#with open('dUdx_med.pkl', 'wb') as dU_dx:
#    pickle.dump(dUdx, dU_dx)
#    
#with open('dUdy_med.pkl', 'wb') as dU_dy:
#    pickle.dump(dUdy, dU_dy)
#
#with open('dVdx_med.pkl', 'wb') as dV_dx:
#    pickle.dump(dVdx, dV_dx)
#    
#with open('dVdy_med.pkl', 'wb') as dV_dy:
#    pickle.dump(dVdy, dV_dy)

with open('Ut_med.pkl', 'wb') as U_t:
    pickle.dump(Ut_med,U_t)

with open('Vt_med.pkl', 'wb') as V_t:
    pickle.dump(Vt_med,V_t)

# In[]
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
        
for i in range(0,1409):
    plt.title('Scan num. %i' %i)
    #triw.set_mask(masks2[i])
    #tri_r.set_mask(masks_r[i])
    ax.cla()
    
    ax.triplot(triw, color='black',lw=.5)
    im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+U200_clust_int1[scan_n]**2),cmap='rainbow')
    #im = ax.tricontourf(triw,U200_int_med[i],cmap='rainbow')
    Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data, units='width', color='b')#, scale=.05, zorder=3, color='b')
    
    tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
    tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
    a = dUdx[i]+dVdy[i]
#    im=ax.tricontourf(triw,a,cmap='rainbow')
    Q = ax.quiver(xmid, ymid, tciU(xmid,ymid).data, tciV(xmid,ymid).data, np.hypot(tciU(xmid,ymid).data, tciV(xmid,ymid)),cmap='rainbow',units='xy')
    qk = ax.quiverkey(Q, 0.9, 0.9,np.mean(tciU(xmid,ymid)), labelpos='E',
                   coordinates='figure')
    cb = plt.colorbar(im)
    ax.set_title('Scan num. %i' %i)
    
    plt.pause(.1)
 
## Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def animate(i): 
    cont = plt.tricontourf(triw,U200_int_med[i],cmap='rainbow')
    plt.title('Scan num. %i' %i)
    return cont  

anim = animation.FuncAnimation(fig, animate, frames = 1409)
anim.save('animation.html',)

# In[]



i=900 
c=0
levels=np.linspace(0,1,200)
tau = np.linspace(-alpha*(np.max(triw.x)-np.min(triw.x)),alpha*(np.max(triw.x)-np.min(triw.x)),N)
eta = np.linspace(-alpha*(np.max(triw.y)-np.min(triw.y)),alpha*(np.max(triw.y)-np.min(triw.y)),N)

plt.figure()

plt.contourf(tau,eta,np.reshape(A_tot[i][c],(N,N)), levels, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, U', fontsize=12, weight='bold')

tau2 = np.linspace(-alpha*(np.max(triw.x)-np.min(triw.x)),alpha*(np.max(triw.x)-np.min(triw.x)),50)
eta2 = np.linspace(-alpha*(np.max(triw.y)-np.min(triw.y)),alpha*(np.max(triw.y)-np.min(triw.y)),50)
plt.figure()
plt.contourf(tau2,eta2,np.reshape(A2[i][c],(50,50)), levels, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, U', fontsize=12, weight='bold')

c=1
plt.figure()

plt.contourf(tau,eta,np.reshape(A_tot[i][c],(N,N)), levels, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, V', fontsize=12, weight='bold')

tau2 = np.linspace(-alpha*(np.max(triw.x)-np.min(triw.x)),alpha*(np.max(triw.x)-np.min(triw.x)),50)
eta2 = np.linspace(-alpha*(np.max(triw.y)-np.min(triw.y)),alpha*(np.max(triw.y)-np.min(triw.y)),50)
plt.figure()
plt.contourf(tau2,eta2,np.reshape(A2[i][c],(50,50)), levels, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, V', fontsize=12, weight='bold')

fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(triw, color='black',lw=.5)
im=ax.tricontourf(triw,U200_clust_int[i],cmap='rainbow')
fig.colorbar(im)


fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(triw, color='black',lw=.5)
im=ax.tricontourf(triw,V200_clust_int[i],cmap='rainbow')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(triw, color='black',lw=.5)
im=ax.tricontourf(triw,U200_int_med[i],cmap='rainbow')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(triw, color='black',lw=.5)
im=ax.tricontourf(triw,V200_int_med[i],cmap='rainbow')
fig.colorbar(im)




# In[Autocorrelation]

A_tot = []
N = 50
alpha = 0.6
for i in range(0,1410):
    print(i)
    if len(U200_clust_int1[i])>0:
        A_tot.append(spatial_autocorr(tri_r,Ur[i],Vr[i],N,alpha))
    else:
        aux = np.empty([N,N])
        aux[:] = np.nan
        A_tot.append(aux)
            
    #B.append(spatial_autocorr(triw,U200[i],V200[i],N,alpha))

with open('r_200_clust_int2.pkl', 'wb') as r_200_int:
    pickle.dump(A_tot, r_200_int)
    
with open('r_200_int3.pkl', 'rb') as r_200:
    A2 = pickle.load(r_200)

# In[Autocorrelation]

A = []
N = 50
alpha = 0.6
for i in range(0,1409):
    print(i)
    A.append(spatial_autocorr(triw,U200_int[i],V200_int[i],N,alpha))
    #B.append(spatial_autocorr(triw,U200[i],V200[i],N,alpha))

with open('r_200_int3.pkl', 'wb') as r_200_int:
    pickle.dump(A, r_200_int)
    
#with open('r_2003.pkl', 'wb') as r_200:
#    pickle.dump(B, r_200)

    
## In[]
#
#A2 = []
#B2 = []
#N = 300
#grid = np.meshgrid(np.linspace(np.min(triw.x),np.max(triw.x),N),np.linspace(np.min(triw.y),np.max(triw.y),N))
#
#for i in range(0,1409):
#    print(i)
#    A2.append(spatial_autocorr_fft(triw,grid,U200_int[i],V200_int[i]))
#    B2.append(spatial_autocorr_fft(triw,grid,U200[i],V200[i]))
#
#with open('r_200_intfft.pkl', 'wb') as r_200_int:
#    pickle.dump(A2, r_200_int)
#    
#with open('r_200fft.pkl', 'wb') as r_200:
#    pickle.dump(B2, r_200)

# In[]
    
with open('r_200_int3.pkl', 'rb') as r_200_int:
    A = pickle.load(r_200_int)
    
with open('r_2003.pkl', 'rb') as r_200:
    B = pickle.load(r_200)

# In[]
    
with open('df_clust_s_200f.pkl', 'rb') as clust_200:
    df_clust2 = pickle.load(clust_200)

with open('df_clust_v_200f.pkl', 'rb') as clust_200:
    df_clust = pickle.load(clust_200)


# In[PLots]
    
i = 400
plt.triplot(triw, color='black',lw=.5)
plt.tricontourf(triw,U200_int[i],cmap='rainbow')
plt.colorbar()
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
plt.title('Wind speed, U (X) component', fontsize=12, weight='bold')

plt.figure()
plt.triplot(triw, color='black',lw=.5)
plt.tricontourf(triw,V200_int[i],cmap='rainbow')
plt.colorbar()
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')
plt.title('Wind speed, V (Y) component', fontsize=12, weight='bold')

plt.figure()
plt.contourf(tau,eta,np.reshape(A[i][0],(N,N)), 100, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, U', fontsize=12, weight='bold')

plt.figure()
plt.contourf(tau,eta,np.reshape(A[i][1],(N,N)), 100, cmap='rainbow')
plt.colorbar()
plt.ylabel(r'Lag in X, $\tau$ [m]', fontsize=12, weight='bold')
plt.xlabel(r'Lag in Y, $\eta$ [m]', fontsize=12, weight='bold')
plt.title('Spatial autocorrelation, V', fontsize=12, weight='bold')

#plt.figure()
#plt.contourf(tau,eta,np.reshape(B[i][0],(N,N)), 100, cmap='rainbow')
#plt.colorbar()
#
#plt.figure()
#plt.contourf(tau,eta,np.reshape(B[i][1],(N,N)), 100, cmap='rainbow')
#plt.colorbar()

fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

ax.quiver(xmid, ymid, Vt2[i], Ut2[i], units='xy', scale=.03, zorder=3, color='k')

plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

ax.quiver(xmid, ymid, Vt[i], Ut[i], units='xy', scale=.05, zorder=3, color='k')

plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

## In[]
#filt0 = data_filt_LOF(vara_w_df.loc[vara_w_df.scan<50],['CNR','dvdr','dvdt','ws','azim','range_gate'])

scan_n = 8

levels = np.linspace(np.nanmin(filt_CNR(vara_w_df.loc[vara_w_df.scan==scan_n], [-50,-8]).values),np.nanmax(filt_CNR(vara_w_df.loc[vara_w_df.scan==scan_n], [-24,-8]).values),100)
#plt.figure()
#plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),vara_w_df.loc[vara_w_df.scan==scan_n,'ws'].values,levels=levels,cmap='rainbow')
#cbar = plt.colorbar()
#cbar.set_label('$V_{LOS}$ [m/s]', rotation=270)
#plt.title('Raw scan')
#

plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt1.loc[filt1.scan==scan_n,'ws'].values,levels=levels,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan')

plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt2.loc[filt2.scan==scan_n,'ws'].values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan')


plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt_med_sirocco_w.loc[filt_med_sirocco_w.scan==256].ws.values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan')

#plt.figure()
#plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt3.loc[filt3.scan==scan_n,'ws'].values,levels=levels,cmap='rainbow')
#plt.colorbar()
#plt.title('Filtered scan')

plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt_CNR(vara_w_df.loc[vara_w_df.scan==scan_n], [-27,-8]).values,100,cmap='rainbow')
cbar = plt.colorbar(format="%0.1f")
cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
plt.title('CNR Filter (threshold of CNR >= -27)', fontsize=12, weight='bold')
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

#plt.figure()
#plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df0_200[0].loc[scan_n].values,levels=levels,cmap='rainbow')
#cbar = plt.colorbar(format="%0.1f")
#cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
#plt.title('Multivariate filter', fontsize=12, weight='bold')
#plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df1_200[0].loc[scan_n].values,100,cmap='rainbow')
cbar = plt.colorbar(format="%0.1f")
cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
plt.title('Multivariate filter', fontsize=12, weight='bold')
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')


plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),vara_w_df.ws.loc[vara_w_df.scan==scan_n].values,100,cmap='rainbow')
cbar = plt.colorbar(format="%0.1f")
cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
plt.title('Multivariate filter', fontsize=12, weight='bold')
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),vara_w_df.ws.loc[vara_w_df.scan==scan_n].rolling(10,axis=1,min_periods=1,center=True).median().values,100,cmap='rainbow')
cbar = plt.colorbar(format="%0.1f")
cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
plt.title('Multivariate filter', fontsize=12, weight='bold')
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')


plt.figure()
plt.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),vara_w_df.ws.loc[vara_w_df.scan==scan_n].values-vara_w_df.ws.loc[vara_w_df.scan==scan_n].rolling(10,axis=1,min_periods=1,center=True).median().values,100,cmap='rainbow')
cbar = plt.colorbar(format="%0.1f")
cbar.set_label('$V_{LOS}$ [m/s]', rotation=270, labelpad=10, fontsize=12, weight='bold')
plt.title('Multivariate filter', fontsize=12, weight='bold')
plt.ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
plt.xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')


# In[]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
for scan_n in range(0,800):
    ax.cla()
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),filt_med_vara_w_int.loc[scan_n].values,100,cmap='rainbow')
    plt.pause(.1)

 
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
for scan_n in range(700,800):
    ax.cla()
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df0_200[0].loc[scan_n].values,100,cmap='rainbow')
    plt.pause(.2)    


# In[]







#    dd = d.flatten()
#    col = i.flatten()
#    row = np.repeat(np.arange(len(d)),nn)
#    
#    dist = csr_matrix((dd, (row, col)), shape=(len(d),len(d))).toarray()
    
#    print(dist)
    
    #d=np.mean(d,axis=1)

    #kappa = np.log(np.abs(spl.derivative(n=2)(l_resample))/(1+spl.derivative(n=1)(l_resample)**2)**1.5)
    
    #kappa = (kappa-np.min(kappa))/(np.max(kappa)-np.min(kappa)) #RobustScaler(quantile_range=(25, 75)).fit_transform(kappa.reshape(-1,1))



#    plt.figure()
#    #plt.plot(np.abs(np.diff(d_resample,n=2))/(1+np.diff(d_resample,n=1)[:-1]**2)**1.5)
#    
#    plt.plot(l_resample,kappa,marker = 'o')
#    
#    peaks = signal.argrelmax(-kappa)
#    
#    plt.scatter(int(len(d)*x),min(kappa),s=100,color = 'red')
#    
#    plt.scatter(l_resample[peaks],kappa[peaks],s=100,color = 'green')
    
    #clst = np.argmin(np.abs(l_resample[peaks]-int(len(d)*x)))
    
    #print(l_resample[peaks],l_resample[peaks][clst])
    
    #plt.plot([0,X.shape[0]],[m+s,m+s])
       
    #print(m,s,np.min(d_diff))
     
    #eps = np.exp(d[int(len(d)*x)])
    
    #eps = np.exp(d[int(.5*(l_resample[peaks][clst]+int(len(d)*x)))])    

# In[Spectral clustering filter]
    
#def data_filt_spectral(df_ini,features,CNR_value = [-27,-8],method = 'LOF',nn=100,lf=30,clusters=2):
#
#    df = df_ws_grad(df_ini)
#    df = df_mov_med(df,'ws',10)
#    clf = []
#    clf = cluster.SpectralClustering(n_clusters=clusters,
#        affinity="nearest_neighbors",n_neighbors=nn)
#
#     
#    a,b = df.CNR.values.shape 
#    
#    # Data preparation
#    y_pred=np.empty(df.ws.values.flatten().shape)
#    ind = ~np.isnan(df.ws.values.flatten())
#    X = np.empty((sum(ind),len(features)))
#    
#    for i,f in enumerate(features):
#        if (f =='azim') | (f =='scan') :
#            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
#        else:
#            X[:,i] = df[f].values.flatten().astype('float')[ind]
#    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)         
#    # Prediction
#    r1 = df.ws.copy()
#    r2 = df.ws.copy()
#    r3 = df.ws.copy()
#    y_pred[ind] = clf.fit_predict(X)
#    y_pred[~ind] = 2
#    
#    y_pred = np.reshape(y_pred,(a,b))
#    
#    print(clf.fit_predict(X))
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    
#
#    ax.scatter( X[y_pred.flatten()[ind]==0,0], X[y_pred.flatten()[ind]==0,1], X[y_pred.flatten()[ind]==0,2], color='blue')
#    ax.scatter( X[y_pred.flatten()[ind]==1,0], X[y_pred.flatten()[ind]==1,1], X[y_pred.flatten()[ind]==1,2], color='red')
#    #ax.scatter( X[y_pred.flatten()==2,0], X[y_pred.flatten()==2,1], X[y_pred.flatten()==2,2], color='green')
#    
#    #ax.scatter( X[y_pred.flatten()==1,0], X[y_pred.flatten()==1,1], color='red')
#    
#    mask1 = pd.DataFrame(y_pred!=0)
#    mask1.columns = ['ws']*b
#    r1.ws = df.ws.copy().mask(mask1)
#    r1['scan'] = df.scan
#    
#    mask2 = pd.DataFrame(y_pred!=1)
#    mask2.columns = ['ws']*b
#    r2.ws = df.ws.copy().mask(mask2)
#    r2['scan'] = df.scan
#    
#    mask3 = pd.DataFrame(y_pred!=2)
#    mask3.columns = ['ws']*b
#    r3.ws = df.ws.copy().mask(mask3)
#    r3['scan'] = df.scan
#    
#    return (r1,r2,r3,y_pred)
    
# In[]
#
#def data_filt_LOF(df_ini,features,CNR_value = [-24,-8],method = 'LOF',nn=20,lf=30,out_frac = 0.1, lim=1.5, loops=1):
#    
#
#    df = df_ws_grad(df_ini)
#    df = df_mov_med(df,'ws',15)
#    
#    #df['movmed'] = pd.DataFrame(df.movmed.values/df.Sb.values, columns=df.movmed.columns, index=df.movmed.index)
#    
#    clf = []
#    clf = LocalOutlierFactor(n_neighbors=nn,contamination=out_frac,leaf_size=lf,p=1)
#        
#    a,b = df.CNR.values.shape 
#    X = np.empty((a*b,len(features)))
#    # Data preparation
#    
#
#    ind = ~np.isnan(df.ws.values.flatten())
#
#    X = np.empty((sum(ind),len(features)))
#    
#    for i,f in enumerate(features):
#        if (f =='azim') | (f =='scan') :
#            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
#
#        else:
#            X[:,i] = df[f].values.flatten()[ind]
#
#    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
#    
#    ind_LOF = ind
#    
#    LOF = np.zeros(ind_LOF.shape)
#    
#    for i in range(0,loops):
#        
#        print(sum(ind_LOF),lim)
#    
#        LOF_aux=-clf.fit(X[ind_LOF,:]).negative_outlier_factor_
#        
#        LOF[ind_LOF] = LOF_aux
#    
#        ind_LOF = (LOF<lim) & ind_LOF
#        
#        lim*=0.95
#        
#
#    #df['movmed'] = pd.DataFrame(df.movmed.values*df.Sb.values, columns=df.movmed.columns, index=df.movmed.index)
#    
#    for i,f in enumerate(features):
#        if (f =='azim') | (f =='scan') :
#            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
#        else:
#            X[:,i] = df[f].values.flatten()[ind]
#    
#    #y_pred = np.reshape(y_pred,(a,b))
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
##    ax.scatter( X[y_pred.flatten()[ind]==-1,0], X[y_pred.flatten()[ind]==-1,1], X[y_pred.flatten()[ind]==-1,2], color='blue')
##    ax.scatter( X[y_pred.flatten()[ind]==1,0], X[y_pred.flatten()[ind]==1,1], X[y_pred.flatten()[ind]==1,2], color='red')
#    #ax.scatter(X[:,0],X[:,1],X[:,2],c=c,cmap='rainbow')
#    #plt.colorbar()
#
#    ax.scatter( X[LOF[ind]>=lim,0], X[LOF[ind]>=lim,1], X[LOF[ind]>=lim,2], color='blue')
#    ax.scatter( X[LOF[ind]<lim,0], X[LOF[ind]<lim,1], X[LOF[ind]<lim,2], color='red')
#       
#    #mask = pd.DataFrame(y_pred==-1)
#    
#    LOF=np.reshape(LOF,(a,b))
#    
#    mask = pd.DataFrame(LOF>=lim)
#    
#    mask.columns = ['ws']*b
#    mask=mask.set_index(df.ws.index)
#    r = df.ws.copy().mask(mask)
#    r['scan'] = df.scan
#    
#    return (r,LOF)
    
# In[]
scan_n=850

loc = (vara_w_df.scan >= scan_n) & (vara_w_df.scan <= scan_n+5)
filt_mask = data_filt_DBSCAN(vara_w_df.loc[loc],feat,nn = 5,plot=True)

plt.figure()
plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),filt_med_vara_w.loc[filt_med_vara_w.scan==scan_n,'ws'].values,200,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan median, Line-of-Sight wind speed [m/s]')
plt.xlabel('West-East [m]', fontsize=12, weight='bold')
plt.ylabel('North-South [m]', fontsize=12, weight='bold')



plt.figure()

plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),filt_CNR(vara_w_df.loc[vara_w_df.scan==scan_n], [-27,-8]).values,200,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan CNR threshold, Line-of-Sight wind speed [m/s]')
plt.xlabel('west-east [m]', fontsize=12, weight='bold')
plt.ylabel('north-south [m]', fontsize=12, weight='bold')


plt.figure()
plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),vara_w_df.loc[vara_w_df.scan==scan_n].ws.mask(filt_mask).values,200,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan DBSCAN, Line-of-Sight wind speed [m/s]')
plt.xlabel('west-east [m]', fontsize=12, weight='bold')
plt.ylabel('north-south [m]', fontsize=12, weight='bold')

plt.figure()
plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),vara_w_df.loc[vara_w_df.scan==scan_n].ws.values,200,cmap='rainbow')
plt.colorbar()
plt.title('Non-filtered, Line-of-Sight wind speed [m/s]')
plt.xlabel('west-east [m]', fontsize=12, weight='bold')
plt.ylabel('north-south [m]', fontsize=12, weight='bold')

#plt.figure()
#
#plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),vara_w_df.ws.loc[vara_w_df.scan==scan_n].values,100,cmap='rainbow')
#plt.colorbar()

# In[]

plt.figure()


plt.contourf(r_siw*np.sin(phi_siw),r_siw*np.cos(phi_siw),df_clust2.loc[df_clust2.scan==scan_n,'ws'].values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan clustering')

plt.figure()


plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),df_clust.loc[df_clust.scan==scan_n,'ws'].values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan clustering')



plt.figure()
plt.contourf(r_vaw*np.sin(phi_vaw),r_vaw*np.cos(phi_vaw),filt_med_vara_w.loc[filt_med_vara_w.scan==scan_n,'ws'].values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan clustering')

plt.figure()
plt.contourf(r_siw*np.sin(phi_siw),r_siw*np.cos(phi_siw),filt_med_sirocco_w.loc[filt_med_sirocco_w.scan==scan_n,'ws'].values,100,cmap='rainbow')
plt.colorbar()
plt.title('Filtered scan clustering')

# In[]
scan_n = 300
#levels1 = np.linspace(-18,15,100)
#levels2 = np.linspace(-15,15,100)
   
fig, (ax1, ax2) = plt.subplots(ncols=2)    

ax1.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax1.use_sticky_edges = False
ax1.margins(0.07)
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax2.use_sticky_edges = False
ax2.margins(0.07)


ax1.triplot(triw, color='black',lw=.1)
ax2.triplot(triw, color='black',lw=.1)

im1 = ax1.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=np.linspace(5,10,300),cmap='rainbow')
#ax1.set_title('Scan = %i' %i)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.set_ylabel("U comp. [m/s]")
ax1.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax1.set_xlabel(r'North-South [m]', fontsize=12, weight='bold')
tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
Q = ax1.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold')

im2 = ax2.tricontourf(triw,dUdx[scan_n]+dVdy[scan_n],levels=np.linspace(-.01,0.01,300),cmap='rainbow')
#ax2.set_title('Scan = %i' %i)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.set_ylabel("V comp. [m/s]")
#ax2.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax2.set_xlabel(r'North-South [m]', fontsize=12, weight='bold')
# In[]
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

#plt.title('Scan = %scan_n' %scan_n)
tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
ax.triplot(triw, color='black',lw=.2)
im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=np.linspace(5,12,200),cmap='rainbow')
cb = plt.colorbar(im)
cb.ax.set_ylabel("Wind speed [m/s]")
Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='grey')#, scale=.05, zorder=3, color='b')
ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold')  

# In[]
scan_n = 300
N=50
alpha=.6

f = 24

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

rho = spatial_autocorr(tri_r,Ur[scan_n],Vr[scan_n],N,alpha)

rhoU=np.reshape(rho[0],(N,N))
rhoV=np.reshape(rho[1],(N,N))

tau = np.linspace(-alpha*(np.max(triw.x)-np.min(triw.x)),alpha*(np.max(triw.x)-np.min(triw.x)),N)
eta = np.linspace(-alpha*(np.max(triw.x)-np.min(triw.x)),alpha*(np.max(triw.y)-np.min(triw.y)),N)

fig, (ax1, ax2) = plt.subplots(ncols=2)    

ax1.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax1.use_sticky_edges = False
ax1.margins(0.05)
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax2.use_sticky_edges = False
ax2.margins(0.05)

im1 = ax1.contourf(tau,eta,rhoU, levels=np.linspace(-1.0,1.0,249), cmap='rainbow')

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)

cbar1 = fig.colorbar(im1, cax=cax1, format=ticker.FuncFormatter(fm))

cbar1.ax.tick_params(labelsize=f)
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=f,
        verticalalignment='top')
#cbar1.ax.set_ylabel("U $[m/s]$")
ax1.tick_params(labelsize=f)
ax1.set_ylabel(r'Lag in West-East, $\tau$ [m]', fontsize=f, weight='bold')
ax1.set_xlabel(r'Lag in North-South, $\eta$ [m]', fontsize=f, weight='bold')

im2 = ax2.contourf(tau,eta,rhoV, levels=np.linspace(-1.0,1.0,249), cmap='rainbow')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2, format=ticker.FuncFormatter(fm))

cbar2.ax.tick_params(labelsize=f)

ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=f,
        verticalalignment='top')

#cbar2.ax.set_ylabel("$\rho$ ")
ax2.tick_params(labelsize=f)
#ax2.set_ylabel(r'Lag in West-East, $\tau$ [m]', fontsize=f, weight='bold')
ax2.set_xlabel(r'Lag in North-South, $\eta$ [m]', fontsize=f, weight='bold')


 # In[]
scan_n=300
levels=np.linspace(4,12,300)
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

#plt.title('Scan = %scan_n' %scan_n)
tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
ax.triplot(triw, color='black',lw=.2)
#im=ax.tricontourf(triw,np.sqrt(U200_clust_int1[scan_n]**2+V200_clust_int1[scan_n]**2),levels=levels,cmap='rainbow')
im=ax.tricontourf(triw,dUdx[scan_n]+dVdy[scan_n],levels=np.linspace(-.01,0.01,300),cmap='rainbow')
divider = make_axes_locatable(ax)
cb = fig.colorbar(im)
cb.ax.set_ylabel("Wind speed [m/s]")
#Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 




for scan_n in range(301,800):     
    ax.cla()   
    #fig.title('Scan = %scan_n' %scan_n)
    #tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
    #tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
    
    ax.triplot(triw, color='black',lw=.5)
    
    im=ax.tricontourf(triw,dUdy[scan_n]-dVdx[scan_n],levels=levels,cmap='rainbow')
    #cb = plt.colorbar(im)
    #Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data, scale=400, units='width', color='grey')#, scale=.05, zorder=3, color='b')

    #check if there is more than one axes
    if len(fig.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts = fig.axes[-1].get_position().get_points()
        # and its label
        label = fig.axes[-1].get_ylabel()
        # and then remove the axes
        fig.axes[-1].remove()
        # then we draw a new axes a the extents of the old one
        divider = make_axes_locatable(ax)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.set_ylabel("Wind speed [m/s]")
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        fig.colorbar(im)
    
    
    plt.pause(.3) 
# In[] 
class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter("%.2f")
        
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.animation as animation
scan_n=500
levels=np.linspace(4,12,300)
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)

refiner = UniformTriRefiner(triw) 
tri_r = refiner.refine_triangulation(return_tri_index=False, subdiv=2)
xmid_r = tri_r.x[tri_r.triangles].mean(axis=1)
ymid_r = tri_r.y[tri_r.triangles].mean(axis=1)

aux0 = Ur[scan_n].data + np.nanmean(U200_clust_int1[scan_n])
aux1 = Vr[scan_n].data + np.nanmean(V200_clust_int1[scan_n])
ind = np.isnan(aux0)
aux0[ind]=-100
ind = np.isnan(aux1)
aux1[ind]=-100
#plt.title('Scan = %scan_n' %scan_n)
tciU = CubicTriInterpolator(tri_r, aux0, kind = 'geom')
tciV = CubicTriInterpolator(tri_r, aux1, kind = 'geom')
ax.triplot(tri_r, color='black',lw=.1)
im=ax.tricontourf(tri_r,np.sqrt(aux0**2+aux1**2),levels=levels,cmap='rainbow')
divider = make_axes_locatable(ax)
cax= divider.append_axes("right", size="5%", pad=0.05)
cb = fig.colorbar(im, cax=cax)
cb.ax.set_ylabel("Wind speed [m/s]")

U_mean = avetriangles(np.c_[triw.x,triw.y], U200_clust_int1[scan_n], triw.triangles)
V_mean = avetriangles(np.c_[triw.x,triw.y], V200_clust_int1[scan_n], triw.triangles)

Q = ax.quiver(3000.00,-830.00,U_mean,V_mean,pivot='middle', scale=75, units='width', color='k')
circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
ax.add_artist(circle)
#Q = ax.quiver(xmid_r, ymid_r,  tciU(xmid_r,ymid_r).data, tciV(xmid_r,ymid_r).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 


def animate(i):
    i=i+500
    ax.cla()  
    ax.set_aspect('equal')
    # Enforce the margins, and enlarge them to give room for the vectors.
    ax.use_sticky_edges = False
    ax.margins(0.07)
    #ax.set_title('Scan = %i' %i)
    aux0 = Ur[i].data + np.nanmean(U200_clust_int1[i])
    aux1 = Vr[i].data + np.nanmean(V200_clust_int1[i])
    ind = np.isnan(aux0)
    aux0[ind]=-100
    ind = np.isnan(aux1)
    aux1[ind]=-100
    tciU = CubicTriInterpolator(tri_r, aux0, kind = 'geom')
    tciV = CubicTriInterpolator(tri_r, aux1, kind = 'geom')

#    tciU = CubicTriInterpolator(triw, U200_clust_int1[i],kind = 'geom')
#    tciV = CubicTriInterpolator(triw, V200_clust_int1[i],kind = 'geom')
    ax.triplot(tri_r, color='black',lw=.1)
    
    im=ax.tricontourf(tri_r,np.sqrt(aux0**2+aux1**2),levels=np.linspace(4,12,300),cmap='rainbow')
    #im=ax.tricontourf(triw,dUdy[scan_n]-dVdx[scan_n],levels=np.linspace(-.01,.01,300),cmap='rainbow')
    #cb = plt.colorbar(im)
    #Q = ax.quiver(xmid, ymid,  tciU(xmid,ymid).data, tciV(xmid,ymid).data, scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
    U_mean = avetriangles(np.c_[triw.x,triw.y], U200_clust_int1[i], triw.triangles)
    V_mean = avetriangles(np.c_[triw.x,triw.y], V200_clust_int1[i], triw.triangles)

    Q = ax.quiver(3000.00,-830.00,U_mean,V_mean,pivot='middle', scale=75, units='width', color='k')
    circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
    ax.add_artist(circle)
    #check if there is more than one axes
    if len(fig.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts = fig.axes[-1].get_position().get_points()
        # and its label
        label = fig.axes[-1].get_ylabel()
        # and then remove the axes
        fig.axes[-1].remove()
        # then we draw a new axes a the extents of the old one
        divider = make_axes_locatable(ax)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.set_ylabel("Wind speed [m/s]")
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        fig.colorbar(im)

#interval = 1#in seconds     
#ani = animation.FuncAnimation(fig,animate,50,interval=interval*500,blit=False)
#
#plt.show()
#
#
#
#mywriter = animation.ImageMagickFileWriter(fps=1)
#ani.save('animationU.gif',writer=mywriter)

interval = 1#in seconds     
ani = animation.FuncAnimation(fig,animate,100,interval=interval*100,blit=False)

plt.show()

mywriter = animation.FFMpegFileWriter(fps=3)
ani.save('animationU.mp4',writer=mywriter)

# In[]
loc = (vara_w_df.scan<50)
plt.figure()
plt.scatter(vara_w_df.loc[loc].ws.values.flatten(),vara_w_df.loc[loc].CNR.values.flatten(),marker = 'o', c = 'm', edgecolor='k')
plt.xlabel(r'$V_{LOS}$ [m/s]', fontsize=12, weight='bold')
plt.ylabel('CNR', fontsize=12, weight='bold')
plt.grid()
plt.xlim(-35,35)
plt.plot([-35,35],[-27,-27],lw=2,color='r')


# In[]
scan_n = 500
refiner = UniformTriRefiner(triw) 
tri_r = refiner.refine_triangulation(return_tri_index=False, subdiv=2)
xmid_r = tri_r.x[tri_r.triangles].mean(axis=1)
ymid_r = tri_r.y[tri_r.triangles].mean(axis=1)

#for i in range(scan_n,scan_n+2):
#    print(i)
#    Lidar_sir = (df_clust2.ws.loc[df_clust2.scan==i],phi_siw,wsiw,neighsiw,indexsiw) 
#    Lidar_var = (df_clust.ws.loc[df_clust.scan==i],phi_vaw,wvaw,neighvaw,indexvaw)
#    auxU, auxV,_,_ = wind_field_rec(Lidar_var, Lidar_sir, treew, triw, d)
#    
#auxU, auxV = data_interp_triang(auxU, auxV,triw.x,triw.y,45)
#    
#tciU = CubicTriInterpolator(triw, auxU[1], kind = 'geom')
#tciV = CubicTriInterpolator(triw, auxV[1], kind = 'geom')

DUDX = []
DUDY = []
DVDX = []
DVDY = []
Ur = []
Vr = []

for scan_n in range(0,1410):
    print(scan_n)
    if len(U200_clust_int1[scan_n])>0:
        tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n]-np.nanmean(U200_clust_int1[scan_n]), kind = 'geom')
        tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n]-np.nanmean(V200_clust_int1[scan_n]), kind = 'geom')
    
        Ura = tciU(tri_r.x,tri_r.y)
        Vra = tciV(tri_r.x,tri_r.y)
        
        DUDXa = tciU.gradient(tri_r.x, tri_r.y)[0]
        DUDYa = tciU.gradient(tri_r.x, tri_r.y)[1]
        DVDXa = tciV.gradient(tri_r.x, tri_r.y)[0]
        DVDYa = tciV.gradient(tri_r.x, tri_r.y)[1]
        
        DUDXa[np.isnan(DUDXa.data)] = 0
        DUDYa[np.isnan(DUDYa.data)] = 0
        DVDXa[np.isnan(DVDXa.data)] = 0
        DVDYa[np.isnan(DVDYa.data)] = 0
        
        DUDX.append(DUDXa)
        DUDY.append(DUDYa)
        DVDX.append(DVDXa)
        DVDY.append(DVDYa)
        Ur.append(Ura)
        Vr.append(Vra)
    else:
        DUDX.append([])
        DUDY.append([])
        DVDX.append([])
        DVDY.append([])
        Ur.append([])
        Vr.append([])

# In[]
    
with open('DUDX_clust.pkl', 'wb') as dU_dx:
    pickle.dump(DUDX, dU_dx)
    
with open('DUDY_clust.pkl', 'wb') as dU_dy:
    pickle.dump(DUDY, dU_dy)

with open('DVDX_clust.pkl', 'wb') as dV_dx:
    pickle.dump(DVDX, dV_dx)
    
with open('DVDY_clust.pkl', 'wb') as dV_dy:
    pickle.dump(DVDY, dV_dy)
    
with open('Ur.pkl', 'wb') as U_t:
    pickle.dump(Ur, U_t)

with open('Vr.pkl', 'wb') as V_t:
    pickle.dump(Vr, V_t)


# In[]    
with open('U200_clust_int.pkl', 'rb') as U_200_clust:
    U200_clust_int1 = pickle.load(U_200_clust)
    
with open('V200_clust_int.pkl', 'rb') as V_200_clust:
    V200_clust_int1 = pickle.load(V_200_clust)  
     
with open('U200_int_med.pkl', 'rb') as U_200_med:
    U200_int_med = pickle.load(U_200_med)
    
with open('V200_int_med.pkl', 'rb') as V_200_med:
    V200_int_med = pickle.load(V_200_med)   

with open('Ur.pkl', 'rb') as U_r:
    Ur = pickle.load(U_r)
    
with open('Vr.pkl', 'rb') as V_r:
    Vr = pickle.load(V_r)
    
with open('DUDX_clust.pkl', 'rb') as dU_dx:
    DUDX = pickle.load(dU_dx)
    
with open('DUDY_clust.pkl', 'rb') as dU_dy:
    DUDY = pickle.load(dU_dy)

with open('DVDX_clust.pkl', 'rb') as dV_dx:
    DVDX = pickle.load(dV_dx)
    
with open('DVDY_clust.pkl', 'rb') as dV_dy:
    DVDY = pickle.load(dV_dy)
    
# In[]
scan_n = 500
f=24
refiner = UniformTriRefiner(triw) 
tri_r = refiner.refine_triangulation(return_tri_index=False, subdiv=2)

fig, (ax1, ax2) = plt.subplots(ncols=2) 
ax1.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax1.use_sticky_edges = False
ax1.margins(0.05)
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax2.use_sticky_edges = False
ax2.margins(0.05)


ax1.triplot(tri_r, color='black',lw=.05)
#ax2.triplot(tri_r, color='black',lw=.05)
#
#aux0 = Ur[scan_n].data + np.nanmean(U200_clust_int1[scan_n])
#ma = np.nanmax(aux0)
#mi = np.nanmin(aux0)
#ind = np.isnan(aux0)
#aux0[ind]=-100
#
#aux1 = Vr[scan_n].data + np.nanmean(V200_clust_int1[scan_n])
#ma = np.nanmax(aux1.data)
#mi = np.nanmin(aux1.data)
#aux1[np.isnan(aux1)]=-100


aux0 = Ur[scan_n].data + np.nanmean(U200_clust_int1[scan_n])
aux1 = Vr[scan_n].data + np.nanmean(V200_clust_int1[scan_n])
aux = np.sqrt(aux0**2+aux1**2)
ma = np.nanmax(aux.data)
mi = np.nanmin(aux.data)
ind = np.isnan(aux0)
aux[ind]=-100
#
#
#ma = np.nanmax(aux1.data)
#mi = np.nanmin(aux1.data)
#aux1[np.isnan(aux1)]=-100


#im1 = ax1.tricontourf(tri_r, aux0, levels=np.linspace(mi,ma,300),cmap='rainbow')

im1 = ax1.tricontourf(tri_r, aux, levels=np.linspace(mi,ma,300),cmap='rainbow')

Q = ax1.quiver(3000.00,-830.00,np.nanmean(U200_clust_int1[scan_n]),
               np.nanmean(V200_clust_int1[scan_n]),pivot='middle', scale=75, units='width', color='k')

circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
ax1.add_artist(circle)
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=f,
        verticalalignment='top')
#ax1.set_title('Scan = %i' %i)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1, format=ticker.FuncFormatter(fmt))

cbar1.ax.tick_params(labelsize=f)
ax1.tick_params(labelsize=f)

cbar1.ax.set_ylabel("Wind speed [m/s]", fontsize=18)
ax1.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax1.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')
#tciU = CubicTriInterpolator(triw, U200_clust_int1[scan_n],kind = 'geom')
#tciV = CubicTriInterpolator(triw, V200_clust_int1[scan_n],kind = 'geom')
#Q = ax1.quiver(xmid, ymid,  tciU(xmid_r,ymid_r).data, tciV(xmid_r,ymid_r).data,scale=400, units='width', color='w')#, scale=.05, zorder=3, color='b')
#ax1.set_ylabel(r'West-East [m]', fontsize=18, weight='bold')
#ax1.set_xlabel(r'North-South [m]', fontsize=18, weight='bold')

#aux = .5*(DUDY[scan_n]-DVDX[scan_n])**2






im2 = ax2.contourf(tau,eta,rhoV, levels=np.linspace(-1.0,1.0,249), cmap='rainbow')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2, format=ticker.FuncFormatter(fm))

cbar2.ax.tick_params(labelsize=f)

ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=f,
        verticalalignment='top')

#cbar2.ax.set_ylabel("$\rho$ ")
ax2.tick_params(labelsize=f)
ax2.set_ylabel(r'Lag in West-East, $\tau$ [m]', fontsize=f, weight='bold')
ax2.set_xlabel(r'Lag in North-South, $\eta$ [m]', fontsize=f, weight='bold')
# In[]


im2 = ax2.tricontourf(tri_r,aux1,levels=np.linspace(mi,ma,300),cmap='rainbow')
#ax2.set_title('Scan = %i' %i)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2, format=ticker.FuncFormatter(fm))

cbar2.ax.tick_params(labelsize=f)
ax2.tick_params(labelsize=f)

#cbar2.ax.set_ylabel("V component [m/s]", fontsize=f)
#ax2.set_ylabel(r'West-East [m]', fontsize=f, weight='bold')
ax2.set_xlabel(r'North-South [m]', fontsize=f, weight='bold')

Q = ax2.quiver(3000.00,-830.00,np.nanmean(U200_clust_int1[scan_n]),
               np.nanmean(V200_clust_int1[scan_n]),pivot='middle', scale=75, units='width', color='k')

circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
ax2.add_artist(circle)
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=f,
        verticalalignment='top')

#tciU = CubicTriInterpolator(tri_r, aux0, kind = 'geom')
#tciV = CubicTriInterpolator(tri_r, aux1, kind = 'geom')
#
#xmid_r = tri_r.x[tri_r.triangles].mean(axis=1)
#ymid_r = tri_r.y[tri_r.triangles].mean(axis=1)
#
#index = (np.linspace(0,1,500)*(len(xmid_r)-1)).astype(int)

#Q = ax2.quiver(xmid_r[index], ymid_r[index], tciU(xmid_r[index], ymid_r[index]), tciV(xmid_r[index], ymid_r[index]), scale=100, units='width', color='w')

# In[]  
def animate(i):
    i=i+500
    #ax.cla()        
#for scan_n in range(501,800):
    #print(i)     
    if len(fig.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts1 = fig.axes[0].get_position().get_points()
        pts2 = fig.axes[1].get_position().get_points()
        pts3 = fig.axes[2].get_position().get_points()
        pts4 = fig.axes[3].get_position().get_points()
        # and its label
#        label1 = fig.axes[2].get_ylabel()
#        label2 = fig.axes[3].get_ylabel()
        # and then remove the axes
        fig.axes[0].remove()
        fig.axes[0].remove()
        fig.axes[0].remove()
        fig.axes[0].remove()
        
        cax1= fig.add_axes([pts1[0][0],pts1[0][1],pts1[1][0]-pts1[0][0],pts1[1][1]-pts1[0][1]  ])
        cax2= fig.add_axes([pts2[0][0],pts2[0][1],pts2[1][0]-pts2[0][0],pts2[1][1]-pts2[0][1]  ])
        cax3= fig.add_axes([pts3[0][0],pts3[0][1],pts3[1][0]-pts3[0][0],pts3[1][1]-pts3[0][1]  ])
        cax4= fig.add_axes([pts4[0][0],pts4[0][1],pts4[1][0]-pts4[0][0],pts4[1][1]-pts4[0][1]  ])
        
        cax1.triplot(tri_r, color='black',lw=.1)
        cax2.triplot(tri_r, color='black',lw=.1)
        
        cax1.set_aspect('equal')
        # Enforce the margins, and enlarge them to give room for the vectors.
        cax1.use_sticky_edges = False
        cax1.margins(0.07)
        
        cax2.set_aspect('equal')
        # Enforce the margins, and enlarge them to give room for the vectors.
        cax2.use_sticky_edges = False
        cax2.margins(0.07)
        
        im1 = cax1.tricontourf(tri_r,np.sqrt(Ur[i]**2+Vr[i]**2),levels=np.linspace(0,4,300),cmap='rainbow')
        im2 = cax2.tricontourf(tri_r,.5*(DUDY[i]-DVDX[i])**2,levels=np.linspace(0,0.00015,300),cmap='rainbow')

#        # then we draw a new axes a the extents of the old one
#        ax1.set_aspect('equal')
#        # Enforce the margins, and enlarge them to give room for the vectors.
#        ax1.use_sticky_edges = False
#        ax1.margins(0.07)
#        
#        ax1.triplot(triw, color='black',lw=.5)
#        
#        im1 = cax1.tricontourf(triw,dVdx[i]-dUdy[i],cmap='rainbow')
#        
#        cax1= fig.add_axes([pts1[0][0],pts1[0][1],pts1[1][0]-pts1[0][0],pts1[1][1]-pts1[0][1]  ])
        # and add a colorbar to it
        cbar1 = fig.colorbar(im1, cax=cax3)
        cbar2 = fig.colorbar(im2, cax=cax4)
        cbar1.ax.set_ylabel("TKE $[m^2/s^2]$")
        cbar2.ax.set_ylabel("Enstrophy $[s^{-2}]$")
#        cbar1.ax.set_ylabel(label1)
#        cbar2.ax.set_ylabel(label2)
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
        #print(fig.axes)
    else:
        plt.colorbar(im)
    
    
#    plt.pause(.1)
        
interval = 1#in seconds     
ani = animation.FuncAnimation(fig,animate,300,interval=interval*100,blit=False)

plt.show()

mywriter = animation.FFMpegFileWriter(fps=3)
ani.save('animationHD.mp4',writer=mywriter)
