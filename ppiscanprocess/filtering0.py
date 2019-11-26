# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:08:50 2018

Package for line-of-sight wind speed filtering and interpolation in long range PPI scans.
The algorithms here presented where tested with the balcony experiment data. 

Four filters are implemented:
    
    1.- CNR threshold: 
    2.- Median-like filter
    3.- Kernel density filter (not used)
    4.- Clustering via DBSCAN

@author: 
Leonardo Alcayaga
lalc@dtu.dk

"""
# In[Packages used]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
from scipy.spatial import Delaunay
from scipy.signal import find_peaks

# In[############# [Functions] #################]

# In[DataFrame manipulation and misc]

def runningmedian(x,N):
    idx = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    b = [row[row>0] for row in x[idx]]
    return np.array([np.median(c) for c in b])

def df_ws_grad(df):
    """
    Function that calculates the average difference in V_LOS between a
    particular point and its neighbours. V_LOS difference in the same point along ssuccesive
    scans is also calculated. Additional features for each point will be then callled dvdr and dvdt.
    
    Could be inproved estimating an actual gradient instead of
    just a difference
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
    Output:
    ------
        df_prime - modified Pandas dataframe with additional dvdr and dvdt
                   columns contaning the average of V_LOS difference in 
                   space ant time.  
    """
    a,b = df.CNR.values.shape
    
    phi_g = np.array([list(df.azim.values),]*b).transpose()
    
    rho_g = df.range_gate.values
    
    
    x = pd.DataFrame(rho_g*np.cos(phi_g*np.pi/180))
    y = pd.DataFrame(rho_g*np.sin(phi_g*np.pi/180))
    x.columns = df.ws.columns
    y.columns = df.ws.columns
    x.index = df.ws.index
    y.index = df.ws.index
    
    #Azimuth angles
    
    phi = df.azim.unique()
    #V_LOS difference forward in range gate
    dx = x.diff(axis=1)
    dy = y.diff(axis=1)
    dv = df.ws.diff(axis=1) 
    #dvdr1 = (x/np.sqrt(x**2+y**2))*dv/dx + (x/np.sqrt(x**2+y**2))*dv/dy
    dvdx1 = dv/dx
    dvdy1 = dv/dy
    #dvdr1 = df.ws.diff(axis=1).div(np.sqrt(x.diff(axis=1)**2+y.diff(axis=1)**2))
    
    #V_LOS difference backward in range gate
    
    dx = x.diff(axis=1,periods=-1)
    dy = y.diff(axis=1,periods=-1)
    dv = df.ws.diff(axis=1,periods=-1) 
    #dvdr2 = (x/np.sqrt(x**2+y**2))*dv/dx + (x/np.sqrt(x**2+y**2))*dv/dy
    dvdx2 = dv/dx
    dvdy2 = dv/dy
    #dvdr2 = df.ws.diff(axis=1,periods=-1).div(np.sqrt(x.diff(axis=1,periods=-1)**2+y.diff(axis=1,periods=-1)**2))
    #V_LOS difference forward in azimuth
    
    dx = x.diff(axis=0)
    dy = y.diff(axis=0)
    dv = df.ws.diff(axis=0) 
    #dvdr3 = (x/np.sqrt(x**2+y**2))*dv/dx + (x/np.sqrt(x**2+y**2))*dv/dy
    dvdx3 = dv/dx
    dvdy3 = dv/dy
    #dvdr3 = df.ws.diff(axis=0).div(np.sqrt(x.diff(axis=0)**2+y.diff(axis=0)**2))
    dvdx3.loc[df.azim==phi[0]] = np.nan
    dvdy3.loc[df.azim==phi[0]] = np.nan
    #dvdr3.loc[df.azim==phi[0]] = np.nan
    #V_LOS difference backward in azimuth
    
    dx = x.diff(axis=0,periods=-1)
    dy = y.diff(axis=0,periods=-1)
    dv = df.ws.diff(axis=0,periods=-1) 
    #dvdr4 = (x/np.sqrt(x**2+y**2))*dv/dx + (x/np.sqrt(x**2+y**2))*dv/dy
    dvdx4 = dv/dx
    dvdy4 = dv/dy
    #dvdr4 = df.ws.diff(axis=0,periods=-1).div(np.sqrt(x.diff(axis=0,periods=-1)**2+y.diff(axis=0,periods=-1)**2))
    dvdx4.loc[df.azim==phi[0]] = np.nan
    dvdy4.loc[df.azim==phi[0]] = np.nan
    #dvdr4.loc[df.azim==phi[-1]] = np.nan
    # Average over diferences
    dvdx = pd.concat([dvdx1, dvdx2, dvdx3, dvdx4]).groupby(level=0).median()
    dvdy = pd.concat([dvdy1, dvdy2, dvdy3, dvdy4]).groupby(level=0).median()
    #dvdr = pd.concat([dvdr1, dvdr2, dvdr3, dvdr4]).groupby(level=0).median() 
    dvdx.columns = ['dvdx']*(df['ws'].shape[1])
    dvdy.columns = ['dvdy']*(df['ws'].shape[1])
    #dvdr.columns = ['dvdr']*(df['ws'].shape[1])
    # V_LOS change in time (between sucessive scans)
    dvdt = df.ws.diff(axis=0,periods=45).fillna(value=0.0)  
    dvdt.columns = ['dvdt']*(df['ws'].shape[1])
    #Additional columns
    #dv = pd.concat([dvdr,dvdt], axis=1)
    dv = pd.concat([dvdx,dvdy,dvdt], axis=1)
    df_prime = pd.concat([df,dv], axis=1)
    return(df_prime)

def df_mov_med(df,col,n):
    """
    Function that calculates moving average of the specified feature
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
                   
        col      - Feature in df on which moving median is estimated.
        
        n        - Moving window size.
        
    Output:
    ------
        df_prime - modified Pandas dataframe with additional column containing 
                   the moving median of specific feature.  
    """

    movmedian = df[col].rolling(n,axis=1,min_periods=1).median()
    movmedian = df[col].sub(movmedian).abs()
    movmedian.columns = ['movmed']*(df[col].shape[1])
    df_prime = pd.concat([df,movmedian], axis=1)
    
    return(df_prime)

# In[############## Filtering functions ################]
    
# In[CNR threshold]
    
def filt_CNR(df,value=[-27.0,-8.0]):
    """
    Filter based on a CNR threshold, measurements with a Carrier-to-Noise value less than value are
    rejected. Acts on V_LOS.
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
        
        value    - CNR threshold value.
        
    Output:
    ------
        df_prime - Filtered Pandas dataframe. 
    """
    #Output DataFrame
    df_prime=df.copy()
    # Extraction of V_LOS from input DataFrame
    vel = df_prime.ws
    # Mask for V_LOS values with CNR values outside the valid interval
    mask_cnr = (df.CNR<=value[0]) | (df.CNR>=value[1])
    # Convertion of Mask DataFrame columns and indexes to input-like 
    mask_cnr.columns = vel.columns
    # Output DataFrame masking
    df_prime.ws = vel.mask(mask_cnr)
    return df_prime

# In[Median-like filter]

def data_filt_median(df,feature='ws',lim_m=6,lim_g=15,n=10):
    """
    Filter based on 2-level thresholds for the difference in V_LOS and the moving V_LOS median
    (1st-level filter, acting in both, line-of-sight and azimuth components), and a global V_LOS 
    median (2nd-level filter, acting only in line-of-sight component).
    
    Input:
    -----
        df       - Pandas dataframe containing the positional information
                   (range gate and azimuth angle) and line-of-sight wind speed.
        
        feature  - Feature to use to estimate the median. Default is V_LOS,
                   which is what we are interested in now.
        
        lim_m    - Threshold for the differnce between measured V_LOS and moving median of V_LOS.
        
        lim_g    - Threshold for the differnce between measured V_LOS and global median of V_LOS.
        
        n        - Window size for moving median calculation.
        
    Output:
    ------
        df_prime - Filtered Pandas dataframe. 
    """
    # New DataFrame for output
    df_prime = df.copy()
    # Filter in line-of sight component
    # Local or moving median estimation
    mmr = df_prime[feature].rolling(n,axis=1,min_periods=1).median()
    # Global or line-of-sight median calculation
    gm = df_prime[feature].sub(df[feature].median(axis=1),axis=0)
    # filter in azimuth component
    # Local or moving median estimation
    mmp = df_prime.groupby('scan')[feature].rolling(10,axis=0,min_periods=1).median().reset_index(
            level='scan')[feature]
    # Substraction of V_LOS moving and global median from original DataFrame
    mmp = df_prime[feature].sub(mmp)
    mmr = df_prime[feature].sub(mmr)
    gm = df_prime[feature].sub(gm)  
    # Masking according to threshold values
    mask = []
    mask = (mmr.abs()>lim_m) | (gm.abs()>lim_g) | (mmp.abs()>lim_m)   
    # Output DataFrame masking
    df_prime.ws = df_prime.copy().ws.mask(mask) 
    return df_prime

# In[Clustering Filter with DBSCAN algorithm]
    
def data_filt_DBSCAN(df_ini,features,nn=5,eps=0.3,CNR_n=-24, plot=False):
    """
    Filter based on clustering algorithm Density-Based Spatial Clustering for Applications
    with Noise or DBSAN (Ester M. et al 1996). Each observation is represented as a point in a 
    k-dimensional space, when k features are specified as input. The DBSCAN algorithm sklearn library
    is used here. The minimum number of nearest neigbours, as recommended by (Ester M. et al 1996),
    is kept at 5, and the non-dimensional distance, eps, is calculated as the average between the 
    distance corresponing to the first knee in the k-distance graph, and a "noise level", which will
    depend on the fraction of "reliable" measurements or CNR values greater than the CNR_noise 
    threshold.
    
    Input:
    -----
        df_ini   - Pandas dataframe containing the positional information
                   (range gate and azimuth angle), line-of-sight wind speed, as well as all the
                   features included in the input, apart from "dvdr", "dvdt" and "movmed", which are
                   added to the input DataFrame inside this function, if required.
        
        feature  - List of all features that will be used to characterize observations.
        
        nn       - Minimum number of neighbouring points to define a "core" point. Default is 5.
        
        eps      - Non-dimensional distance within the nn points should lay in a "core" point. 
        
        CNR_n    - CNR_n threshold that defines "reliable" observations
        
        plot     - Boolean. If True, will plot k-diatance plot (log-scale) and the eps final value
                   used by DBSCAN.
        
    Output:
    ------
        mask     - Pandas dataframe with boolean values to later use in masking the input dataframe,
                   df_ini.
    """
    # Additional features
    df = df_ini.copy()
    if 'dvdx' in features:
        df = df_ws_grad(df_ini)
    if 'dvdt' in features:
        df = df_ws_grad(df_ini)
    if 'movmed' in features:
        df = df_mov_med(df,'ws',10) 

    # Initialization of input array of high-dimensional datapoints for DBSCAN
    a,b = df.CNR.values.shape 
    X = np.empty((a*b,len(features)))   
    # Data preparation
    # If input is already filtered
    ind = ~np.isnan(df.ws.values.flatten())
    X = np.empty((sum(ind),len(features)))
    # DBSCAN input filling with data form df
    for i,f in enumerate(features):
        if (f =='elev') | (f =='scan') : # Check elev or azim!!
            X[:,i] = np.array([list(df[f].values),]*b).transpose().flatten()[ind]
        else:
            X[:,i] = df[f].values.flatten()[ind]
    # After input creation, it is re-scaled to non-dimensional space.
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X) 
    # Tree creation for k-distance graph
    tree_X = KDTree(X)
    # first k distance estimation
    d,i = tree_X.query(tree_X.data,k = nn)  
    #k-nearest distance
    d=d[:,-1]
    # log transformation to level-up values and easier identification of "knees"
    # d is an array with k-distances sorted in increasing value.
    d = np.log(np.sort(d))
    # x axis (point label)
    l = np.arange(0,len(d))  
    # Down sampling to speed up calculations
    d_resample = np.array(d[::int(len(d)/400)])
    print(len(d_resample))
    # same with point lables
    l_resample = l[::int(len(d)/400)]
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
#    fig, ax1 = plt.subplots()
#    ax1.plot(l_resample,kappa)
#    plt.grid()
#    ax2 = ax1.twinx()
#    ax2.plot(100*d)
#    fig.tight_layout()
#    
    # location of knee (first point with k-distance above 1 std of k-distance mean)
    #ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
    ind_kappa, _ = find_peaks(kappa,prominence=1) 
    # Just after half of the graph
    ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
    # The first knee...
    l1 = l_resample[ind_kappa][0]
    
#    plt.plot([l1,l1],100*np.array([min(d),max(d)]))
    
    # the corresponding eps distance
    eps0 = np.exp(d[l1])
    # If many knees, noise level is also taken into account
    # Fraction of reliable measurements
    x = np.sum(df.CNR.values>=-24)/len(df.CNR.values.flatten()) 
    # Some limits that can be changed. The distance as function of x, eps(x) is assumed linear.
    # The upper limit x=1 (no noise) specifies eps = 0.53 (this value is obtained just from the data 
    # and is very large). The other bound is x = 0 and eps = 0.075 a very small value.
    c = .075  
    m = .45
    # eps(x) linear relation
    eps = m*x+c   
    eps1 = eps0
    # if reliable data is less than 45% then jusat the noise level criteria eps(x) is used
    #(it is indeed more strict)
    if x<.45:
        eps1=eps
    # DBSCAN object creation, the eps parameter is chosen as the average between the value
    # estimated with the noise-knee criteria, eps0 and the noise-level criteria, eps.
    clf = []
    clf = DBSCAN(eps=(eps1+eps)*.5, min_samples=nn)
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
        plt.figure()
        plt.plot(d)
        plt.plot([0,X.shape[0]],[np.log(eps),np.log(eps)])
        plt.plot([0,X.shape[0]],[np.log(eps0),np.log(eps0)])
        plt.plot([0,X.shape[0]],[np.log(.5*(eps+eps0)),np.log(.5*(eps+eps0))])
        plt.xlabel('Data point', fontsize=12, weight='bold')
        plt.ylabel('k-nearest distance [-]', fontsize=12, weight='bold')
        plt.grid()
        plt.legend(['k-dist.','Noise level','Knee','Average'])
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
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)
        ax.set_xlabel('$V_{LOS} [m/s]$', fontsize=12, weight='bold')
        ax.set_ylabel('Range gate $[m]$', fontsize=12, weight='bold')
        ax.set_zlabel('CNR', fontsize=12, weight='bold')
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                       s=20)
            xy = X[class_member_mask & ~core_samples_mask]
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], marker = 'o', c = tuple(col), edgecolor='k',
                       s=6)  
    # First reliable value and the corresponding cluster to which it belongs
    ind =  (df.CNR.values.flatten()>-24) & (labels != -1)
    if sum(ind)>0:
        lab = sp.stats.mode(labels[ind])[0]
    else:
        lab = np.nan
    # 2-D array
    labels = np.reshape(labels,(a,b))
    # Mask and re index and column names
    mask = pd.DataFrame(labels!=lab)
    mask.columns = ['ws']*b
    mask=mask.set_index(df.ws.index)
    return mask

# In[Interpolation]    
    
def data_interp_kdtree(df_ini,dt,r=[],phi=[],col_r=[],col_phi=[]):
    """
    Interpolation of missing data after filtering. The interpolation uses a kdtree approach and it
    is done both, spatial and temporal. The "distances" in time are estimated as the prodict of dt
    and V_LOS.
    
    Input:
    -----
        df_ini   - Filtered pandas dataframe containing the positional information
                   (range gate and azimuth angle, optional), line-of-sight wind speed.
        
        r        - Numpy array with line-of-sight range gates, if not contained in df.
        
        phi      - Azimuth angles of each laser beam, if not contained in df.
        
        dt       - Time elapsed between sucessive scans. 
        
    Output:
    ------
        df       - Pandas dataframe with missing data, after filtering, filled through interpolation.
        
    """
    df = df_ini.copy()
    # Initialization of the kdtree regressor. The number of neighbours, n_neighbours is set equal to
    # the number of corners and midpoints of a cube.
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30,
                                n_jobs=1)
    
    # If df has positional information
    if (len(r)==0) & (len(phi)==0):
        r = np.unique(df[col_r].values).astype(float)
        phi = np.unique(df[col_phi].values).astype(float)
        
    # Meshgrid in polar coordinates and time.
    rg,phig,tg = np.meshgrid(r,phi,np.array([-dt,0,dt]))
    # Central slice in 3D array
    ind_slice = np.isnan(rg)
    ind_slice[:,:,1] = True
    ws_f = []
    for scan in np.unique(df.index.values):
        print(scan)
        #Initial and final scans are interpolated only in space
        if (scan == 0) | (scan == np.unique(df.index.values)[-1]):
            rg0,phig0 = np.meshgrid(r,phi)
            rp = np.c_[rg0.flatten(),phig0.flatten()]  
            ws = df.ws.loc[scan].values.flatten()
            ind = np.isnan(ws)
            if sum(ind)>0:
                neigh.fit(rp[~ind,:], ws[~ind])
                # Kdtree Regression
                ws[ind] = neigh.predict(rp[ind,:])
            ws_f.append(np.reshape(ws,rg0.shape))
        # The rest of scans are interpolated in space and time        
        else:
            # 3D array 
            ws = np.dstack((df.ws.loc[scan-1].values, df.ws.loc[scan].values,df.ws.loc[scan+1].values))
            # Temporal "distance"
            temp = ws.flatten()*tg.flatten()
            # Invalid points are just put "far" away
            temp[np.isnan(temp)] = tg.flatten()[np.isnan(temp)]*10000
            # Points in space and "time" to interpolate
            rpt = np.c_[rg.flatten()*np.cos(phig.flatten()),rg.flatten()*np.cos(phig.flatten()),temp]
            # Non valid V_LOS spotted
            ind = np.isnan(ws).flatten()
            # Non valid V_LOS in the current scan to be interpolated
            ind_int = (ind) & (ind_slice.flatten())
            # Interpolation is carried out
            if sum(ind_int)>0:
                ws = ws.flatten()
                neigh.fit(rpt[~ind,:], ws[~ind])
                ws[ind_int] = neigh.predict(rpt[ind_int,:])
            ws_f.append(np.reshape(ws,rg.shape)[:,:,1]) 
    # The wind speeds are replaced by the conyinous field...
    df.ws= pd.DataFrame(data = np.vstack(ws_f), index = df.ws.index, columns = df.ws.columns)     
    return (df) 

# In[Kernel density filter-NOT USED]
        
def filt_stat(df_ini,col,P,ngrid,g_by,N,bw=0.0,init=0):  
    """

    
    Input:
    -----
        df_ini     - 
                   

    Output:
    ------
        df_prime - 
    """
    
    df = df_ws_grad(df_ini)
    
    var_g,var_g_s,grid_s,grid_shape = kernel_grid(df,ngrid,col)
    
    #To speed up calculations, the bandwidth is specified beforehand. 
    #This is not mandatory.
    
    bw = 0.0142 
    
    # Definition of scan as index in the Dataframe
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