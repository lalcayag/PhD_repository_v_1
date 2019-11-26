# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize
from os import listdir

sirocco_loc = np.array([6322832.3,0])
vara_loc = np.array([6327082.4,0])
d = vara_loc-sirocco_loc

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(198):

    labels = np.concatenate((labels,vel_lab))
    
sirocco_dir = 'Data/Data_Phase_2/SiroccoWest'
filelist_s = [(filename,getsize(join(sirocco_dir,filename)))
             for filename in listdir(sirocco_dir) if getsize(join(sirocco_dir,filename))>1000000]
size_s = list(list(zip(*filelist_s))[1])
filelist_s = list(list(zip(*filelist_s))[0])

vara_dir = 'Data/Data_Phase_2/VaraWest'
filelist_v = [(filename,getsize(join(vara_dir,filename)))
             for filename in listdir(vara_dir) if getsize(join(vara_dir,filename))>1000000]
size_v = list(list(zip(*filelist_v))[1])
filelist_v = list(list(zip(*filelist_v))[0])

feat = ['ws','range_gate','CNR','azim','dvdr']

# In[Data loading]
#West scanning
n=0
for counter, file_v in enumerate(filelist_v[-2:],0):
    print(counter,file_v)
    if file_v in filelist_s:
        sirocco_w_path = join(sirocco_dir,file_v)
        sirocco_w_df = pd.read_csv(sirocco_w_path,sep=";", header=None) 
        
        vara_w_path =  join(vara_dir,file_v)
        vara_w_df = pd.read_csv(vara_w_path,sep=";", header=None) 

        sirocco_w_df.columns = labels
        vara_w_df.columns = labels
        
        sirocco_w_df['scan'] = sirocco_w_df.groupby('azim').cumcount()
        vara_w_df['scan'] = vara_w_df.groupby('azim').cumcount()
        
        if n == 0:
            
            phi0w = vara_w_df.azim.unique()
            phi1w = sirocco_w_df.azim.unique()
            r0w = np.array(vara_w_df.iloc[(vara_w_df.azim==
                                           min(phi0w)).nonzero()[0][0]].range_gate)
            r1w = np.array(sirocco_w_df.iloc[(sirocco_w_df.azim==
                                           min(phi1w)).nonzero()[0][0]].range_gate)
            
            r_vaw, phi_vaw = np.meshgrid(r0w, np.radians(phi0w)) # meshgrid
            r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid
            
            treew,triw, wvaw, neighvaw, indexvaw, wsiw, neighsiw, indexsiw =  grid_over2(
                                                       (r_vaw, phi_vaw),(r_siw, phi_siw),d)
            print('the tree')
            n = 1
        mask=pd.DataFrame()
        #df_clust = vara_w_df.copy()
        t_step = 3
        ind=np.unique(vara_w_df.scan.values)%t_step==0
        times= np.unique(np.append(np.unique(vara_w_df.scan.values)[ind],
                                       vara_w_df.scan.values[-1]))

        for i in range(len(times)-1):
           print(file_v,size_v[counter],times[i])
           loc = (vara_w_df.scan>=times[i]) & (vara_w_df.scan<times[i+1])
           mask = pd.concat([mask,data_filt_DBSCAN(vara_w_df.loc[loc],feat)])   
           if i == range(len(times)-1):
              loc = vara_w_df.scan == times[i+1]
              mask = pd.concat([mask,data_filt_DBSCAN(vara_w_df.loc[loc],feat)])
    
        #df_clust.ws = df_clust.ws.mask(mask)

        with open('df_mask_clust_v_'+file_v[:14]+'.pkl', 'wb') as clust_200:
             pickle.dump(mask, clust_200)
             
        #df_clust = sirocco_w_df.copy()
        mask=pd.DataFrame()
        ind=np.unique(sirocco_w_df.scan.values)%t_step==0
        times= np.unique(np.append(np.unique(sirocco_w_df.scan.values)[ind],
                                       sirocco_w_df.scan.values[-1]))

        for i in range(len(times)-1):
           print(file_v,size_v[counter],times[i])
           loc = (sirocco_w_df.scan>=times[i]) & (sirocco_w_df.scan<times[i+1])
           mask = pd.concat([mask,data_filt_DBSCAN(sirocco_w_df.loc[loc],feat)])   
           if i == range(len(times)-1):
              loc = sirocco_w_df.scan == times[i+1]
              mask = pd.concat([mask,data_filt_DBSCAN(sirocco_w_df.loc[loc],feat)])

        with open('df_mask_clust_s_'+file_v[:14]+'.pkl', 'wb') as clust_200:
             pickle.dump(mask, clust_200)

# In[]

with open('df_mask_clust_v_'+file_v[:14]+'.pkl', 'rb') as clust_200:
     mask_v = pickle.load(clust_200)

with open('df_mask_clust_s_'+file_v[:14]+'.pkl', 'rb') as clust_200:
     mask_s = pickle.load(clust_200)

mask_CNR_v = (vara_w_df.CNR>-24) & (vara_w_df.CNR<-8)

mask_CNR_v.columns =  mask_v.columns

mask_v.mask(mask_CNR_v,other=False,inplace=True)
             
df_vara = vara_w_df.copy()
df_vara.ws = df_vara.ws.mask(mask_v)

mask_CNR_s = (sirocco_w_df.CNR>-24) & (sirocco_w_df.CNR<-8)

mask_CNR_s.columns =  mask_s.columns

mask_s.mask(mask_CNR_s,other=False,inplace=True)

df_sirocco = sirocco_w_df.copy()
df_sirocco.ws = df_sirocco.ws.mask(mask_s)
             
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(11999,12300):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df_vara.ws.loc[df_vara.scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.001)
               
             

# In[]
             
U = []
V = []
#dphi200_clust0 = []
#dphi200_clust1 = []

for scan_n in range(min(df_sirocco.scan.max(),df_vara.scan.max())):
    print(scan_n)
    tot_s = (8910-mask_s[df_sirocco.scan==scan_n].sum().sum())/8910
    tot_v = (8910-mask_v[df_vara.scan==scan_n].sum().sum())/8910
    if (tot_s>.3) & (tot_v>.3):
        print(scan_n)    
        Lidar_sir = (df_sirocco.ws.loc[df_sirocco.scan==scan_n],phi_siw,wsiw,neighsiw,indexsiw) 
        Lidar_var = (df_vara.ws.loc[df_vara.scan==scan_n],phi_vaw,wvaw,neighvaw,indexvaw)
        auxU, auxV= wind_field_rec(Lidar_var, Lidar_sir, treew, triw, d)
        U.append(auxU) 
        V.append(auxV)
    else:
        U.append([]) 
        V.append([])

with open('U_'+file_v[:14]+'.pkl', 'wb') as U_clust:
    pickle.dump(U, U_clust)
    
with open('V_'+file_v[:14]+'.pkl', 'wb') as V_clust:
    pickle.dump(V, V_clust) 

# In[]
    
Uint, Vint = data_interp_triang(U_c,V_c,triw.x,triw.y,45)

with open('U_int_'+file_v[:14]+'.pkl', 'wb') as U_200_int:
    pickle.dump(Uint, U_200_int)
    
with open('V_int_'+file_v[:14]+'.pkl', 'wb') as V_200_int:
    pickle.dump(Vint, V_200_int)
 
# In[]

with open('U_int_'+file_v[:14]+'.pkl', 'rb') as U_200_int:
   Uint = pickle.load(U_200_int)
    
with open('V_int_'+file_v[:14]+'.pkl', 'rb') as V_200_int:
   Vint = pickle.load(V_200_int)
# In[]
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
        
for i in range(8000,10000):
    #triw.set_mask(masks2[i])
    #tri_r.set_mask(masks_r[i])
    ax.cla()
    ax.set_aspect('equal')
    # Enforce the margins, and enlarge them to give room for the vectors.
    ax.use_sticky_edges = False
    ax.margins(0.07)
    plt.title('Scan num. %i' %i)
    ax.triplot(triw, color='black',lw=.5)
    U_mean = avetriangles(np.c_[triw.x,triw.y], Uint[i], triw.triangles)
    V_mean = avetriangles(np.c_[triw.x,triw.y], Vint[i], triw.triangles)
    
    im=ax.tricontourf(triw,np.sqrt(Uint[i]**2+Vint[i]**2),levels=np.linspace(10,20,300),cmap='jet')
    #im=ax.tricontourf(triw,Uint[i]-U_mean,levels=np.linspace(-10,5,300),cmap='jet')

    
    Q = ax.quiver(3000.00,-830.00,V_mean,U_mean,pivot='middle', scale=75, units='width', color='k')
    circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
    ax.add_artist(circle)
    ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
    ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 
    
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
    plt.pause(.5)

# In[]
Su_u = np.zeros((15050,256))
Sv_v = np.zeros((15050,256))
Su_v = np.zeros((15050,256))
k_1 = np.zeros((15050,256))
k_2 = np.zeros((15050,256))

for scan in range(0,15050):
    print(scan)
    if len(Uint[scan])>0:
        Su,Sv,Suv,k1,k2=spatial_autocorr_fft(triw,Vint[scan],Uint[scan],transform = True,N_grid=256,interp='cubic')
        Su_u[scan,:] = sp.integrate.simps(Su,k2,axis=1)
        Sv_v[scan,:] = sp.integrate.simps(Sv,k2,axis=1)
        Su_v[scan,:] = sp.integrate.simps(Suv,k2,axis=1)
        k_1[scan,:] = k1
        k_1[scan,:] = k2


for scan in range(0,15050): 
    if len(Uint[scan])>0:       
        k1,k2 = wavenumber(triw,Vint[scan],Uint[scan],N_grid=256) 
        k_1[scan,:] = k1
        k_1[scan,:] = k2
        C = 1/(4*max(k1)*max(k2))
        print(scan,C**2)
        Su_v[scan,:] = Su_v[scan,:]*C**2 
        
with open('S_minE_k_1_2.pkl', 'wb') as V_t:
     pickle.dump((Su_u,Sv_v,Su_v,k_1,k_2),V_t)    

fig1, ax1 = plt.subplots()
ax1.set_xscale('log')
ax1.set_yscale('log')

for i in range(0,len(Su_u)):
    ax1.scatter(S1[3],S1[3]*Su_u[i,:],color='black')
    
# In[]

X = RobustScaler(quantile_range=(25, 75)).fit_transform(Su_u)
tree_X = KDTree(X)
# first k distance estimation
d,i = tree_X.query(tree_X.data,k = 5)  
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
# location of knee (first point with k-distance above 1 std of k-distance mean)
#ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
ind_kappa, _ = find_peaks(kappa,prominence=1) 
# Just after half of the graph
ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
# The first knee...
l1 = l_resample[ind_kappa][0]
# the corresponding eps distance
eps0 = np.exp(d[l1])   
plt.plot(l_resample,d_resample) 

clf = DBSCAN(eps=eps0, min_samples=5)
clf.fit(X)
# Cluster-core samples and boundary samples identification
core_samples_mask = np.zeros_like(clf.labels_, dtype=bool)
core_samples_mask[clf.core_sample_indices_] = True
# Array with labels identifying the different clusters in the data
labels = clf.labels_
# Number of identified clusters (excluding noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 