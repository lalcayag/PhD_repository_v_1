# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:19:58 2018

@author: lalc
"""
# In[]

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
with open('U200_clust_int.pkl', 'rb') as U_200_clust:
    U200_clust_int1 = pickle.load(U_200_clust)
    
with open('V200_clust_int.pkl', 'rb') as V_200_clust:
    V200_clust_int1 = pickle.load(V_200_clust)  
    
with open('U200_clust.pkl', 'rb') as U_200_clust:
    U200_clust1 = pickle.load(U_200_clust)
    
with open('V200_clust.pkl', 'rb') as V_200_clust:
    V200_clust1 = pickle.load(V_200_clust)  
     
#with open('rho_cub_650-758.pkl', 'rb') as U_200_med:
#    rho_cub.append(pickle.load(U_200_med))
# In[]

chunks = np.linspace(0,1409,1409/100).astype(int)
rho_cub2 = []
for i in range(len(chunks)-1):
    print(i)
    with open('rho_cub_'+str(chunks[i])+'-'+str(chunks[i+1])+'.pkl', 'rb') as V_t:
        rho_cub2.append(pickle.load(V_t))
rho_cub2 = [val for sublist in rho_cub2 for val in sublist]        
rho_cub2[:] = [item for item in rho_cub2 if item != []]
# In[already saved]
rho_cub = []
for scan in range(300,501):
    print(scan)
    if len(U200_clust_int1[scan])>0:
        rho_cub.append(spatial_autocorr_fft(triw,U200_clust_int1[scan],V200_clust_int1[scan],transform = True,interp='cubic'))
    else:
        rho_cub.append([])
       
rho_cub[:] = [item for item in rho_cub if item != []]
# In[Many scans]
rho_lin = []
for scan in range(300,501):
    print(scan)
    if len(U200_clust_int1[scan])>0:
        rho_lin.append(spatial_autocorr_fft(triw,U200_clust_int1[scan],V200_clust_int1[scan],transform = True))
    else:
        rho_lin.append([])      
rho_lin[:] = [item for item in rho_lin if item != []]

    
         
# In[]
    

chunk_lin = rho_lin[:100]
chunk_cub = rho_cub
chunk_cub2 = rho_cub2[300:400]

S_u_cub = np.array(list(zip(*chunk_cub ))[0])
S_v_cub = np.array(list(zip(*chunk_cub ))[1])
S_uv_cub = np.array(list(zip(*chunk_cub ))[2])

S_u_cub2 = np.array(list(zip(*chunk_cub2 ))[0])
S_v_cub2 = np.array(list(zip(*chunk_cub2 ))[1])
S_uv_cub2 = np.array(list(zip(*chunk_cub2 ))[2])

S_u_lin = np.array(list(zip(*chunk_lin ))[0])
S_v_lin = np.array(list(zip(*chunk_lin ))[1])
S_uv_lin = np.array(list(zip(*chunk_lin ))[2])

#
#fig1, ax1 = plt.subplots()
#im=ax1.contourf(np.mean(S_u_chunk,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')

bins=20
Su_lin = spectra_average(np.mean(S_u_lin,axis=0),(np.max(rho_lin[0][3]),np.max(rho_lin[0][4])),bins,angle_bin = 5)
Sv_lin = spectra_average(np.mean(S_v_lin,axis=0),(np.max(rho_lin[0][3]),np.max(rho_lin[0][4])),bins,angle_bin = 5)
Suv_lin = spectra_average(np.mean(S_uv_lin,axis=0),(np.max(rho_lin[0][3]),np.max(rho_lin[0][4])),bins,angle_bin = 5)

Su_cub = spectra_average(np.mean(S_u_cub,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
Sv_cub = spectra_average(np.mean(S_v_cub,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
Suv_cub = spectra_average(np.mean(S_uv_cub,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)

Su_cub2 = spectra_average(np.mean(S_u_cub2,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)
Sv_cub2 = spectra_average(np.mean(S_v_cub2,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)
Suv_cub2 = spectra_average(np.mean(S_uv_cub2,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)


fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber, k')
ax.set_ylabel('kS(k)')

ax.plot(Su_lin.k,Su_lin.k*Su_lin.S)
ax.plot(Su_cub2.k,Su_cub2.k*Su_cub2.S)
ax.plot(Su_cub.k,Su_cub.k*Su_cub.S)
fig.legend(['$S_{uu}$ linear','$S_{uu}$ cubic', '$S_{uu}$ cubic q-c'],bbox_to_anchor=(.9, .89)) 



fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber, k')
ax.set_ylabel('kS(k)')

ax.plot(Sv_lin.k,Sv_lin.k*Sv_lin.S)
ax.plot(Sv_cub2.k,Sv_cub2.k*Sv_cub2.S)
ax.plot(Sv_cub.k,Sv_cub.k*Sv_cub.S)
fig.legend(['$S_{vv}$ linear','$S_{vv}$ cubic', '$S_{vv}$ cubic q-c'],bbox_to_anchor=(.9, .89)) 


fig, ax = plt.subplots()
im=ax.contourf(rho_cub[0][3],rho_cub[0][4],np.mean(S_v_lin,axis=0),3000,locator=ticker.LogLocator(),cmap='rainbow')
ax.set_xlim(-.005,.005)
ax.set_ylim(-.005,.005)

# In[Fit, slopes]
import scipy.optimize as optimization

def spec_slop(k,a,b):
    return a*(k**b)

limits = [10**-4,3*10**-3,10**-2,4*10**-2]

params = np.zeros((3,2))

for i in range(len(limits)-1):
    print(i)
    #sigma = np.zeros(Su.k.shape)
    #sigma[(Su.k>limits[i])&(Su.k<limits[i+1])] = 1.0
    ind = [(Su.k>limits[i])&(Su.k<limits[i+1])]
    #params[i,:] = optimization.curve_fit(spec_slop, np.log(Su.k)[ind], np.log(Su.k*Su.S)[ind], [-10,-2/3])[0]
    params[i,:] = optimization.curve_fit(spec_slop, Su.k[ind], Su.S[ind], [-10,-2/3])[0]
    ax.plot(Su.k,Su.k*spec_slop(Su.k,params[i,0],params[i,1]))
# In[]
scan_n = 500

fig, ax = plt.subplots()
fig, ax = plt.subplots()
im=ax.contourf(rho2[scan_n][3],rho2[scan_n][4],rho2[scan_n][0],300,locator=ticker.LogLocator(),cmap='rainbow')
fig, ax = plt.subplots()
im=ax.contourf(rho2[scan_n][3],rho2[scan_n][4],rho2[scan_n][1],300,locator=ticker.LogLocator(),cmap='rainbow')
fig, ax = plt.subplots()
im=ax.contourf(rho2[scan_n][3],rho2[scan_n][4],rho2[scan_n][2],300,locator=ticker.LogLocator(),cmap='rainbow')


bins=20
S_r_u = spectra_average(rho2[scan_n][0],(np.max(rho2[scan_n][3]),np.max(rho2[scan_n][4])),bins,angle_bin = 5)
S_r_v = spectra_average(rho2[scan_n][1],(np.max(rho2[scan_n][3]),np.max(rho2[scan_n][4])),bins,angle_bin = 5)
S_r_uv = spectra_average(rho2[scan_n][2],(np.max(rho2[scan_n][3]),np.max(rho2[scan_n][4])),bins,angle_bin = 5)
fig, ax = plt.subplots()
ax.plot(S_r_u.k,S_r_u.k*S_r_u.S)
ax.plot(S_r_v.k,S_r_v.k*S_r_v.S)
ax.plot(S_r_uv.k,S_r_uv.k*S_r_uv.S)
ax.set_xscale('log')
ax.set_yscale('log')

plt.figure()
plt.polar()
plt.plot(S_r_u.phi*np.pi/180,np.log(S_r_u.S_p),marker='o')
plt.plot(S_r_v.phi*np.pi/180,np.log(S_r_v.S_p),marker='o')
plt.plot(S_r_uv.phi*np.pi/180,np.log(S_r_uv.S_p),marker='o')

# In[Many scans]
rho2 = []
for scan in range(len(U200_clust_int1)):
    print(scan)
    if len(Ur[scan])>0:
        rho2.append(spatial_autocorr_fft(triw,U200_clust_int1[scan],V200_clust_int1[scan],transform = True))
    else:
        rho2.append([])
        

# In[]
S_r_u2 = []
S_r_v2 = []
S_r_uv2 = []

bins=20

for scan in range(len(Ur)):
    print(scan)
    if len(rho2[scan][0])>0:
        S_r_u.append(spectra_average(rho2[scan][0],(np.max(rho2[scan][3]),np.max(rho2[scan][4])),
                                     bins,angle_bin = 5))
        S_r_v.append(spectra_average(rho2[scan][1],(np.max(rho2[scan][3]),np.max(rho2[scan][4])),
                                     bins,angle_bin = 5))
        S_r_uv.append(spectra_average(rho2[scan][2],(np.max(rho2[scan][3]),np.max(rho2[scan][4])),
                                      bins,angle_bin = 5))    
    else:
        S_r_u2.append([])
        S_r_v2.append([])
        S_r_uv2.append([])

# In[]
    
scans = [300,700]

chunk = rho2[500:700]

S_u_chunk2 = np.array(list(zip(*chunk ))[0])
S_v_chunk2 = np.array(list(zip(*chunk ))[1])
S_uv_chunk2 = np.array(list(zip(*chunk ))[2])

#


bins=20
Su2 = spectra_average(np.mean(S_u_chunk2,axis=0),(np.max(rho2[0][3]),np.max(rho2[0][4])),bins,angle_bin = 5)
Sv2 = spectra_average(np.mean(S_v_chunk2,axis=0),(np.max(rho2[0][3]),np.max(rho2[0][4])),bins,angle_bin = 5)
Suv2 = spectra_average(np.mean(S_uv_chunk2,axis=0),(np.max(rho2[0][3]),np.max(rho2[0][4])),bins,angle_bin = 5)
fig, ax = plt.subplots()
ax.plot(Su2.k,Su2.k*Su2.S)
ax.plot(Sv2.k,Sv2.k*Sv2.S)
ax.plot(Suv2.k,Suv2.k*Suv2.S)
#ax.plot(Su.k,(10**-1/np.max(Su.k**(-2/3)))*Su.k**(-2/3))
#ax.plot(Su.k,(10/np.max(Su.k**(-3)))*Su.k**(-3))
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber')
ax.set_ylabel('kS(k)')
fig.legend([r'S_{uu}',r'S_{vv}',r'S_{uv}']) 
#S_u_chunk = np.array(list([0]))




# In[Fit, slopes]
import scipy.optimize as optimization

def spec_slop(k,a,b): 
    return a*(k**b)

limits = [4*10**-3,10**-2,2*10**-2,5*10**-2]

params2 = np.zeros((3,2))

for i in range(len(limits)-1):
    print(i)
    #sigma = np.zeros(Su.k.shape)
    #sigma[(Su.k>limits[i])&(Su.k<limits[i+1])] = 1.0
    ind = [(Su3.k>limits[i])&(Su3.k<limits[i+1])]
    #params[i,:] = optimization.curve_fit(spec_slop, np.log(Su.k)[ind], np.log(Su.k*Su.S)[ind], [-10,-2/3])[0]
    params2[i,:] = optimization.curve_fit(spec_slop, Su3.k[ind], Su3.S[ind], [-10,-2/3])[0]
    ax.plot(Su3.k,Su3.k*spec_slop(Su3.k,params2[i,0],params2[i,1]))

ax.set_ylim(10**-6,10**-1)
# In[Many scans]
rho_cub = []
for scan in range(703,1410):
    print(scan)
    if len(Ur[scan])>0:
        rho_cub.append(spatial_autocorr_fft(triw,U200_clust_int1[scan],V200_clust_int1[scan],transform = True,interp='cubic'))
    else:
        rho_cub.append([])
        
S_u_chunk3 = np.array(list(zip(*rho_cub[:200] ))[0])
S_v_chunk3 = np.array(list(zip(*rho_cub[0:100] ))[1])
S_uv_chunk3 = np.array(list(zip(*rho_cub[0:100] ))[2])
        
bins=20
Su3 = spectra_average(np.mean(S_u_chunk3,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
Sv3 = spectra_average(np.mean(S_v_chunk3,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
Suv3 = spectra_average(np.mean(S_uv_chunk3,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
Suu_vv3 = spectra_average(.5*(np.mean(S_u_chunk3,axis=0)+np.mean(S_v_chunk3,axis=0)),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5) 
fig, ax = plt.subplots()
ax.plot(Su3.k,Su3.k*Su3.S)
ax.plot(Sv3.k,Sv3.k*Sv3.S)
ax.plot(Suv3.k,Suv3.k*Suv3.S)
ax.plot(Suu_vv3.k,Suu_vv3.k*Suu_vv3.S)

ax.plot(Su.k,(10**-1/np.max(Su.k**(-2/3)))*Su.k**(-2/3))
ax.plot(Su.k,Su.k**(-1)/10000)
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber')
ax.set_ylabel('kS(k)')
fig.legend([r'$S_{uu}$',r'$S_{vv}$',r'$S_{uv}$']) 
ax.set_ylim(-10**-2,1)
ax.set_xlim(5*10**-4,5*10**-2)


fig1, ax1 = plt.subplots()
im=ax1.contourf(rho_cub[0][3],rho_cub[0][4],np.mean(S_u_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')
ax1.set_xlim(-.005,.005)
ax1.set_ylim(-.005,.005)
fig1, ax1 = plt.subplots()
im=ax1.contourf(rho3[0][3],rho3[0][4],np.mean(S_v_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')

fig1, ax1 = plt.subplots()
im=ax1.contourf(rho3[0][3],rho3[0][4],-np.mean(S_uv_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')

   
chunks = np.linspace(0,len(rho_cub),len(rho_cub)/5).astype(int)
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber')
ax.set_ylabel('kS(k)')
ax.set_ylim(10**-5,10**0)
for i in range(len(chunks)-1):
    ax.cla()
    S_u_chunki = np.array(list(zip(*rho_cub[chunks[i]:chunks[i+1]]))[0])
    #S_v_chunki = np.array(list(zip(*rho_cub[chunks[i]:chunks[i+1]]))[1])
    #S_uv_chunki = np.array(list(zip(*rho_cub[chunks[i]:chunks[i+1]]))[2])
    
    Sui = spectra_average(np.mean(S_u_chunki,axis=0),(np.max(rho_cub[0][3]),np.max(rho_cub[0][4])),bins,angle_bin = 5)
    #Svi = spectra_average(np.mean(S_v_chunk3,axis=0),(np.max(rho3[0][3]),np.max(rho3[0][4])),bins,angle_bin = 5)
    #Suvi = spectra_average(np.mean(S_uv_chunk3,axis=0),(np.max(rho3[0][3]),np.max(rho3[0][4])),bins,angle_bin = 5)
    ax.plot(Sui.k,Sui.k*Sui.S)
    ax.set_xscale('log')
    ax.set_yscale('log')    
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('kS(k)')
    ax.set_ylim(10**-5,10**0)
    plt.pause(.5)
    
# In[]

rho_cub2 = []
for scan in range(300,700):
    print(scan)
    if len(Ur[scan])>0:
        rho_cub2.append(spatial_autocorr_fft(triw,U200_clust1[scan],V200_clust1[scan],transform = True,interp='cubic'))
    else:
        rho_cub2.append([])


rho_cub2[:] = [item for item in rho_cub2 if item != []]
        
S_u_chunk4 = np.array(list(zip(*rho_cub2[0:200]))[0])
S_v_chunk4 = np.array(list(zip(*rho_cub2[0:200] ))[1])
S_uv_chunk4 = np.array(list(zip(*rho_cub2[0:200] ))[2])
        
bins=20
Su3 = spectra_average(np.mean(S_u_chunk3,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)
Sv3 = spectra_average(np.mean(S_v_chunk3,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)
Suv3 = spectra_average(np.mean(S_uv_chunk3,axis=0),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5)
Suu_vv3 = spectra_average(.5*(np.mean(S_u_chunk3,axis=0)+np.mean(S_v_chunk3,axis=0)),(np.max(rho_cub2[0][3]),np.max(rho_cub2[0][4])),bins,angle_bin = 5) 
fig, ax = plt.subplots()
ax.plot(Su3.k,Su3.k*Su3.S)
ax.plot(Sv3.k,Sv3.k*Sv3.S)
ax.plot(Suv3.k,Suv3.k*Suv3.S)
ax.plot(Suu_vv3.k,Suu_vv3.k*Suu_vv3.S)

ax.plot(Su.k,(10**-1/np.max(Su.k**(-2/3)))*Su.k**(-2/3))
ax.plot(Su.k,Su.k**(-1)/10000)
ax.set_xscale('log')
ax.set_yscale('log')    
ax.set_xlabel('Wavenumber')
ax.set_ylabel('kS(k)')
fig.legend([r'$S_{uu}$',r'$S_{vv}$',r'$S_{uv}$']) 
ax.set_ylim(-10**-2,2*10**-2)
ax.set_xlim(5*10**-4,5*10**-2)

fig1, ax1 = plt.subplots()
im=ax1.contourf(rho3[0][3],rho3[0][4],np.mean(S_u_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')
ax1.set_xlim(-.005,.005)
ax1.set_ylim(-.005,.005)
fig1, ax1 = plt.subplots()
im=ax1.contourf(rho3[0][3],rho3[0][4],np.mean(S_v_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')

fig1, ax1 = plt.subplots()
im=ax1.contourf(rho3[0][3],rho3[0][4],-np.mean(S_uv_chunk3,axis=0),300,locator=ticker.LogLocator(),cmap='rainbow')

# In[]
