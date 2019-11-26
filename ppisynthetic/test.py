# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:39:14 2019

@author: lalc
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

r = np.linspace(-200,200,1000)
dp = 35
dl = 75 
rp = dl/(2*np.sqrt(np.log(2)))

error = sp.special.erf((r+.5*dp)/rp)-sp.special.erf((r-.5*dp)/rp)

w = (1/2/dp)*error

plt.plot(r,w)

dp = 35
dl = 75 
r = np.linspace(105,7000,198)
n=21
r_refine = np.linspace(r.min()-2*dp,r.max()+2*dp,len(r)*(n-1)+1)  
aux_1 = np.reshape(np.repeat(r,len(r_refine),axis = 0),(len(r),len(r_refine)))
aux_2 = np.reshape(np.repeat(r_refine,len(r)),(len(r_refine),len(r))).T
r_F = aux_1-aux_2
erf = sp.special.erf((r_F+.5*dp)/rp)-sp.special.erf((r_F-.5*dp)/rp)
w = (1/2/dp)*erf
plt.plot(r_refine,w[1,:])




r_unique = np.linspace(105,7000,198)
n=21
delta_r = 35
r_refine = np.linspace(r_unique.min()-delta_r/2,r_unique.max()+
                           delta_r/2,len(r_unique)*(n-1)+1) 
h= 2*(np.r_[np.repeat(r_unique,(n-1)),r_unique[-1]]-r_refine)/delta_r 

plt.plot(r_refine,h)
    
w = .75*(1-h**2)

plt.plot(r_refine,w)
 
w = np.reshape(np.repeat(w,phi_t_refine.shape[0]),(phi_t_refine.T).shape).T
norm = np.sum(w[0,:(n-1)])
w = w/norm




#####################################


# In[]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim Output dir')
root.destroy()


cwd = os.getcwd()
os.chdir(file_in_path)


onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)

i = 200
ae_i = ae.flatten()[i]
L_i = L.flatten()[i]
G_i = G.flatten()[i]
seed_i = seed.flatten()[i]
N_x = 2048
N_y = 2048
u_file_name = 'simu'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
v_file_name = 'simv'+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
u = np.reshape(np.fromfile(u_file_name, dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile(v_file_name, dtype=np.float32),(N_x,N_y)).T

plt.figure()
plt.contourf(u, cmap = 'jet')
plt.figure()
plt.contourf(v, cmap = 'jet')
plt.figure()
plt.contourf((v**2+u**2)**.5, cmap = 'jet')

r_s = np.linspace(150,7000,198)
r_v = np.linspace(150,7000,198)
phi_s = np.linspace(256,344,45)*np.pi/180
phi_v = np.linspace(196,284,45)*np.pi/180
r_s_g, phi_s_g = np.meshgrid(r_s,phi_s)
r_v_g, phi_v_g = np.meshgrid(r_v,phi_v)  
siro = np.array([6322832.3,0])
vara = np.array([6327082.4,0])
d = vara-siro
r_s_t,phi_s_t = wr.translationpolargrid((r_s_g, np.pi-phi_s_g),-d/2)
r_v_t,phi_v_t = wr.translationpolargrid((r_v_g, np.pi-phi_v_g),d/2)

x_max = np.max(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])
x_min = np.min(np.r_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_v_t*np.cos(phi_v_t)).flatten()])
y_max = np.max(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])
y_min = np.min(np.r_[(r_s_t*np.sin(phi_s_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()])
L_x = x_max-x_min
L_y = y_max-y_min
N_x = 2048
N_y = 2048
x = np.linspace(x_min,x_max,N_x)
y = np.linspace(y_min,y_max,N_y)
grid = np.meshgrid(x,y)
xy = np.c_[grid[0].flatten(),grid[1].flatten()]

tri_try = Delaunay(np.c_[grid[0].flatten(),grid[1].flatten()], qhull_options = "QJ")


r_s = np.linspace(105,7000,198)
r_v = np.linspace(105,7000,198)
phi_s = np.linspace(256,344,45)*np.pi/180
phi_v = np.linspace(196,284,45)*np.pi/180
r_s_g, phi_s_g = np.meshgrid(r_s,phi_s)
r_v_g, phi_v_g = np.meshgrid(r_v,phi_v)  
siro = np.array([6322832.3,0])
vara = np.array([6327082.4,0])
d = vara-siro
r_s_t,phi_s_t = wr.translationpolargrid((r_s_g, np.pi-phi_s_g),-d/2)
r_v_t,phi_v_t = wr.translationpolargrid((r_v_g, np.pi-phi_v_g),d/2)

Dir = np.linspace(90,270,5)*np.pi/180

dir_mean = Dir[4]

u_mean = 15
U_in = u_mean + u
V_in = 0 + v

vtx0, wts0, w0, c_ref0, s_ref0, shapes = early_weights_pulsed(r_s_g,np.pi-phi_s_g, dl, dir_mean , tri_try, -d/2, y[0]/2)
vtx1, wts1, w1, c_ref1, s_ref1, shapes = early_weights_pulsed(r_v_g,np.pi-phi_v_g, dl, dir_mean , tri_try, d/2, y[0]/2)

vlos0 = num_pulsed_lidar(U_in,V_in,vtx0,wts0,w0,c_ref0, s_ref0, shapes)
vlos1 = num_pulsed_lidar(U_in,V_in,vtx1,wts1,w1,c_ref1, s_ref1, shapes)


tree,tri, wva, neighva, indexva, wsi, neighsi, indexsi = wr.grid_over2((r_v_g, np.pi-phi_v_g),(r_s_g, np.pi-phi_s_g),-d)


r_min=np.min(np.sqrt(tri.x**2+tri.y**2))
d_grid = r_min*2*np.pi/180
n_next = int(2**np.ceil(np.log(L_y/d_grid+1)/np.log(2))) 

x_new = np.linspace(x.min(),x.max(),n_next)
y_new = np.linspace(y.min(),y.max(),n_next)
grid_new = np.meshgrid(x_new,y_new)

vlos1_int_sq = sp.interpolate.griddata(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()],
                                             vlos1.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
vlos0_int_sq = sp.interpolate.griddata(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()],
                                             vlos0.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')

vlos1_int_sq = np.reshape(vlos1_int_sq,grid_new[0].shape)
vlos0_int_sq = np.reshape(vlos0_int_sq,grid_new[0].shape)

r_tri_s = np.sqrt(grid_new[0]**2 + grid_new[1]**2)
phi_tri_s = np.arctan2(grid_new[1],grid_new[0])
r_tri_v_s, phi_tri_v_s = wr.translationpolargrid((r_tri_s, phi_tri_s),-d/2)
r_tri_s_s, phi_tri_s_s = wr.translationpolargrid((r_tri_s, phi_tri_s),d/2)

U,V = dir_rec_rapid(vlos1_int_sq.flatten(),vlos0_int_sq.flatten(), phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),grid_new[0].shape)

vtx02, wts02, w02, c_ref02, s_ref02, shapes = early_weights2(r_s_g,np.pi-phi_s_g, dir_mean , tri_try, -d/2, y[0]/2)
vtx12, wts12, w12, c_ref12, s_ref12, shapes = early_weights2(r_v_g,np.pi-phi_v_g, dir_mean , tri_try, d/2, y[0]/2)

vlos02 = num_lidar_rot_del(U_in.flatten(),V_in.flatten(), vtx02, wts02, w02, c_ref02.flatten(), s_ref02.flatten(), shapes)
vlos12 = num_lidar_rot_del(U_in.flatten(),V_in.flatten(), vtx12, wts12, w12, c_ref12.flatten(), s_ref12.flatten(), shapes)

vlos12_int_sq = sp.interpolate.griddata(np.c_[(r_v_t*np.cos(phi_v_t)).flatten(),(r_v_t*np.sin(phi_v_t)).flatten()],
                                             vlos12.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
vlos02_int_sq = sp.interpolate.griddata(np.c_[(r_s_t*np.cos(phi_s_t)).flatten(),(r_s_t*np.sin(phi_s_t)).flatten()],
                                             vlos02.flatten(), (grid_new[0].flatten(), grid_new[1].flatten()), method='cubic')
vlos12_int_sq = np.reshape(vlos12_int_sq,grid_new[0].shape)
vlos02_int_sq = np.reshape(vlos02_int_sq,grid_new[1].shape)

U2,V2 = dir_rec_rapid(vlos12_int_sq.flatten(),vlos02_int_sq.flatten(),phi_tri_v_s.flatten(),phi_tri_s_s.flatten(),grid_new[0].shape)

##############

fftU2 = np.fft.fft2(r_u_sq8)
fftU2 = np.fft.fftshift(fftU2)
k1 = np.fft.fftshift((np.fft.fftfreq(fftU2.shape[1], d=tau2[256,257])))
k2 = np.fft.fftshift((np.fft.fftfreq(fftU2.shape[0], d=eta2[257,256])))


##########################


    ####################

import time

t0 = time.process_time()
tau0,eta0,r_u0,r_v0,r_uv0 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int, shrink = True, tri_calc = False, calc = 'mean')
elapsed_time0 = time.process_time() - t0

t1 = time.process_time()
tau1,eta1,r_u1,r_v1,r_uv1 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int, shrink = True, tri_calc = False, calc = 'mean',refine=64)
elapsed_time1 = time.process_time() - t1

t2 = time.process_time()
tau2,eta2,r_u2,r_v2,r_uv2 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int, shrink = True, tri_calc = False, calc = 'mean',refine=48)
elapsed_time2 = time.process_time() - t2

Uaux = U.copy()
Uaux = Uaux-np.nanmean(Uaux.flatten())
#Uaux[np.isnan(Uaux)] = 0.0
t0 = time.process_time()
r0 = np.correlate(Uaux[~np.isnan(Uaux)].flatten(),Uaux[~np.isnan(Uaux)].flatten())/(Uaux[~np.isnan(Uaux)].flatten().shape[0])
elapsed_time0 = time.process_time() - t0

t1 = time.process_time()
r1 = np.mean(Uaux[~np.isnan(Uaux)].flatten()*Uaux[~np.isnan(Uaux)].flatten())
elapsed_time1 = time.process_time() - t1


# In[]
auto = []
spec = []
spec_1D = []
refine = [32,64,96,128]
t=[]
for i,ref in enumerate(refine):
    t0 = time.process_time()
    auto.append(sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int, shrink = True, tri_calc = False, calc = 'mean',refine=ref))
    t.append(time.process_time() - t0)
    print(i,t[i])
    spec.append(sc.spectra_fft((auto[i][0],auto[i][1]),auto[i][2],auto[i][3],auto[i][4],K=2))
    Suu = sp.integrate.simps(np.abs(spec[i][2]),spec[i][1],axis=0)[spec[i][0]>0]
    Svv = sp.integrate.simps(np.abs(spec[i][3]),spec[i][1],axis=0)[spec[i][0]>0]
    Suv = sp.integrate.simps(np.abs(spec[i][4]),spec[i][1],axis=0)[spec[i][0]>0]
    spec_1D.append((spec[i][0][spec[i][0]>0],Suu,Svv,Suv))

for i,ref in enumerate(refine):
    k_int_grd=np.meshgrid(spec[i][0],spec[i][1])
    sc.plot_log2D(k_int_grd, np.abs(spec[i][2]), label_S = "$\log_{10}{S}$", C = 10**-4)
    sc.plot_log2D(k_int_grd, np.abs(spec[i][3]), label_S = "$\log_{10}{S}$", C = 10**-4)

plt.figure()
plt.xscale('log')
plt.yscale('log')
for i,ref in enumerate(refine):
    plt.plot(spec_1D[i][0],spec_1D[i][0]*spec_1D[i][1])
    
# In[]
N_test=500
refine = 2**np.array([3,4,5,6,7])
n_tau_next = int(2**np.ceil(np.log(N_test)/np.log(2))) 
n_tau = []
plt.figure()
for ref in refine:    
    n_tau.append(np.exp((np.logspace(0,np.log(n_tau_next)/np.log(10),ref))).astype(int))
    plt.plot()



tau0,eta0,r_u0,r_v0,r_uv0 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int,
                       shrink = True, tri_calc = False, calc = 'mean',refine=32)

tau1,eta1,r_u1,r_v1,r_uv1 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int,
                       shrink = True, tri_calc = False, calc = 'mean',refine=32)

tau2,eta2,r_u2,r_v2,r_uv2 = sc.spatial_autocorr_sq(grid_new,U,V,tri_del = tri_del, mask_int = mask_int,
                       shrink = True, tri_calc = False, calc = 'mean',refine=32)


n_tau=np.arange(len(tau0[0,:]))-int(len(tau0[0,:])/2-1)
n_eta=np.arange(len(eta0[:,0]))-int(len(eta0[:,0])/2-1)

n_tau[n_tau<=0]= n_tau[n_tau<=0]-1
n_eta[n_eta<=0]= n_eta[n_eta<=0]-1


#number of valid samples

U = np.reshape(U,grid_new[0].shape)
V = np.reshape(V,grid_new[0].shape)

def valid_samples(n_tau,n_eta,U):
    
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = U[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = U[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = U[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = U[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = U[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = U[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = U[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = U[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = U
    ind = ~(np.isnan(U.flatten()) | np.isnan(U_del.flatten()))
    return np.sum(ind)

def autocorr(n_tau,n_eta,U):
    
    if (n_tau!=0) & (n_eta!=0):
        if (n_tau>0) & (n_eta>0): #ok
            U_del = U[n_eta:,:-n_tau]
            U = U[:-n_eta,n_tau:]                        
        if (n_tau<0) & (n_eta>0): #ok
            U_del = U[n_eta:,-n_tau:]
            U = U[:-n_eta,:n_tau]                    
        if (n_tau>0) & (n_eta<0): #ok
            U_del = U[:n_eta,:-n_tau]
            U = U[-n_eta:,n_tau:]
        if (n_tau<0) & (n_eta<0): #ok
            U_del = U[:n_eta,-n_tau:] 
            U = U[-n_eta:,:n_tau]                       
    if (n_tau==0) & (n_eta!=0):
        if n_eta>0:
            U_del = U[n_eta:,:]
            U = U[:-n_eta,:]                    
        if n_eta<0:
            U_del = U[:n_eta,:] 
            U = U[-n_eta:,:]            
    if (n_tau!=0) & (n_eta==0):
        if n_tau>0:
            U_del = U[:,:-n_tau]
            U = U[:,n_tau:]             
        if n_tau<0:
            U_del = U[:,-n_tau:] 
            U = U[:,:n_tau]           
    if (n_tau==0) & (n_eta==0):
        U_del = U
    ind = ~(np.isnan(U.flatten()) | np.isnan(U_del.flatten()))
    
    return np.sum(U.flatten()[ind]*U_del.flatten()[ind])
    



def shrink(grid,U,V):
        patch = ~np.isnan(U)
        ind_patch_x = np.sum(patch,axis=1) != 0
        ind_patch_y = np.sum(patch,axis=0) != 0
        if np.sum(ind_patch_x) > np.sum(ind_patch_y):
            ind_patch_y = ind_patch_x
        elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
            ind_patch_x = ind_patch_y        
        n = np.sum(ind_patch_x)
        m = np.sum(ind_patch_y)          
        ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
        ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
        U = np.reshape(U[ind_patch_grd],(n,m))
        V = np.reshape(V[ind_patch_grd],(n,m)) 
        grid_x = np.reshape(grid[0][ind_patch_grd],(n,m))
        grid_y = np.reshape(grid[1][ind_patch_grd],(n,m))
        return ((grid_x,grid_y),U,V)

U_file_name = 'U'+str(u_mean)+str(int(Dir[1]*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
V_file_name = 'V'+str(u_mean)+str(int(Dir[1]*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)

U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)        
U, V, mask, mask_int, tri_del = sc.field_rot(x, y, U, V, tri_calc = True)
U[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, U[mask])(np.c_[grid_new[0].flatten(),grid_new[1].flatten()][mask_int])
U[~mask_int] = np.nan
V[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, V[mask])(np.c_[grid_new[0].flatten(),grid_new[1].flatten()][mask_int])
V[~mask_int] = np.nan      

U = np.reshape(U,grid_new[0].shape)
V = np.reshape(V,grid_new[0].shape)
su = np.nanvar(U.flatten())
U_mean = np.nanmean(U.flatten())
U = U-np.nanmean(U.flatten())
V = V-np.nanmean(V.flatten())

grd_shr,U,V = shrink(grid_new,U,V)

n, m = grd_shr[0].shape
n_tau_max = n
n_eta_max = m

n_tau_next = n_tau_max#int(2**np.floor(np.log(n_tau_max)/np.log(2))) 
n_eta_next = n_eta_max#int(2**np.floor(np.log(n_eta_max)/np.log(2)))

n_tau_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_tau_next),32))).astype(int)))
n_tau_exp = n_tau_exp[:int(.5*len(n_tau_exp))+1]
d_tau_exp = np.diff(n_tau_exp)[-1]
n_tau_lin = np.arange(n_tau_exp[-1],n_tau_next,d_tau_exp)[1:]
n_tau = np.r_[n_tau_exp,n_tau_lin]
n_tau = np.r_[-np.flip(n_tau),0,n_tau]

n_eta_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_eta_next),32))).astype(int)))
n_eta_exp = n_eta_exp[:int(.5*len(n_eta_exp))+1]
d_eta_exp = np.diff(n_eta_exp)[-1]
n_eta_lin = np.arange(n_eta_exp[-1],n_eta_next,d_eta_exp)[1:]
n_eta = np.r_[n_eta_exp,n_eta_lin]
n_eta = np.r_[-np.flip(n_eta),0,n_eta]

tau = n_tau *np.min(np.diff(grd_shr[0][0,:]))
eta = n_eta *np.min(np.diff(grd_shr[1][:,0]))

n_tau,n_eta = np.meshgrid(n_tau,n_eta)
tau,eta = np.meshgrid(tau,eta)

valid = np.zeros(n_tau.shape)
valid[n_tau>=0] = np.array([valid_samples(t,e,U) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())])
valid[n_tau<0] = np.flip(valid[n_tau>0])
print('autcorrelation:', time.process_time()-t0)
t0 = time.process_time()
ucorr = np.zeros(n_tau.shape)
ucorr[n_tau>=0] = np.array([autocorr(t,e,U) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())])
ucorr[n_tau<0] = np.flip(ucorr[n_tau>0])

vcorr = np.zeros(n_tau.shape)
vcorr[n_tau>=0] = np.array([autocorr(t,e,V) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())])
vcorr[n_tau<0] = np.flip(vcorr[n_tau>0])

n_r = np.sqrt(n_tau**2 + n_eta**2)
n_phi = np.arctan2(n_eta,n_tau)

N = np.sum(~np.isnan(U.flatten()))

bins = np.linspace(np.min(n_phi),np.max(n_phi),360)
bin_array = np.ones(n_r.shape)
for i in range(len(bins)-1):
    ind = (n_phi>=bins[i])&(n_phi<bins[i+1])
    bin_array[ind] = bin_array[ind]*.5*(bins[i]+bins[i+1])*180/np.pi

aux = np.argwhere(ucorr<=0)
aux2 = n_r[aux[:,0],aux[:,1]]    
aux3 = n_phi[aux[:,0],aux[:,1]]*180/np.pi

aux4 = np.c_[n_r[aux[np.argsort(aux3),0],aux[np.argsort(aux3),1]],
             valid[aux[np.argsort(aux3),0],aux[np.argsort(aux3),1]],
             ucorr[aux[np.argsort(aux3),0],aux[np.argsort(aux3),1]]/valid[aux[np.argsort(aux3),0],aux[np.argsort(aux3),1]],
             bin_array[aux[np.argsort(aux3),0],aux[np.argsort(aux3),1]]]

bin_center = np.unique(bin_array.flatten())
valid_aux = valid.copy()
valid_aux[valid_aux<=.4*N] = int(.4*N)

#Uncertainty
#Integral length scale

zero_crossings = np.where(np.diff(np.sign(np.nanmean(ucorr,axis=0))))[0]
tau_zero = tau[57,:][zero_crossings][np.argsort(np.abs(tau[57,:][zero_crossings]-tau[57,:][tau[57,:]==0]))[1]]
zero_crossings = np.where(np.diff(np.sign(np.nanmean(ucorr,axis=1))))[0]
eta_zero = eta[:,57][zero_crossings][np.argsort(np.abs(eta[:,57][zero_crossings]-eta[:,57][eta[:,57]==0]))[1]]

valid_zero = valid.copy()
valid_zero[valid==0] = np.nan
ucorr_zero = ucorr.copy()
ucorr_zero[valid==0] = np.nan
u_corr_mean_x = np.nanmean(ucorr,axis=0)/np.nanmean(valid_zero,axis=0)
u_corr_mean_y = np.nanmean(ucorr,axis=1)/np.nanmean(valid_zero,axis=1)
u_corr_mean_x = u_corr_mean_x/np.nanmax(u_corr_mean_x)
u_corr_mean_y = u_corr_mean_y/np.nanmax(u_corr_mean_y)
ind_tau = np.abs(tau[57,:]) > np.abs(tau_zero)
ind_eta = np.abs(eta[:,57]) > np.abs(eta_zero)
u_corr_mean_x[ind_tau] = 0
u_corr_mean_y[ind_eta] = 0
Lx = np.trapz(u_corr_mean_x,tau[57,:])/2
Ly = np.trapz(u_corr_mean_y,eta[:,57])/2
dx = np.min(np.diff(grd_shr[0][0,:]))
dy = np.min(np.diff(grd_shr[1][:,0]))
L_eff = np.sqrt(valid*dx*dy)

e = np.sqrt(2*np.max([Lx,Ly])*su/U_mean**2/L_eff)
#e[e>.08] = 1
N_max = np.nanmax(valid_zero[e>.1])
valid_final = valid.copy()
valid_final[e>.035] = N
Ru = ucorr/valid_final
Rv = vcorr/valid_final
Ruv = vcorr/valid_final
Ru[valid==0]=0
Rv[valid==0]=0
Ruv[valid==0]=0


#zero crossing
rmin = []
bin_c = []
for bc in bin_center:
    ind_aux = aux4[:,3] == bc
    if np.sum(ind_aux)>0:
        rmin.append(np.min(aux4[:,0][ind_aux]))
        bin_c.append(bc)
        ind_N = (n_r>rmin[-1]) & (bin_array==bc)
        n_r[ind_N] = rmin[-1]
    

ucorr_aver_rad = sc.spectra_average(ucorr,(tau[0,:],eta[:,0]),50,angle_bin = 5)

tau_int = np.linspace(np.min(tau[0,tau[0,:]>0]),np.max(tau[0,tau[0,:]>0]),256)
tau_int = np.r_[-np.flip(tau_int),0,tau_int]
eta_int = np.linspace(np.min(eta[eta[:,0]>0,0]),np.max(eta[eta[:,0]>0,:]),256)
eta_int = np.r_[-np.flip(eta_int),0,eta_int]

tau_int, eta_int = np.meshgrid(tau_int,eta_int)
_,_,Ru_i = sc.autocorr_interp_sq(Ru, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,Rv_i = sc.autocorr_interp_sq(Rv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,Ruv_i = sc.autocorr_interp_sq(Ruv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)

ku_r,kv_r,Su_r,Sv_r,Suv_r = sc.spectra_fft((tau_int,eta_int),Ru_i,Rv_i,Ruv_i,K=0)





zero_crossings = np.where(np.diff(np.sign(np.nanmean(r_u,axis=0))))[0]
tau_zero = tau[0,:][zero_crossings][np.argsort(np.abs(tau[0,:][zero_crossings]-tau[0,:][tau[0,:]==0]))[1]]
zero_crossings = np.where(np.diff(np.sign(np.nanmean(r_u,axis=1))))[0]
eta_zero = eta[:,0][zero_crossings][np.argsort(np.abs(eta[:,0][zero_crossings]-eta[:,0][eta[:,0]==0]))[1]]

valid_zero = valid.copy()
valid_zero[valid==0] = np.nan
ucorr_zero = r_u.copy()
ucorr_zero[valid==0] = np.nan
Ax = np.trapz(np.nanmean(r_u,axis=0),tau)/2
Ay = np.trapz(np.nanmean(r_u,axis=1),eta)/2
u_corr_mean_x = np.nanmean(r_u,axis=0)/np.nanmean(valid_zero,axis=0)
u_corr_mean_y = np.nanmean(r_u,axis=1)/np.nanmean(valid_zero,axis=1)
u_corr_mean_x = u_corr_mean_x/np.nanmax(u_corr_mean_x)
u_corr_mean_y = u_corr_mean_y/np.nanmax(u_corr_mean_y)
ind_tau = np.abs(tau[0,:]) > np.abs(tau_zero)
ind_eta = np.abs(eta[:,0]) > np.abs(eta_zero)
u_corr_mean_x[ind_tau] = 0
u_corr_mean_y[ind_eta] = 0
Lx = np.trapz(u_corr_mean_x,tau[0,:])/2
Ly = np.trapz(u_corr_mean_y,eta[:,0])/2

e = np.sqrt(2*Lx*Ly/(valid*dx*dy))
#e[e>.5]=np.nan

valid_zero[e>.3] = np.nanmax(valid_zero[e>.3])

zero_crossings = np.where(np.diff(np.sign(np.nanmean(r_v,axis=0))))[0]
tau_zero = tau[0,:][zero_crossings][np.argsort(np.abs(tau[0,:][zero_crossings]-tau[0,:][tau[0,:]==0]))[1]]
zero_crossings = np.where(np.diff(np.sign(np.nanmean(r_v,axis=1))))[0]
eta_zero = eta[:,0][zero_crossings][np.argsort(np.abs(eta[:,0][zero_crossings]-eta[:,0][eta[:,0]==0]))[1]]

valid_zero = valid.copy()
valid_zero[valid==0] = np.nan
ucorr_zero = r_v.copy()
ucorr_zero[valid==0] = np.nan
Ax = np.trapz(np.nanmean(r_v,axis=0),tau)/2
Ay = np.trapz(np.nanmean(r_v,axis=1),eta)/2
u_corr_mean_x = np.nanmean(r_v,axis=0)/np.nanmean(valid_zero,axis=0)
u_corr_mean_y = np.nanmean(r_v,axis=1)/np.nanmean(valid_zero,axis=1)
u_corr_mean_x = u_corr_mean_x/np.nanmax(u_corr_mean_x)
u_corr_mean_y = u_corr_mean_y/np.nanmax(u_corr_mean_y)
ind_tau = np.abs(tau[0,:]) > np.abs(tau_zero)
ind_eta = np.abs(eta[:,0]) > np.abs(eta_zero)
u_corr_mean_x[ind_tau] = 0
u_corr_mean_y[ind_eta] = 0
Lx = np.trapz(u_corr_mean_x,tau[0,:])/2
Ly = np.trapz(u_corr_mean_y,eta[:,0])/2

e = np.sqrt(2*Lx*Ly/(valid*dx*dy))
e[e>.5]=np.nan




tau_o,eta_o,r_u_o,r_v_o,r_uv_o,valid_o,indicator_o,e,egr = sc.spatial_autocorr_sq(grd,U,V,gamma=gamma,e_lim=.1)
r_u_o[np.reshape(indicator_o,r_u_o.shape)<=.1] = np.nan
r_v_o[np.reshape(indicator_o,r_u_o.shape)<=.1] = np.nan

taur,etra,r_ur,r_vr,r_uvr,valid,indicator,e,egr = sc.spatial_autocorr_sq(grd,U,V, transform = False, transform_r = True,gamma=gamma,e_lim=.05)
r_ur[np.reshape(indicator,r_ur.shape)<=1] = np.nan
r_vr[np.reshape(indicator,r_ur.shape)<=1] = np.nan


r_ur[taur<0] = np.flip(r_ur[taur>0])
r_vr[taur<0] = np.flip(r_vr[taur>0])

tau_int_r = np.linspace(np.min(taur[0,taur[0,:]>0]),np.max(taur[0,taur[0,:]>0]),256)
tau_int_r = np.r_[-np.flip(tau_int_r),0,tau_int_r]
eta_int_r = np.linspace(np.min(etra[etra[:,0]>0,0]),np.max(etra[etra[:,0]>0,:]),256)
eta_int_r = np.r_[-np.flip(eta_int_r),0,eta_int_r]
tau_int_r, eta_int_r = np.meshgrid(tau_int_r,eta_int_r)

_,_,ru_i = sc.autocorr_interp_sq(r_ur, etra, taur, tau_lin = tau_int_r, eta_lin = eta_int_r)
_,_,rv_i = sc.autocorr_interp_sq(r_vr, etra, taur, tau_lin = tau_int_r, eta_lin = eta_int_r)
_,_,ruv_i = sc.autocorr_interp_sq(r_uvr, etra, taur, tau_lin = tau_int_r, eta_lin = eta_int_r)

ru_i[np.isnan(ru_i)]=0
rv_i[np.isnan(rv_i)]=0
ruv_i[np.isnan(ruv_i)]=0

ru_i[tau_int_r<0]=np.flip(ru_i[tau_int_r>0])
rv_i[tau_int_r<0]=np.flip(rv_i[tau_int_r>0])

ru_i[eta_int_r<0]=np.flip(ru_i[eta_int_r>0])
rv_i[eta_int_r<0]=np.flip(rv_i[eta_int_r>0])

k1 = np.fft.fftfreq(ru_i[:-1,:-1].shape[1],d=1/ru_i[:-1,:-1].shape[1])
k2 = np.fft.fftfreq(ru_i[:-1,:-1].shape[1],d=1/ru_i[:-1,:-1].shape[1])
k1,k2 = np.meshgrid(k1,k2)

phase_change_x = (-1)**np.abs(k1) #np.exp(-1j*2*np.pi*k1*np.max(tau_int_r))
phase_change_y = (-1)**np.abs(k2)

ind_r = (np.abs(tau_int_r)<6000) & (np.abs(eta_int_r)<2000)

ru_i_mod = ru_i
ru_i_mod[~ind_r] = 0

fftU2 = np.fft.fftshift(np.fft.fft2(ru_i[:-1,:-1]))*np.diff(tau_int_r[0,:])[0]*np.diff(eta_int_r[:,0])[0]

Su = sp.integrate.simps(np.abs(fftU2),k2_int,axis=1)[k1_int>0]


fftU1D_x = np.fft.fft(ru_i[:-1,:-1],axis=1)*phase_change_x
fftU1D_y = np.fft.fftshift(np.fft.fft(fftU1D_x,axis=0)*phase_change_y)


fft1col = np.fft.fftshift(np.fft.fft(np.real(fftU1D_x[:,255])))*(phase_change_y[:,255])
fft1col_i = np.fft.fftshift(np.fft.fft(1j*np.imag(fftU1D_x[:,255])))*phase_change_y[:,255]

ku_r,kv_r,Su_r,Sv_r,Suv_r = sc.spectra_fft((tau_int_r,eta_int_r),ru_i,rv_i,ruv_i,K=0)

tau_int_o = np.linspace(np.min(tau_o[0,tau_o[0,:]>0]),np.max(tau_o[0,tau_o[0,:]>0]),256)
tau_int_o = np.r_[-np.flip(tau_int_o),0,tau_int_o]
eta_int_o = np.linspace(np.min(eta_o[eta_o[:,0]>0,0]),np.max(eta_o[eta_o[:,0]>0,:]),256)
eta_int_o = np.r_[-np.flip(eta_int_o),0,eta_int_o]
tau_int_o, eta_int_o = np.meshgrid(tau_int_o,eta_int_o)

_,_,ru_i_o = sc.autocorr_interp_sq(r_u_o, eta_o, tau_o, tau_lin = tau_int_o, eta_lin = eta_int_o)
_,_,rv_i_o = sc.autocorr_interp_sq(r_v_o, eta_o, tau_o, tau_lin = tau_int_o, eta_lin = eta_int_o)
_,_,ruv_i_o = sc.autocorr_interp_sq(r_uv_o, eta_o, tau_o, tau_lin = tau_int_o, eta_lin = eta_int_o)

ru_i_o[np.isnan(ru_i_o)]=0
rv_i_o[np.isnan(rv_i_o)]=0
ruv_i_o[np.isnan(ruv_i_o)]=0

ru_i_o[tau_int_o<0]=np.flip(ru_i_o[tau_int_o>0])
rv_i_o[tau_int_o<0]=np.flip(rv_i_o[tau_int_o>0])
ru_i_o[eta_int_o<0]=np.flip(ru_i_o[eta_int_o>0])
rv_i_o[eta_int_o<0]=np.flip(rv_i_o[eta_int_o>0])


ku_o,kv_o,Su_o,Sv_o,Suv_o = sc.spectra_fft((tau_int,eta_int),ru_i_o,rv_i_o,ruv_i_o,K=0)

k1 = np.fft.fftfreq(ru_i_o[:-1,:-1].shape[1],d=1/ru_i_o[:-1,:-1].shape[1])
k2 = np.fft.fftfreq(ru_i_o[:-1,:-1].shape[1],d=1/ru_i_o[:-1,:-1].shape[1])
k1,k2 = np.meshgrid(k1,k2)

phase_change_x = (-1)**np.abs(k1) #np.exp(-1j*2*np.pi*k1*np.max(tau_int_r))
phase_change_y = (-1)**np.abs(k2)

fftU1D_x_o = np.fft.fft(ru_i[:-1,:-1],axis=1)*phase_change_x
fftU1D_y_o = np.fft.fftshift(np.fft.fft(fftU1D_x,axis=0)*phase_change_y)

fftU2_o = np.fft.fftshift(np.fft.fft2(ru_i_o[:-1,:-1]))



###################################################

_,_,Us = sc.shrink(grd,U)
grd_shr_x,grd_shr_y,Vs = sc.shrink(grd,V)
grd_shr = (grd_shr_x,grd_shr_y)

n, m = grd_shr[0].shape
n_tau_max = n
n_eta_max = m

n_tau_next = n_tau_max#int(2**np.floor(np.log(n_tau_max)/np.log(2))) 
n_eta_next = n_eta_max#int(2**np.floor(np.log(n_eta_max)/np.log(2)))

n_tau_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_tau_next),32))).astype(int)))
n_tau_exp = n_tau_exp[:int(.5*len(n_tau_exp))+1]
d_tau_exp = np.diff(n_tau_exp)[-1]
n_tau_lin = np.arange(n_tau_exp[-1],n_tau_next,d_tau_exp)[1:]
n_tau = np.r_[n_tau_exp,n_tau_lin]
n_tau = np.r_[-np.flip(n_tau),0,n_tau]

n_eta_exp = np.unique(np.round(np.exp((np.linspace(0,np.log(n_eta_next),32))).astype(int)))
n_eta_exp = n_eta_exp[:int(.5*len(n_eta_exp))+1]
d_eta_exp = np.diff(n_eta_exp)[-1]
n_eta_lin = np.arange(n_eta_exp[-1],n_eta_next,d_eta_exp)[1:]
n_eta = np.r_[n_eta_exp,n_eta_lin]
n_eta = np.r_[-np.flip(n_eta),0,n_eta]

tau = n_tau *np.min(np.diff(grd_shr[0][0,:]))
eta = n_eta *np.min(np.diff(grd_shr[1][:,0]))

n_tau,n_eta = np.meshgrid(n_tau,n_eta)
tau,eta = np.meshgrid(tau,eta)

validt = np.zeros(n_tau.shape)
validt[n_tau>=0] = np.array([valid_samples(t,e,U) for t, e in zip(n_tau[n_tau>=0].flatten(),n_eta[n_tau>=0].flatten())])
validt[n_tau<0] = np.flip(validt[n_tau>0])

x = grd[0][0,:]
y = grd[1][:,0]
dx = np.diff(grd[0][0,:])[0]
dy = np.diff(grd[1][:,0])[0]
Ur, Vr, mask,mask_int, tri_del = sc.field_rot(x, y, U, V, gamma = gamma, grid = grd,
                                            tri_calc = True)
Ur[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, Ur[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
Ur[~mask_int] = np.nan
Vr[mask_int] = sp.interpolate.CloughTocher2DInterpolator(tri_del, Vr[mask])(np.c_[grd[0].flatten(),grd[1].flatten()][mask_int])
Vr[~mask_int] = np.nan      
Ur = np.reshape(Ur,grd[0].shape)
Vr = np.reshape(Vr,grd[0].shape)

_,_,Ur = sc.shrink(grd,Ur)
grd_shr_xr,grd_shr_yr,Vr = sc.shrink(grd,Vr)
grd_shrr = (grd_shr_xr,grd_shr_yr)

nr, mr = grd_shrr[0].shape
n_tau_maxr = nr
n_eta_maxr = mr

n_tau_nextr = n_tau_maxr#int(2**np.floor(np.log(n_tau_max)/np.log(2))) 
n_eta_nextr = n_eta_maxr#int(2**np.floor(np.log(n_eta_max)/np.log(2)))

n_tau_expr = np.unique(np.round(np.exp((np.linspace(0,np.log(n_tau_nextr),32))).astype(int)))
n_tau_expr = n_tau_expr[:int(.5*len(n_tau_expr))+1]
d_tau_expr = np.diff(n_tau_expr)[-1]
n_tau_linr = np.arange(n_tau_expr[-1],n_tau_nextr,d_tau_expr)[1:]
n_taur = np.r_[n_tau_expr,n_tau_linr]
n_taur = np.r_[-np.flip(n_taur),0,n_taur]

n_eta_expr = np.unique(np.round(np.exp((np.linspace(0,np.log(n_eta_nextr),32))).astype(int)))
n_eta_expr = n_eta_expr[:int(.5*len(n_eta_expr))+1]
d_eta_expr = np.diff(n_eta_expr)[-1]
n_eta_linr = np.arange(n_eta_expr[-1],n_eta_nextr,d_eta_expr)[1:]
n_etar = np.r_[n_eta_expr,n_eta_linr]
n_etar = np.r_[-np.flip(n_etar),0,n_etar]

taur = n_taur *np.min(np.diff(grd_shrr[0][0,:]))
etar = n_etar *np.min(np.diff(grd_shrr[1][:,0]))

n_taur,n_etar = np.meshgrid(n_taur,n_etar)
taur,etar = np.meshgrid(taur,etar)

validr = np.zeros(n_taur.shape)
validr[n_taur>=0] = np.array([valid_samples(t,e,U) for t, e in zip(n_taur[n_taur>=0].flatten(),n_etar[n_taur>=0].flatten())])
validr[n_taur<0] = np.flip(validr[n_taur>0])



S11 = np.cos(-gamma)
S12 = np.sin(-gamma)
T = np.array([[S11,S12], [-S12,S11]])
tau_eta = np.array(np.c_[tau.flatten(),eta.flatten()]).T
tau_eta = np.dot(T,tau_eta)       
tau_prime = tau_eta[0,:]
eta_prime = tau_eta[1,:]            
validt0 = sp.interpolate.griddata(np.c_[tau_prime,eta_prime],
      validt.flatten(), (tau.flatten(),eta.flatten()),
      method='cubic')
validt0 = np.reshape(validt0,tau.shape)

taui = np.linspace(np.min(taur),np.max(taur),100)
etai = np.linspace(np.min(etar),np.max(etar),100)
taui,etai = np.meshgrid(taui,etai)

validri = sp.interpolate.griddata(np.c_[taur.flatten(),etar.flatten()],
      validr.flatten(), (taui.flatten(),etai.flatten()),
      method='cubic')
validti = sp.interpolate.griddata(np.c_[tau.flatten(),eta.flatten()],
      validt.flatten(), (taui.flatten(),etai.flatten()),
      method='cubic')
validti = np.reshape(validti,taui.shape)
validri = np.reshape(validri,taui.shape)

########################
gamma_par,L,alphaepsilon = 2.0,500,.025


k,Psi1 = mn.mann_table_lookup(gamma_par,L,alphaepsilon,ku_r[ku_r>0])
plt.figure()
plt.plot(k,k*Psi1[:,0])
plt.plot(k,k*Psi1[:,1])
plt.plot(k,k*Psi1[:,2])
plt.plot(k,-k*Psi1[:,3])
plt.xscale('log')
plt.yscale('log')



k_u_o = np.fromfile(join(file_in_path_o,'ku_o_r_name90500.02.00.025-1'), dtype=np.float32)
k_v_o = np.fromfile(join(file_in_path_o,'kv_o_r_name90500.02.00.025-1'), dtype=np.float32)
S_u_o = np.fromfile(join(file_in_path_o,'Su_o_r_name90500.02.00.025-1'), dtype=np.float32)
S_v_o = np.fromfile(join(file_in_path_o,'Sv_o_r_name90500.02.00.025-1'), dtype=np.float32)
S_uv_o= np.fromfile(join(file_in_path_o,'Suv_o_r_name90500.02.00.025-1'), dtype=np.float32)

kuo,kvo = np.meshgrid(k_u_o,k_v_o)
S_u_o = np.reshape(S_u_o,kuo.shape)
S_v_o = np.reshape(S_v_o,kuo.shape)
S_uv_o = np.reshape(S_uv_o,kuo.shape)

root = tkint.Tk()
file_in_path_test = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

u,v = sy.wind_sim(alphaepsilon, L, gamma_par, -10, N_x, N_y, L_x, L_y, file_in_path_test,'test')


gamma_par,L,alphaepsilon,seed = 2.0,125.0,.025,-9


u = np.fromfile('simu1252.00.025-9', dtype=np.float32)
v = np.fromfile('simv1252.00.025-9', dtype=np.float32)

gamma_par,L,alphaepsilon,seed = 2.0,125.0,.025,-9

x0 = np.linspace(0,L_x,N_x)
y0 = np.linspace(0,L_y,N_y)
k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x0,y0,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)
i=3
tau_name = 'tau'+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
eta_name = 'eta'+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
r_u_name = 'r_u'+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
r_v_name = 'r_v'+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
r_uv_name = 'r_uv'+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
tau = np.fromfile(tau_name, dtype=np.float32)
eta = np.fromfile(eta_name, dtype=np.float32)
r_u = np.fromfile(r_u_name, dtype=np.float32)
r_v = np.fromfile(r_v_name, dtype=np.float32)
r_uv = np.fromfile(r_uv_name, dtype=np.float32)   


U_file_name = 'U'+str(15)+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)
V_file_name = 'V'+str(15)+str(int(Dir[i]*180/np.pi))+str(L)+str(gamma_par)+str(alphaepsilon)+str(seed)

U = np.reshape(np.fromfile(U_file_name, dtype=np.float32),grid_new[0].shape)
V = np.reshape(np.fromfile(V_file_name, dtype=np.float32),grid_new[0].shape)
U_mean = np.nanmean(U.flatten())
V_mean = np.nanmean(V.flatten())
gamma = np.arctan2(V_mean,U_mean)

tau,eta,r_u,r_v,r_uv,valid,indicator,e,egrad = sc.spatial_autocorr_sq(grid_new,
                                        U,V, transform = False, transform_r = True,gamma=gamma,e_lim=.15,refine=32)         
tau=tau.flatten()
eta=eta.flatten()
r_u,r_v,r_uv = r_u.flatten(),r_v.flatten(),r_uv.flatten()
        
tau_int = np.linspace(np.min(tau[tau>0]),np.max(tau[tau>0]),256)
tau_int = np.r_[-np.flip(tau_int),0,tau_int]
eta_int = np.linspace(np.min(eta[eta>0]),np.max(eta[eta>0]),256)
eta_int = np.r_[-np.flip(eta_int),0,eta_int]
tau_int, eta_int = np.meshgrid(tau_int,eta_int)
         
_,_,ru_i = sc.autocorr_interp_sq(r_u, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,rv_i = sc.autocorr_interp_sq(r_v, eta, tau, tau_lin = tau_int, eta_lin = eta_int)
_,_,ruv_i = sc.autocorr_interp_sq(r_uv, eta, tau, tau_lin = tau_int, eta_lin = eta_int)          
ru_i[np.isnan(ru_i)]=0
rv_i[np.isnan(rv_i)]=0
ruv_i[np.isnan(ruv_i)]=0   
ru_i[tau_int<0]=np.flip(ru_i[tau_int>0])
rv_i[tau_int<0]=np.flip(rv_i[tau_int>0])           
ru_i[eta_int<0]=np.flip(ru_i[eta_int>0])
rv_i[eta_int<0]=np.flip(rv_i[eta_int>0])
ku_r,kv_r,Su_r,Sv_r,Suv_r = sc.spectra_fft((tau_int,eta_int),ru_i,rv_i,ruv_i,K=0)         

#plt.figure();plt.contourf(k_u_o,k_v_o,np.log(S_u_o),np.linspace(np.min(np.log(S_u_o)),np.max(np.log(S_u_o)),10),cmap='jet');plt.colorbar()
#plt.figure();plt.contourf(ku_r,kv_r,np.log(Su_r),np.linspace(np.min(np.log(S_u_o)),np.max(np.log(S_u_o)),10),cmap='jet');plt.colorbar()

Su_o = sp.integrate.simps(S_u_o,k_v_o,axis=0)
Sv_o = sp.integrate.simps(S_v_o,k_v_o,axis=0)
Sur = sp.integrate.simps(Su_r,kv_r,axis=0)
Svr = sp.integrate.simps(Sv_r,kv_r,axis=0)

plt.figure()
plt.plot(k_u_o,k_u_o*Su_o,'-b')
plt.plot(k_u_o,k_u_o*Sv_o,'--b')
plt.plot(ku_r,ku_r*Sur,'-r')
plt.plot(ku_r,ku_r*Svr,'--r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

Suo_ave=sc.spectra_average(S_u_o,(k_u_o, k_v_o),bins=20).S
Svo_ave=sc.spectra_average(S_v_o,(k_u_o, k_v_o),bins=20).S
Sur_ave=sc.spectra_average(Su_r,(ku_r, kv_r),bins=20).S
Svr_ave=sc.spectra_average(Sv_r,(ku_r, kv_r),bins=20).S

Su_o1D_ave = sp.integrate.simps(Suo_ave,k_v_o,axis=0)
Sv_o1D_ave = sp.integrate.simps(Svo_ave,k_v_o,axis=0)
Sh_o1D_ave = sp.integrate.simps(.5*(Suo_ave+Svo_ave),k_v_o,axis=0)
Su_r1D_ave = sp.integrate.simps(Sur_ave,kv_r,axis=0)
Sv_r1D_ave = sp.integrate.simps(Svr_ave,kv_r,axis=0)
Sh_r1D_ave = sp.integrate.simps(.5*(Sur_ave+Svr_ave),kv_r,axis=0)
Su_o1D_ave_it = np.exp(sp.interpolate.interp1d(np.log(k_u_o[k_u_o>0]),
                        np.log(Su_o1D_ave[k_u_o>0]))(np.log(ku_r[ku_r>np.min(k_u_o[k_u_o>0])])))
k1d,Phi1d = mn.mann_table_lookup(gamma_par,L,alphaepsilon,k_u_o[k_u_o>0])

plt.figure();plt.contourf(k_u_o,k_v_o,Suo_ave,np.linspace(np.min(Suo_ave),np.max(Suo_ave),10),cmap='jet');plt.colorbar()
plt.figure();plt.contourf(ku_r,kv_r,np.log(Su_r),np.linspace(np.min(np.log(S_u_o)),np.max(np.log(S_u_o)),10),cmap='jet');plt.colorbar()


plt.figure()
plt.plot(ku_r[ku_r>np.min(k_u_o[k_u_o>0])],ku_r[ku_r>np.min(k_u_o[k_u_o>0])]*Su_o1D_ave_it,'-ok')
plt.plot(ku_r[ku_r>np.min(k_u_o[k_u_o>0])],ku_r[ku_r>np.min(k_u_o[k_u_o>0])]*Su_r1D_ave[ku_r>np.min(k_u_o[k_u_o>0])],'-ob')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

plt.plot(k_u_o,k_u_o*Su_o,'-k')
plt.plot(k_u_o,k_u_o*Sv_o,'--k')

plt.plot(k_u_o,k_u_o*Su_o1D_ave,'-r')
plt.plot(k_u_o,k_u_o*Sv_o1D_ave,'--r')
plt.plot(ku_r,ku_r*Su_r1D_ave,'-b')
plt.plot(ku_r,ku_r*Sv_r1D_ave,'--b')


plt.plot(k_u_o,k_u_o*Sh_o1D_ave,'--r')
plt.plot(ku_r,ku_r*Sh_r1D_ave,'-b')

plt.plot(k_u_o,k_u_o*Su_o1D_ave,'-ok')

plt.plot(ku_r,ku_r*Su_r1D_ave,'-ob')
plt.plot(k_u_o,k_u_o*Su_o,'-k')
plt.plot(k_u_o,k_u_o*Sv_o,'--k')

plt.plot(ku_r,ku_r*Su_r,'-k')
plt.plot(k_u_o,k_u_o*Sv_r,'--k')

plt.plot(k1d,k1d*Phi1d[:,0],'--b')
plt.plot(k1d,k1d*Phi1d[:,1],'--r')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

# Theoretical
logkmin1 = np.min(np.log10(k_u_o[k_u_o>0]))
logkmax1 = np.max(np.log10(k_u_o[k_u_o>0]))
logkmin2 = np.min(np.log10(k_v_o[k_v_o>0]))
logkmax2 = np.max(np.log10(k_v_o[k_v_o>0]))
k1 = np.r_[-np.flip(10**np.linspace(-6,logkmax1,100)),10**np.linspace(-6,logkmax1,100)]#k_u_o[k_u_o!=0]
k2 = np.r_[-np.flip(10**np.linspace(-6,logkmax2,100)),10**np.linspace(-6,logkmax2,100)]#k_v_o[k_v_o!=0]
k3 = np.r_[-np.flip(10**np.linspace(-6,logkmax1,100)),10**np.linspace(-6,logkmax1,100)]#k_v_o[k_v_o!=0]
k1,k2,k3 = np.meshgrid(k1,k2,k3)
Phi11,Phi22,Phi33,Phi12,Phi13,Phi23 = mn.mann_tensor(k1.flatten(),k2.flatten(),k3.flatten(),gamma_par,L,alphaepsilon)
Phi11 = np.reshape(Phi11,k1.shape)
Phi22 = np.reshape(Phi22,k1.shape)
Phi33 = np.reshape(Phi33,k1.shape)
k1 = np.r_[-np.flip(10**np.linspace(-6,logkmax1,100)),10**np.linspace(-6,logkmax1,100)]#k_u_o[k_u_o!=0]
k2 = np.r_[-np.flip(10**np.linspace(-6,logkmax2,100)),10**np.linspace(-6,logkmax2,100)]#k_v_o[k_v_o!=0]
k3 = np.r_[-np.flip(10**np.linspace(-6,logkmax1,100)),10**np.linspace(-6,logkmax1,100)]
k1,k2 = np.meshgrid(k1,k2)
Phi11 = sp.integrate.simps(Phi11,k3,axis=-1)/2/np.pi
Phi22 = sp.integrate.simps(Phi22,k3,axis=-1)/2/np.pi
Phi33 = sp.integrate.simps(Phi33,k3,axis=-1)/2/np.pi



#ind_k = np.sqrt(k1**2+k2**2) < 3/L
#Phi11[ind_k] =  Phi11[ind_k]/(2*np.pi/L_x)

Phi11 = np.exp(sp.interpolate.interp2d(k1[0,:], k2[:,0], np.log(Phi11), kind='cubic')(k_u_o,k_v_o))
Phi22 = np.exp(sp.interpolate.interp2d(k1[0,:], k2[:,0], np.log(Phi22), kind='cubic')(k_v_o,k_u_o))
Phi33 = np.exp(sp.interpolate.interp2d(k1[0,:], k2[:,0], np.log(Phi33), kind='cubic')(k_v_o,k_u_o))

plt.figure();plt.contourf(k_u_o,k_v_o,np.log(Phi11),np.linspace(np.min(np.log(Phi11)),np.max(np.log(Phi11)),10),cmap='jet');plt.colorbar()
plt.figure();plt.contourf(k_u_o,k_v_o,np.log(S_u_o),np.linspace(np.min(np.log(Phi11)),np.max(np.log(Phi11)),10),cmap='jet');plt.colorbar()
plt.figure();plt.contourf(k_u_o,k_v_o,np.log(S_u_o/Phi11),cmap='jet');plt.colorbar()

k1d,Phi1d = mn.mann_table_lookup(gamma_par,L,alphaepsilon,k_u_o[k_u_o>0])

Su_o = sp.integrate.simps(S_u_o,k_v_o,axis=0)
Sv_o = sp.integrate.simps(S_v_o,k_v_o,axis=0)
Su_phi11 = sp.integrate.simps(Phi11,k_v_o,axis=0)
Su_phi22 = sp.integrate.simps(Phi22,k_v_o,axis=0)
Su_phi33 = sp.integrate.simps(Phi33,k_v_o,axis=0)
plt.figure()
#plt.plot(k_u_o,k_u_o*Su_o,'-k')
#plt.plot(k_u_o,k_u_o*Sv_o,'--k')
plt.plot(k_u_o,k_u_o*Su_phi11,'-b')
plt.plot(k_u_o,k_u_o*Su_phi22,'-r')
#plt.plot(k_u_o,k_u_o*Su_phi33,'-g')
plt.plot(k1d,k1d*Phi1d[:,0],'--b')
plt.plot(k1d,k1d*Phi1d[:,1],'--r')
#plt.plot(k1d,k1d*Phi1d[:,2],'-g')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

sc.plot_log2D((k1,k2), Phi11, label_S = "$\log_{10}{S}$", C = 10**-3)
sc.plot_log2D((kuo,kvo), S_u_o , label_S = "$\log_{10}{S}$", C = 10**-3)

plt.figure()
plt.plot(k_u_o,S_u_o[1023,:])
plt.xscale('log')
plt.yscale('log')
plt.plot(k_u_o,Phi11[1023,:])

plt.figure()
plt.plot(k_u_o,S_u_o[1023,:]/Phi11[1023,:])
plt.xscale('log')
plt.yscale('log')





# In[]

root = tkint.Tk()
file_in_path_o = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

root = tkint.Tk()
file_in_path_r = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

onlyfiles_o = [f for f in listdir(file_in_path_o) if isfile(join(file_in_path_o, f))] 
onlyfiles_r = [f for f in listdir(file_in_path_r) if isfile(join(file_in_path_r, f))] 


U = np.reshape(np.fromfile(join(file_in_path_r,'U15270250.02.00.025-1'), dtype=np.float32),grid_new[0].shape)
V = np.reshape(np.fromfile(join(file_in_path_r,'V15270250.02.00.025-1'), dtype=np.float32),grid_new[0].shape)


u = np.reshape(np.fromfile(join(file_in_path_r,'simu2502.00.025-1'), dtype=np.float32),(N_x,N_y)).T
v = np.reshape(np.fromfile(join(file_in_path_r,'simv2502.00.025-1'), dtype=np.float32),(N_x,N_y)).T              
gamma = (2*np.pi-90*np.pi/180)
S11 = np.cos(gamma)
S12 = np.sin(gamma)
T1 = np.array([[1,0,0], [0,1, -L_y/2], [0, 0, 1]])
T2 = np.array([[1,0,0], [0,1, L_y/2], [0, 0, 1]])
R = np.array([[S11,S12,0], [-S12,S11, 0], [0, 0, 1]])
T = np.dot(np.dot(T1,R),T2)
Xx = np.array(np.c_[tri_i.x,tri_i.y,np.ones(len(tri_i.x))]).T
Xx = np.dot(T,Xx)
tri_rot = Delaunay(Xx.T[:,:2], qhull_options = "QJ")               
mask_rot = tri_rot.find_simplex(np.c_[grid[0].flatten(),grid[1].flatten()])==-1                
mask_rot = np.reshape(mask_rot,grid[0].shape)               
u[mask_rot] = np.nan
v[mask_rot] = np.nan

k_u_o,k_v_o,S_u_o,S_v_o,S_uv_o = sc.spatial_spec_sq(x0,y0,np.flipud(np.reshape(u,(N_x,N_y)).T),np.flipud(np.reshape(v,(N_x,N_y)).T),transform = False, ring=False)


#k_u_o = np.fromfile(join(file_in_path_o,'ku_o_r_name90250.02.00.025-1'), dtype=np.float32)
#k_v_o = np.fromfile(join(file_in_path_o,'kv_o_r_name90250.02.00.025-1'), dtype=np.float32)
#S_u_o = np.fromfile(join(file_in_path_o,'Su_o_r_name90250.02.00.025-1'), dtype=np.float32)
#S_v_o = np.fromfile(join(file_in_path_o,'Sv_o_r_name90250.02.00.025-1'), dtype=np.float32)
#S_uv_o= np.fromfile(join(file_in_path_o,'Suv_o_r_name90250.02.00.025-1'), dtype=np.float32)
#kuo,kvo = np.meshgrid(k_u_o,k_v_o)
#S_u_o = np.reshape(S_u_o,kuo.shape)
#S_v_o = np.reshape(S_v_o,kuo.shape)
#S_uv_o = np.reshape(S_uv_o,kuo.shape)



k_u_r = np.fromfile(join(file_in_path_r,'ku_r_name270250.02.00.025-1'), dtype=np.float32)
k_v_r = np.fromfile(join(file_in_path_r,'kv_r_name270250.02.00.025-1'), dtype=np.float32)
S_u_r = np.fromfile(join(file_in_path_r,'Su_r_name270250.02.00.025-1'), dtype=np.float32)
S_v_r = np.fromfile(join(file_in_path_r,'Sv_r_name270250.02.00.025-1'), dtype=np.float32)
S_uv_r= np.fromfile(join(file_in_path_r,'Suv_r_name270250.02.00.025-1'), dtype=np.float32)
kur,kvr = np.meshgrid(k_u_r,k_v_r)
S_u_r = np.reshape(S_u_r,kur.shape)
S_v_r = np.reshape(S_v_r,kur.shape)
S_uv_r = np.reshape(S_uv_r,kur.shape)

_, _, mask, mask_int, tri_del = sc.field_rot(grid_new[0][0,:], grid_new[1][:,0], U, V, gamma=None, tri_calc = True)
k_u_s,k_v_s,S_u_s,S_v_s,S_uv_s = sc.spatial_spec_sq(grid_new[0][0,:],grid_new[1][:,0],U,V,tri_del = tri_del, mask_int = mask_int, tri_calc = False, transform = True)

plt.figure();plt.contourf(k_u_o,k_v_o,np.log(S_u_o),np.linspace(np.min(np.log(Phi11)),np.max(np.log(Phi11)),10),cmap='jet');plt.colorbar()
plt.figure();plt.contourf(k_u_r,k_v_r,np.log(S_u_r),np.linspace(np.min(np.log(Phi11)),np.max(np.log(Phi11)),10),cmap='jet');plt.colorbar()
plt.figure();plt.contourf(k_u_s,k_v_s,np.log(S_u_s),np.linspace(np.min(np.log(Phi11)),np.max(np.log(Phi11)),10),cmap='jet');plt.colorbar()


Su_o = sp.integrate.simps(S_u_o,k_v_o,axis=0)
Su_r = sp.integrate.simps(S_u_r,k_v_r,axis=0)
Su_s = sp.integrate.simps(S_u_s,k_v_s,axis=0)
plt.figure()
plt.plot(k_u_o,k_u_o*Su_o,'-ok')
plt.plot(k_u_r,k_u_r*Su_r,'-ob')
plt.plot(k_u_s,k_u_s*Su_s,'-og')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

Sv_o = sp.integrate.simps(S_v_o,k_v_o,axis=0)
Sv_r = sp.integrate.simps(S_v_r,k_v_r,axis=0)
Sv_s = sp.integrate.simps(S_v_s,k_v_s,axis=0)
plt.figure()
plt.plot(k_u_o,k_u_o*Sv_o,'-ok')
plt.plot(k_u_r,k_u_r*Sv_r,'-ob')
plt.plot(k_u_s,k_u_s*Sv_s,'-og')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')




Su_o = sp.integrate.simps(.5*(S_u_o+S_v_o),k_v_o,axis=0)
Su_r = sp.integrate.simps(.5*(S_u_r+S_v_r),k_v_r,axis=0)
Su_s = sp.integrate.simps(.5*(S_u_s+S_v_s),k_v_s,axis=0)
plt.figure()
plt.plot(k_u_o,k_u_o*Su_o,'-ok')
plt.plot(k_u_r,k_u_r*Su_r,'-ob')
plt.plot(k_u_s,k_u_s*Su_s,'-og')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')


Suo_ave=sc.spectra_average(S_u_o,(k_u_o, k_v_o),bins=20).S
Svo_ave=sc.spectra_average(S_v_o,(k_u_o, k_v_o),bins=20).S
Sur_ave=sc.spectra_average(S_u_r,(k_u_r, k_v_r),bins=20).S
Svr_ave=sc.spectra_average(S_v_r,(k_u_r, k_v_r),bins=20).S
Sus_ave=sc.spectra_average(S_u_s,(k_u_s, k_v_s),bins=20).S
Svs_ave=sc.spectra_average(S_v_s,(k_u_s, k_v_s),bins=20).S

Su_o1D_ave = sp.integrate.simps(.5*(Suo_ave+Svo_ave),k_v_o,axis=0)
Su_r1D_ave = sp.integrate.simps(.5*(Sur_ave+Svr_ave),k_v_r,axis=0)
Su_s1D_ave = sp.integrate.simps(.5*(Sus_ave+Svs_ave),k_v_s,axis=0)
plt.figure()
plt.plot(k_u_o,k_u_o*Su_o1D_ave,'-ok')
plt.plot(k_u_r,k_u_r*Su_r1D_ave,'-ob')
plt.plot(k_u_s,k_u_s*Su_s1D_ave,'-og')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')

Su_o1D_ave_it = np.exp(sp.interpolate.interp1d(np.log(k_u_o[k_u_o>0]),np.log(Su_o1D_ave[k_u_o>0]))(np.log(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])])))

plt.figure()
plt.plot(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])],k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])]*Su_o1D_ave_it,'-ok')
plt.plot(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])],k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])]*Su_r1D_ave[k_u_r>np.min(k_u_o[k_u_o>0])],'-ob')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')


plt.figure()
plt.plot(k_u_r[k_u_r>np.min(k_u_o[k_u_o>0])],Su_r1D_ave[k_u_r>np.min(k_u_o[k_u_o>0])]/Su_o1D_ave_it,'-ok')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k1')
plt.ylabel('k1*S')



