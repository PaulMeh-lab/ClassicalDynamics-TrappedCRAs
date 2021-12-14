# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:54:48 2021

@author: E20
"""


#%%

import numpy as np
import matplotlib.pyplot as plt


# Numerical aperture
NA=0.24
# Wavelength (um)
lamda=0.82

# Transverse image size (px)
N_pix_trans = 1024
# Longitudinal image size (px)
N_pix_z = 100
# Transverse pixel size (um)
pix_trans = 0.05
# Longitudinal pixel size (um)
pix_z = 0.2

# Power (W)
P0 = 0.2


#waist in the focal plane, in µm
w_out=lamda/np.pi/NA
w_out_pix=w_out/pix_trans   #in pixel #

#waist in the SLM plane, in pixel #
w_in_pix=N_pix_trans/np.pi/w_out_pix

# Area of the SLM, in pixels
u_mesh,v_mesh,z_mesh=np.meshgrid(np.linspace(-N_pix_trans//2,N_pix_trans//2,N_pix_trans),
                                 np.linspace(-N_pix_trans//2,N_pix_trans//2,N_pix_trans),
                                 np.linspace(-N_pix_z//2,N_pix_z//2,N_pix_z))

#radius for the pi shift on the SLM, in pixel
#p=np.sqrt(np.log(2))*w_in_pix #Dark bob
p=0.93*w_in_pix 
print(p)
#mask phase of the circular pi-shift
phi=np.pi*np.heaviside(1-(u_mesh**2+v_mesh**2)/p**2, 0)

#Intensity distribution of the input beam, in the SLM plane
I=np.exp(-2*(u_mesh**2+v_mesh**2)/w_in_pix**2)

#Phase along Z around the SLM plane
z_trans= z_mesh*pix_z * np.pi/lamda*(u_mesh**2+v_mesh**2)/w_in_pix**2*NA**2

# Fourier transform
F=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.sqrt(I)*np.exp(1j*(phi+z_trans)),
                                              axes=(0,1)),axes=(0,1)),axes=(0,1))
S=np.abs(F)**2





#%%


#Normalisation
P=np.sum(S,axis=(0,1))*pix_trans**2
S_norm=S*P0/P


# Plot
roi=int(5*w_in_pix)

#result array, indexed in pixels
S_res=S_norm[N_pix_trans//2-roi:N_pix_trans//2+roi +1,N_pix_trans//2-roi:N_pix_trans//2+roi +1,:]

print(np.shape(S_res))

print(roi)

#extents of the axes, in µm
extent_xy = np.asarray([N_pix_trans//2-roi, N_pix_trans//2+roi,
                        N_pix_trans//2-roi , N_pix_trans//2+roi])*pix_trans
extent_xz = np.asarray([0, N_pix_z*pix_z,
                        (N_pix_trans//2-roi)*pix_trans,
                        (N_pix_trans//2+roi)*pix_trans])

fig,ax=plt.subplots(1, 2)
ax[0].imshow(S_res[:,:,N_pix_z//2], extent = extent_xy)
ax[0].set_xlabel('µm' )
ax[0].set_ylabel('µm')

Ax=ax[1].imshow(S_res[:,roi,:], extent = extent_xz, aspect = 'equal')
cbar = fig.colorbar(Ax,ax=ax)
cbar.set_label('Intensity in W/µm²')


#%%


import pickle

Bob_data = {'N_pix_trans': 2*roi + 1,
           'N_pix_z': N_pix_z,
           'pix_trans': pix_trans,
           'pix_z': pix_z,
           'P0': P0,
           'array': S_res}

output = open('high_bob_profile.pkl', 'wb')
pickle.dump(Bob_data, output)
output.close()




#%%

pkl_file = open('high_bob_profile.pkl', 'rb')

data1 = pickle.load(pkl_file)
plt.plot(data1['array'][:,74,:50])

pkl_file.close()


#%%



a = np.array([1,1,1,1])

print(np.shape(a))

