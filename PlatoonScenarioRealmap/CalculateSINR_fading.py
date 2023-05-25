# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:13:46 2021

@author: CSUST
"""
from SubchCollisionCheck import SubchCollisionCheck
from Distance import Distance
import numpy as np
from scipy.stats import nakagami

import seaborn as sns
import matplotlib.pyplot as plt



sns.set_style("whitegrid",{'axes.edgecolor': '.0', 'axes.linewidth': 1.0,'grid.color': '1.0','xtick.color' : '0.0',})

sns.set_style("ticks",{'axes.edgecolor': '.0', 'axes.linewidth': 1.0,'grid.color': '1.0','xtick.color' : '0.0',})
#sns.set_style("whitegrid", {'font.family': ['Times New Roman'], "axes.labelcolor" : "0.0",  'grid.color': '.7','grid.linestyle': '--', 'text.color' : '0.01',  'xtick.color' : '0.01'})
sns.set_context("notebook", font_scale = 2)
colors=[sns.xkcd_rgb['velvet'],sns.xkcd_rgb['twilight blue'],sns.xkcd_rgb['leaf'],sns.xkcd_rgb['orangered'],sns.xkcd_rgb['slate grey'],sns.xkcd_rgb['medium green']]

#plt.rcParams['font.sans-serif'] = ['SimSun']



def CalculateSINR(i,j,R,NumVehicle,VehicleLocation,power):
    interference=0
    for s in range(0,NumVehicle):
        if True:
        #if Distance(s,j,VehicleLocation)<dint:
            if s == i or s == j or VehicleLocation[s]==VehicleLocation[i] or VehicleLocation[s]==VehicleLocation[j]:
                continue
            else:
                if SubchCollisionCheck(i,s,R) == 1:
                    #same = 1
                    interference += power*Distance(s,j,VehicleLocation)**(-3.68)
    SINR = (power*Distance(i,j,VehicleLocation)**(-3.68))/(interference+10**(-6.46))
    return SINR

def CalculateSINR_fading(i,j,R,NumVehicle,VehicleLocation,power,fading,scale_param_omega):
    
    # parameter for nakagami-m fading (short-term fading)
    # Nakagami-m parameter
    #fading_gain_db = np.random.gamma(m, 1) ** 0.5  # Fading gain in dB (assuming gamma distribution)
    #fading_gain_linear = 10 ** (fading_gain_db / 10)
    
    interference=0
    if fading=='on':
      if abs(i-j)==1:
          fading_param_m=5
      else:
          fading_param_m=1
    else:
        fading_param_m=100000000  
        
    for s in range(0,NumVehicle):
        if True:
        #if Distance(s,j,VehicleLocation)<dint:
            if s == i or s == j or VehicleLocation[s]==VehicleLocation[i] or VehicleLocation[s]==VehicleLocation[j]:
                continue
            else:
                if SubchCollisionCheck(i,s,R) == 1:
                    #same = 1
                    if fading=='on':
                        if abs(s-j)==1:
                            fading_param_m_int=5
                        else:
                            fading_param_m_int=1
                    else:
                        fading_param_m_int=100000000
                        
                    fading_gain_linear = nakagami.rvs(fading_param_m_int, scale=scale_param_omega) 
                    interference += (fading_gain_linear)*power*Distance(s,j,VehicleLocation)**(-3.68)
    fading_gain_linear = nakagami.rvs(fading_param_m, scale=scale_param_omega, size=1) 
    SINR = ((fading_gain_linear)*power*Distance(i,j,VehicleLocation)**(-3.68))/(interference+10**(-6.46))
    return SINR

'''
def CalculateSINR_fading_test(d1,d2,power,fading_param_m,scale_param_omega):
    
    # parameter for nakagami-m fading (short-term fading)
    # Nakagami-m parameter
    #fading_gain_db = np.random.gamma(m, 1) ** 0.5  # Fading gain in dB (assuming gamma distribution)
    #fading_gain_linear = 10 ** (fading_gain_db / 10)
    
    interference=0
    fading_gain_linear = nakagami.rvs(fading_param_m, scale=scale_param_omega) 
    interference = (fading_gain_linear)*power*d2**(-3.68)
    fading_gain_linear = nakagami.rvs(fading_param_m, scale=scale_param_omega)
    SINR = ((fading_gain_linear)*power*d1**(-3.68))/(interference+10**(-6.46))
    return SINR


num_runs = 100000
distances = np.linspace(5, 400, 50) # transmission distances in meters
collisions = np.zeros(len(distances)) # access collision probability
total_loss = np.zeros(len(distances)) # access collision probability

for i in range(len(distances)):
    d1=distances[i]
    fading_param_m=1
    scale_param_omega=1
    power=200
    fading_gain_k = nakagami.rvs(fading_param_m, scale=scale_param_omega, size=num_runs) 
    d2=10
    interference = (fading_gain_k)*power*d2**(-3.68)
    fading_gain_i = nakagami.rvs(fading_param_m, scale=scale_param_omega, size=num_runs)
    SINR_loss = ((fading_gain_i)*power*d1**(-3.68))/(interference+10**(-6.46))
    #SINR_collision = ((fading_gain_i)*power*d1**(-3.68))/(interference+10**(-6.46))
    SINR_fading = ((fading_gain_i)*power*d1**(-3.68))/(10**(-6.46))
    sinr_th_db=2.76
    sinr_th=10**(sinr_th_db/10)
    
    num_loss = np.sum(SINR_loss < sinr_th)
    #num_collisions = np.sum(SINR_collision < sinr_th)
    num_fading = np.sum(SINR_fading < sinr_th)
    #collisions[i] = num_collisions/num_runs
    fading[i] = num_fading/num_runs
    total_loss[i] = num_loss/num_runs
    collisions[i]=total_loss[i] - fading[i]


# fading impact on packet loss
distances = np.linspace(5, 300, 100) # transmission distances in meters
collisions = np.zeros(len(distances)) # access collision probability
total_loss = np.zeros(len(distances)) # access collision probability

fading_all=[]

num_runs = 100000

fading_param_m_range=[0.1,0.5,1,5,10]
for fading_param_m in fading_param_m_range:
    fading = np.zeros(len(distances)) # access collision probability
    for i in range(len(distances)):
        d1=distances[i]
        d2=10
        scale_param_omega=1
        power=200
        fading_gain_k = nakagami.rvs(fading_param_m, scale=scale_param_omega, size=num_runs) 
        fading_gain_i = nakagami.rvs(fading_param_m, scale=scale_param_omega, size=num_runs)
        int_k = (fading_gain_k)*power*d2**(-3.68)
        SINR_fading = ((fading_gain_i)*power*d1**(-3.68))/(10**(-6.46))
        sinr_th_db=2.76
        sinr_th=10**(sinr_th_db/10)
        num_fading = np.sum(SINR_fading < sinr_th)
        #collisions[i] = num_collisions/num_runs
        fading[i] = num_fading/num_runs
    #print(fading)
    fading_all.append(fading)
    
fig1=plt.figure(1,figsize=(10,8))
for i in range(len(fading_param_m_range)):
    plt.plot(distances, fading_all[i], linewidth=3,markeredgewidth=4.4,alpha=1,label='m='+str(fading_param_m_range[i]))
plt.legend(loc='best',fancybox=True,shadow=True, fontsize=25)

plt.xlabel('Transmission distance (m)',fontsize=28)
plt.ylabel('Packet loss ratio caused by fading',fontsize=28)
plt.xlim(5, 300)
#plt.ylim(min(min(throughput_all),min(throughput_wf_all))*0.9, max(max(throughput_all),max(throughput_wf_all))*1.05)
plt.ylim(0, 1.02)

x=[5,50,100,150,200,250,300]
labels=[str(i) for i in x]
plt.xticks(x,labels)
plt.tight_layout() 
plt.show()
fig1.savefig("fig/fading_without_interference.png", dpi=300)
   




def fading_m(d):
    if d<=5:
        return 1
    else:
        return 1

    

# collision+fading impact on packet loss
def output_collision_fading_pls(m,distances,d2_range):
    #d2_range=[5,20,50,100,200] 
    collision_fading=[]
    num_runs = 10000
    for d2 in d2_range:
        fading = np.zeros(len(distances)) # access collision probability
        for i in range(len(distances)):
            d1=distances[i]
            #d2=10
            scale_param_omega=1
            power=200
            fading_gain_k = nakagami.rvs(m, scale=scale_param_omega, size=num_runs) 
            fading_gain_i = nakagami.rvs(m, scale=scale_param_omega, size=num_runs)
            int_k = (fading_gain_k)*power*d2**(-3.68)
            SINR_fading = ((fading_gain_i)*power*d1**(-3.68))/(int_k+10**(-6.46))
            sinr_th_db=2.76
            sinr_th=10**(sinr_th_db/10)
            num_fading = np.sum(SINR_fading < sinr_th)
            #collisions[i] = num_collisions/num_runs
            fading[i] = num_fading/num_runs
        #print(fading)
        collision_fading.append(fading)
        return collision_fading




d2_range=[5,20,50,100,200] 
for t in range(len(fading_param_m_range)):
    collision_fading= output_collision_fading_pls(fading_param_m_range[t],distances)  
    fig1=plt.figure(1,figsize=(10,8))
    plt.plot(distances, fading_all[t], '>--', linewidth=3,markeredgewidth=2,markerfacecolor="None",markersize=10,alpha=1,label='fading, m='+str(fading_param_m_range[i]))
    for i in range(len(d2_range)):
        plt.plot(distances, collision_fading[i], 'o-', linewidth=3,markeredgewidth=2,markerfacecolor="None",markersize=10,alpha=1,label='collision+fading, $d_{int}$='+str(d2_range[i])+'m')
    plt.legend(loc='best',fancybox=True,shadow=True, fontsize=25)
    plt.xlabel('Transmission distance (m)',fontsize=28)
    plt.ylabel('Packet loss ratio',fontsize=28)
    plt.xlim(5, 300)
    x=[5,50,100,150,200,250,300]
    labels=[str(i) for i in x]
    plt.xticks(x,labels)
    plt.ylim(0, 1.02)
    plt.tight_layout() 
    plt.show()
    filename='fading_with_interference_m'+str(fading_param_m_range[t])
    fig1.savefig("fig/%s.png"%filename, dpi=300)




fig2=plt.figure(1,figsize=(10,8))

for i in [2,3]:
    plt.plot(distances, fading_all[i], '>--', linewidth=3,markeredgewidth=2,markerfacecolor="None",markersize=10,alpha=1,label='fading, m='+str(fading_param_m_range[i]))
for i in range(1,len(d2_range)):
    plt.plot(distances, collision_fading[i], 'o-', linewidth=3,markeredgewidth=2,markerfacecolor="None",markersize=10,alpha=1,label='collision+fading, $d_{int}$='+str(d2_range[i])+'m')
plt.legend(loc='lower right',fancybox=True,shadow=True, fontsize=23)

plt.xlabel('Transmission distance (m)',fontsize=28)
plt.ylabel('Packet loss ratio',fontsize=28)

plt.xlim(5, 300)
x=[5,50,100,150,200,250,300]
labels=[str(i) for i in x]
plt.xticks(x,labels)
#plt.ylim(min(min(throughput_all),min(throughput_wf_all))*0.9, max(max(throughput_all),max(throughput_wf_all))*1.05)

plt.ylim(0, 1.02)
plt.tight_layout() 
#plt.yscale('log')

plt.show()
fig2.savefig("fig/fading_collision_compare.png", dpi=300)
'''


'''
def calc_sinr(path_loss, fading_gain, interference, noise):
    """
    Calculate SINR (Signal-to-Interference-plus-Noise Ratio) given path loss, fading gain, interference, and noise.
    
    Args:
        path_loss (float): Path loss in dB.
        fading_gain (float): Fading gain in dB.
        interference (float): Interference in dB.
        noise (float): Noise in dB.
        
    Returns:
        sinr (float): SINR in dB.
    """
    linear_path_loss = 10 ** (-path_loss / 10)
    linear_fading_gain = 10 ** (fading_gain / 10)
    linear_interference = 10 ** (interference / 10)
    linear_noise = 10 ** (noise / 10)
    
    sinr = linear_fading_gain* linear_path_loss / (linear_interference + linear_noise)
    sinr_db = 10 * np.log10(sinr)
    
    return sinr_db




# Example usage:

# Define parameters
path_loss_db = 100  # Path loss in dB
fading_param_m=4
scale_param_omega=5
interference_db = np.random.exponential(1, size=1000)  # Interference in dB (assuming exponential distribution)
noise_db = -100  # Noise in dB

# Calculate SINR for each fading gain and interference realization
sinr_db = []
for i in range(1000):
    fading_gain_linear = 10 ** (nakagami.rvs(fading_param_m, scale=scale_param_omega) / 10)
    sinr_db.append(calc_sinr(path_loss_db, fading_gain_linear, interference_db[i], noise_db))

# Calculate average SINR
avg_sinr_db = np.mean(sinr_db)

print("Average SINR (dB):", avg_sinr_db)
'''