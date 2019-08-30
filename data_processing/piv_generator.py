import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os


def kernal_generator(n=1):
    """Generate the kernal matrix"""
    kernal = []
    size = (2*n+1)*(2*n+1)
    # ind_x * (2*n+1) + ind_y = i
    for i in range(size):
        ind_x = np.floor(i / (2*n+1))
        ind_y = i % (2*n+1)
        kernal.append([int(ind_x-n),int(ind_y-n)])
    return kernal


def gauss_2d(dx, dy, dp=2.81, I_0=1.0):
	"""Generate particle gray value as gaussian distribution"""
    return I_0*np.exp(8*(-np.power(dx,2.0)-np.power(dy,2.0))/(dp*dp))


def PIV_generator(node_array, size=(450,180), ranges=[[0.5, 1.5],[0.0,0.6]], dp=2.81, kernal_size=1, noise=False):
	"""Generate PIV like image using vtp files """
	
    node_mat = np.zeros(size)
    fail = 0
    node_new = node_array * size[0]/1.5
    
    # Calculate the kernal matrix
    kernal = kernal_generator(kernal_size)
    
    
    for x_0,y_0 in np.floor(node_new)[:,0:2]:
        if int(x_0) >= size[0]:
            x_0 = size[0]-1
            fail += 1
        if int(y_0) >= size[0]:
            y_0 = size[1]-1
            fail += 1
        
        id_x = int(x_0)
        id_y = int(y_0)
        
        for ind in kernal:
            if id_x+ind[0] < 0 or id_x+ind[0] >= size[0] or id_y+ind[1] < 0 or id_y+ind[1] >= size[1]:
                fail += 1
                continue
            else:
                node_mat[id_x, id_y] += gauss_2d(ind[0], ind[1], dp=dp)

    if noise:
        thr = 0.8*np.max(node_mat)
        #print(thr)
        # Add some noise
        for i in range(size[0]):
            for j in range(size[1]):
                if  np.random.uniform(0.0, 1.0) < 0.05 and  node_mat[i,j] < thr:
                    node_mat[i,j] += np.random.uniform(0.01, thr)
    
    print("Particle density: ", len(np.where(node_mat!=0)[0])/(size[0]*size[1]))
    print("Out of boundary numbers: ", fail)
    return (node_mat[int(size[0]*ranges[0][0]/1.5):int(size[0]*ranges[0][1]/1.5), \
                    int(size[1]*ranges[1][0]/0.6):int(size[1]*ranges[1][1]/0.6)]).T
