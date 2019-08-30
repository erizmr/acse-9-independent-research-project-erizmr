import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os

from scipy.interpolate import griddata
from collections.abc import Iterable


def interpolate_field(data, ranges=None, x_points=100, dim=2):
    """Do interpolation in the field using the mesh point data
    to make the a complete data matrix
    2D interpolation is developed now, there are three components:
    vx, vy, v_magnitude
    """
    
    field_names = [item for item in data.keys()].pop()
    while not isinstance(field_names, list):
        field_names = [field_names]
    
    if ranges is None:
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
    else:
        xmin, xmax = ranges[0][0], ranges[0][1]
        ymin, ymax = ranges[1][0], ranges[1][1]
    
    npts_x = x_points
    npts_y = np.floor(npts_x * (ymax - ymin) / (xmax - xmin))
    # define grid
    xi = np.linspace(xmin, xmax, npts_x)
    yi = np.linspace(ymin, ymax, npts_y)
        
    x,y,z= data['mesh'][:,0] , data['mesh'][:,1] , data['mesh'][:,2]
    
    field_value_dict = dict()
    if dim==2:
        # 2D interpolate
        for name in field_names:
            size = data[name].shape
            data_components = []
            for i in range(size[1]-1):
                data_components.append(griddata((x, y), data[name][:,i], (xi[None,:], yi[:,None]), method='cubic'))

            magnitude = np.sqrt(np.power(data[name][:,0],2) + np.power(data[name][:,1],2))
            data_components.append(griddata((x, y), magnitude, (xi[None,:], yi[:,None]), method='cubic'))
            field_value_dict[name] = data_components   
    else:
        print("Under construction")
    return field_value_dict