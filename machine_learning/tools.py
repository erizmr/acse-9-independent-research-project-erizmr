from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from net_tools import *


def data_collect(train_data, train_end_frame, test_end_frame):
  """
    Collect, stack and divide data into train and test set
  """
  train_data_features = train_data[0][1:train_end_frame]
  test_data_features = train_data[0][train_end_frame:test_end_frame]

  for train in train_data[1:]:
    train_data_i = train[1:train_end_frame]
    test_data_i = train[train_end_frame:test_end_frame]

    train_data_features = np.vstack([train_data_features, train_data_i])
    test_data_features = np.vstack([test_data_features, test_data_i])

  return train_data_features, test_data_features

def set_seed(seed):
  """
  Control the seed to help re-run experiments
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled   = False

  return True


def norm201(data):
  """Normalization function, make the range of the data in the range 0-1
  """
  for i, chip in enumerate(data):
    data[i][0] = data[i][0] / np.max(chip[0])
    data[i][1] = data[i][1] / np.max(chip[1])
  return data



def train(model, optimizer, criterion, data_loader, device='cpu'):
    """Train function, update every iteration, calculate training loss """
    model.train()
    train_loss, flow_RMSE = 0, 0

    for X, y in data_loader:
        X = X.view(-1, 2, 180, 300)
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X.view(-1, 2, 180, 300))

        flow_RMSE += upRMSE(output[0], y) * X.size(0)
        loss = criterion(output, y)
        loss.backward()

        train_loss += loss * X.size(0)
        optimizer.step()

    return train_loss/len(data_loader.dataset), flow_RMSE/len(data_loader.dataset)

def validate(model, criterion, data_loader, device='cpu'):
    """Validation function, calculate validation loss """
    model.eval()
    validation_loss = 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            output = model(X.view(-1, 2, 180, 300))

            loss = criterion(output, y)
            #print('validation loss', loss)
            validation_loss += loss * X.size(0)

    return validation_loss/len(data_loader.dataset), validation_loss/len(data_loader.dataset)



def infer_plot(predictions, device='cpu', n_max=4, contour=True, quiv=False ,save=False):
    """Visualization function for predictions 
    There are some extra functions could be choosn:
        
    contour: whether to draw the contour line
    quiv: whether to draw the quiver plot
    save: whether to save the figures
    """

    try:
        style = cmocean.cm.haline
    except:
        style = 'viridis'
    v_min = 0.01
    v_max = 0.05
    scale_factor = 20.0
    X = np.arange(0, 300, 10)
    Y = np.arange(0, 180, 10)
    
    if device=='cuda':
        data = predictions.cpu().detach().numpy()/scale_factor
    else:
        data = predictions.detach().numpy()/scale_factor

    for n in range(n_max):

      U = ndimage.interpolation.zoom(data[n][0],.1)*5
      V = ndimage.interpolation.zoom(data[n][1],.1)*5

      fig = plt.figure(figsize=(18, 3))
      ax12 = fig.add_subplot(131)
      ax12.set_xlabel("$v_x$",fontsize=12)
      mappable1 = ax12.pcolormesh(data[n][0], cmap=style, vmin=v_min, vmax=v_max)
      fig.colorbar(mappable1, ax=ax12)
      
      
      ax13 = fig.add_subplot(132)
      ax13.set_xlabel("$v_y$",fontsize=12)
      mappable1 = ax13.pcolormesh(data[n][1], cmap=style, vmin=-v_min*2, vmax=v_min*2)
      fig.colorbar(mappable1, ax=ax13)
      
      
      ax14 = fig.add_subplot(133)
      ax14.set_xlabel("$v_m$",fontsize=12)
      mappable1 = ax14.pcolormesh(np.power(np.power(data[n][0],2)+\
                  np.power(data[n][1],2), 0.5), cmap=style, vmin=v_min, vmax=v_max)
      
      fig.colorbar(mappable1, ax=ax14)
      
      if quiv:
          ax14.quiver(X, Y, U, V)#, scale=1.5)
      
      if contour:
          c1 = ax12.contour(data[n][0], 8, linewidths=1.0, cmap=style)
          ax12.clabel(c1, inline=1,inline_spacing= 3, fontsize=8, colors='k', use_clabeltext=1)
          
          c2 = ax13.contour(data[n][1], 8, linewidths=1.0, cmap=style)
          ax13.clabel(c2, inline=1,inline_spacing= 3, fontsize=8, colors='k', use_clabeltext=1)
          
          c3 = ax14.contour(np.power(np.power(data[n][0],2)+\
                  np.power(data[n][1],2), 0.5), 8, linewidths=1.0, cmap=style)
          ax14.clabel(c3, inline=1,inline_spacing= 3, fontsize=8, colors='k', use_clabeltext=1)
       
      if save:
          plt.savefig(drive_path+"/results/predictions_frame%d.png"%n)
