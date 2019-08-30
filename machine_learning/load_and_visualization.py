from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper')
import matplotlib.patches as mpatches

try:
    import cmocean
except:
    pass

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

from tools import *
from ParticleNet import *

#--------------------------Load the test data---------------------#
drive_path = "../data"
data_collection = []
label_collection = []
data_collection.append(np.load(drive_path+"/synthetic_data/Re3450_DH2/traindata_dp3.00_ir15/particle_positions.npy")) # Re 3450
label_collection.append(np.load(drive_path + "/synthetic_data/Re3450_DH2/labels/vxy_label.npy"))


train_end_frame = 4
test_end_frame = 5
train_data_features, test_data_features = data_collect(data_collection, train_end_frame, test_end_frame)
train_data_label, test_data_label  = data_collect(label_collection ,train_end_frame, test_end_frame)
train_data_features = norm201(train_data_features)
test_data_features = norm201(test_data_features)


##------------------Specify the model path--------------------#
model_save_name = 'ParticleNet_test.pt'
PATH = "../model_trained/"+model_save_name


model_loaded = ParticleNet()
model_loaded.load_state_dict(torch.load(PATH, map_location=device)['model_state_dict'])
model_loaded.to(device)
model_loaded.eval()
print('Model load successfully.')

##----------------------------Make predictions----------------------------------##
y_prediction = model_loaded(torch.from_numpy(train_data_features).float().to(device))
h, w = 180,300
unsampled_y_prediction = F.interpolate(y_prediction, (h,w), mode='bilinear', align_corners=False)

##--------------------- Visualize the images---------------------------------------##
infer_plot(unsampled_y_prediction, device, n_max=4, contour=True, quiv=False ,save=False)