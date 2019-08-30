from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper')
import matplotlib.patches as mpatches
try:
    import cmocean
except:
    pass
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import datetime

from tools import *
from custom_dataset import *
from ParticleNet import *


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

drive_path = "../data"


##-------------------Set training parameters--------------##

seed = 42
lr = 1e-3
batch_size = 8
test_batch_size = 8
n_epochs = 200
weight_decay = 1e-3
random_seed = False

##---------------- Model selection---------------------##
model_input = ParticleNet().to(device)

##-----------------Set loading parameters---------------##

train_end_frame = 5
test_end_frame = 10

data_collection = []
label_collection = []

# Construct the super mini size data for test
#train_Re_num_set = [1150, 2300, 4600]
train_Re_num_set = [3450]
#train_dp_set = [2.00, 3.00, 3.80]
train_dp_set = [2.00]
test_Re_num_set = [3450]
test_dp_set = [2.00, 3.00, 3.80]
test_insert_rate_set = [15, 38, 75, 150]


##---------------- Read all taining data---------------##
for re in train_Re_num_set:
  if re == 1150:
    train_insert_rate_set = [5, 12, 25, 50]
  if re == 2300:
    train_insert_rate_set = [10, 25, 50, 100]
  if re == 3450:
    train_insert_rate_set = [15]#, 38, 75, 150]
  if re == 4600:
    train_insert_rate_set = [20, 50, 100, 200]
  for dp in train_dp_set:
    for ir in train_insert_rate_set:
      filepath = drive_path+"/synthetic_data/Re%d_DH2/traindata_dp%.2f_ir%d/particle_positions.npy"%(re, dp, ir)
      data_collection.append(np.load(filepath))
      print("Insert Rate: ", ir, " Finished.")
    print("Particle Diameter : ", dp, " Finished.")
  print("Re ",re," Finished.")

train_data_features, test_data_features = data_collect(data_collection, train_end_frame, test_end_frame)
print("Check training data: ", train_data_features.shape)
print("Check test data: ", test_data_features.shape)

##---------------- Read all labels---------------##
for re in train_Re_num_set:
  filepath = drive_path+"/synthetic_data/Re%d_DH2/labels/vxy_label.npy"% re
  if re == 1150:
    train_insert_rate_set = [5, 12, 25, 50]
  if re == 2300:
    train_insert_rate_set = [10, 25, 50, 100]
  if re == 3450:
    train_insert_rate_set = [15]#, 38, 75, 150]
  if re == 4600:
    train_insert_rate_set = [20, 50, 100, 200]

  for dp in train_dp_set:
    for ir in train_insert_rate_set:
      label_collection.append(np.load(filepath))
      print("Insert Rate: ", ir, " Finished.")
    print("Particle Diameter : ", dp, " Finished.")
  print("Re ",re," Finished.")

# Divide labels into training and test set
train_data_label, test_data_label  = data_collect(label_collection ,train_end_frame, test_end_frame)
print("Check training data label: ",train_data_label.shape)
print("Check test data label: ",test_data_label.shape)

##---------------- Read all lab data---------------##
#lab_data = np.load(drive_path+"/lab_data/lab_data_particle_positions.npy")

# Generate index for ShuffleSplit
train_data_label_index = np.arange(0,train_data_label.shape[0],1)
test_data_label_index = np.arange(0,test_data_label.shape[0],1)

train_data_features = norm201(train_data_features)
test_data_features = norm201(test_data_features)
#lab_data = norm201(lab_data)


##----------------  Data division ---------------##
shuffler = ShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split(train_data_features, train_data_label_index)
indices = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]
X_train, y_train = torch.from_numpy(train_data_features[indices[0]]), torch.from_numpy(train_data_label[indices[0]])
X_val, y_val = torch.from_numpy(train_data_features[indices[1]]), torch.from_numpy(train_data_label[indices[1]])
X_test, y_test = torch.from_numpy(test_data_features), torch.from_numpy(test_data_label)


print("View the shape of train set and validation set: ")
print("train", X_train.shape)
print("val", X_val.shape)
print("test", X_test.shape)


#----------- Label data ----------##
X_train = X_train.float()
X_val = X_val.float()
X_test = X_test.float()
norm_factor = 20
y_train = y_train * norm_factor
y_val = y_val * norm_factor
y_test = y_test * norm_factor

#------- Construct our own flow dataset ---------#
train_data = FlowDataset(X_train, y_train.float())
validate_data = FlowDataset(X_val, y_val.float())
test_data = FlowDataset(X_test, y_test.float())



def train_model(model, weight_decay=1e-3, random_seed=False):
  """The train function which takes the weight-decay as the argument """
  if not random_seed:
      set_seed(seed)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=weight_decay )
  criterion_train = CE
  criterion_validate = upRMSE

  if device == 'cpu':
      num_workers = 0
  else:
      num_workers = 4

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  #validation_loader = DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
  validation_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
  #test_loader = DataLoader(test_data , batch_size=test_batch_size, shuffle=False, num_workers=4)
  test_loader = DataLoader(validate_data , batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

  for epoch in range(n_epochs):
      train_loss, train_loss_rmse = train(model, optimizer, criterion_train, train_loader, device=device)
      validation_loss, validation_loss_rmse = validate(model, criterion_validate, validation_loader,device=device)
      if epoch==0:
          print("  ")
          print("  ")
          print("Welcome! ")
          print("If you see this line, that means the training process runs successfully!")
          print("You have installed the machine learning part  successfully.")
          print(" ")
      
      print("Epoach: ", epoch)
      print("Training loss: ", train_loss_rmse.item())
      print("Validation loss: ", validation_loss_rmse.item() )
      print("_________________________________________")

  test_loss, test_loss_rmse = validate(model, criterion_validate, test_loader)
  print("Avg. Test Loss: %1.3f" % test_loss, " Avg. RMSE Loss: %1.3f" % test_loss_rmse)
  print("")

  ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
  model_save_name = 'ParticleNet' + st + '.pt'
  PATH = "../model_trained/{model_save_name}"

  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, PATH)

  return model


model_trained = train_model(model_input, random_seed=random_seed)
