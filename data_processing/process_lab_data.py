from PIL import Image
import glob
from scipy import ndimage
import time
import datetime
import os
import numpy as np


def read_lab_data(filename, gray=True):
    im = Image.open(filename)
    imarray = np.array(im)
    im_gray_array = rgb2gray(imarray)
    if gray:
        return im_gray_array
    else:
        return imarray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# Calculate the start frame and end frame
path = '../data/lab_data/undis_PIV_2/'
name_list = sorted(glob.glob(path+'ImageUndistorted_*.jpg'))
start_frame = ''
end_frame = ''
for sta_num_str, end_num_str  in zip(re.findall("\d",name_list[0])[1:], re.findall("\d",name_list[-1])[1:]):
    start_frame += sta_num_str
    end_frame += end_num_str
start_frame = int(start_frame)
end_frame = int(end_frame)
print(start_frame,' to ', end_frame)


# Initialize the holder
pair = []
particle_position = []
interval = 5
crop_size = (300, 900, 1160,1520)
sample_number = int((end_frame-start_frame+1)/interval)
print("Generate %d frame pairs." % sample_number)

raw = False
if raw:
    raw_str = 'raw'
else:
    raw_str = 'downsample'

stopwatch = 0
for i in range(sample_number):
    file_name = path + "ImageUndistorted_%d.jpg"%(start_frame+i*interval)
    data = read_lab_data(file_name)
    if raw:
        img = data[crop_size[0]:crop_size[1],crop_size[2]:crop_size[3]].T
        
    else:
        img = data[crop_size[0]:crop_size[1],crop_size[2]:crop_size[3]].T
        img_small = ndimage.interpolation.zoom(img,.5)
    
    pair.append(img_small)
    if stopwatch == 1:
        particle_position.append(np.array(pair))
        pair.pop(0)
        stopwatch = 0
    stopwatch += 1
    print("Finished ",i, " frame.") # The frame size is ", particle_position[i].shape)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
output_path = "./data/lab_data_"+ raw_str +'_'+st
os.mkdir(output_path)
np.save(output_path + "/lab_data_particle_positions", np.array(particle_position))
with open(output_path+"/Parameters.txt",'w') as file:
    file.write("Dataset size: "+ str(np.array(particle_position).shape)+"\n")