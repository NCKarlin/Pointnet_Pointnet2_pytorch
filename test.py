#%% Write Python 3 code in this online editor and run it.
import numpy as np
import torch
from data_utils.S3DISDataLoader import FracDataset
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join("data", "testdata", "")

print("Start loading training data ...")
TRAIN_DATASET = FracDataset(data_root=DATA_ROOT, 
                            split='train', 
                            num_point=4096, 
                            block_size=4.0, 
                            sample_rate=1.0, 
                            transform=None)
print("Start loading test data ...")
TEST_DATASET = FracDataset(data_root=DATA_ROOT, 
                            split='overtrain-test', 
                            num_point=4096,
                            block_size=4.0,
                            sample_rate=1.0, 
                            transform=None)

# Creating DataLoaders
trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=8, shuffle=True, num_workers=0,
                                                pin_memory=True, drop_last=True,
                                                worker_init_fn=None)
testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=8, shuffle=False, num_workers=0,
                                                pin_memory=True, drop_last=True)

#%%
# 
# print(trainDataLoader[0])
# data = np.load('data\\testdata\\data1_norot.npy')
data = np.load('data\\testdata\\Area_1_conferenceRoom_1.npy')
print(data.shape)

#%%

current_points, current_labels = next(iter(trainDataLoader))
print(current_points.shape)
print(current_labels.shape)

# print(torch.max(current_points, 0))
# print(torch.min(current_points, 0))

# print(current_points.max(dim=2))
# print(current_points.min(dim=2))

print(torch.amax(current_points, dim=(0, 1)))
print(torch.amin(current_points, dim=(0, 1)))