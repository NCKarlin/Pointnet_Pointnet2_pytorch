import argparse
import os
from data_utils.S3DISDataLoader import FracDataset, S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

def worker_init(x):
    return np.random.seed(x + int(time.time()))

def main():
    # root = 'data/testdata/'
    root = 'data/s3dis/stanford_indoor3d/'
    NUM_CLASSES = 2
    NUM_POINT = 4096
    BATCH_SIZE = 16

    #####CHECKING WHAT THE DATA LOOKS LIKE############

    example_data = np.load('data/s3dis/stanford_indoor3d/Area_1_conferenceRoom_1.npy')
    print(example_data.shape)
    print(example_data[0, :])
    # print(np.min(example_data[:, 3:6]))
    # print(np.max(example_data[:, 3:6]))

    our_data = np.load('data/testdata/data_labelled_grid.npy')
    print(our_data.shape) #662525
    print(our_data[0, :])

    ###############OUR DATASET###############################################
    # print("start loading training data ...")
    # TRAIN_DATASET = FracDataset(data_root=root, split='train', num_point=NUM_POINT, block_size=4.0, sample_rate=1.0, transform=None)

    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    #                                                 pin_memory=True, drop_last=True,
    #                                                 worker_init_fn = None)

    ###############ORIGINAL DATASET##########################################
    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=5, block_size=1.0, sample_rate=1.0, transform=None)
    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True)
                                                #   worker_init_fn=worker_init,
                                                #   persistent_workers=True)

    print("The number of training data is: %d" % len(TRAIN_DATASET))

    ###############TESTING DATALOADER ######################################

    # dataloader_iter1 = iter(trainDataLoader)
    # inputs, label = next(dataloader_iter1)
    # print(inputs)
    # print(label)

    for i, (points, target) in enumerate(trainDataLoader):
        points = points.data.numpy()

        print("Points size", points.shape)
        print("Target size", target.size())

        if i == 0:
            break


if __name__ == '__main__':
    main()
