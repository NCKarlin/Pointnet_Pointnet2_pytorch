'''
FRACTURE RECOGNITION DATALOADER

Author: Niklas Karlin
Date: 01-2024

In this script the dataset class for the fracture recognition model will be defined.
'''

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


###### DATASET CREATION CLASS FOR FRACREC MODEL
class FracDataset(Dataset):
    def __init__(
        self, 
        data_root, 
        split='train', 
        num_point=4096, 
        block_size=1.0,
        transform=None
    ):
        super().__init__()
        
        # Setting hyperparameters as dataset attributes
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        
        # Fetching entire sample directory and classify into training and test
        if split == 'train':
            sample_split_dir = os.path.join(data_root, 'train')
            sample_split = [sample for sample in sample_split_dir]
        else:
            sample_split_dir = os.path.join(data_root, 'test')
            sample_split = [sample for sample in sample_split_dir]
        
        # Placeholders for aggregate data from all samples in the dataset
        self.sample_points, self.sample_labels = [], []
        self.sample_coord_min, self.sample_coord_max = [], []
        sample_num_points, sample_num_blocks = [], []
        labelweights = np.zeros(2)
        
        # Iterate through the samples to combine data into a single dataset
        for sample_name in tqdm(sample_split, total=len(sample_split)):
            
            # Loading in all the data from a single sample
            sample_path = os.path.join(sample_split_dir, sample_name)
            # Determining the amount of blocks within that sample
            num_blocks_in_sample = len([block_name for block_name in os.listdir(sample_path)])
            sample_num_blocks.append(num_blocks_in_sample)
            # Placeholders for block data for entire sample
            sample_i_points, sample_i_labels = [], []
            # Looping through blocks of a sample
            for block_idx in range(num_blocks_in_sample):
                # Creating the specific file path
                block_path = os.path.join(sample_path, f"block_{block_idx}")
                # Loading block data
                block_data = np.load(block_path)
                # Splitting into point data and labels
                block_points, block_labels = block_data[:, 0:6], block_data[:, 6]
                # Appending/ Extending placeholder variables
                sample_i_points.extend(block_points)
                sample_i_labels.extend(block_labels)
            
            # Creation of labelweights for each sample
            tmp, _ = np.histogram(sample_i_labels, range(3))
            labelweights += tmp
            # Determining minimum and maximum coordinates
            coord_min = np.amin(sample_i_points, axis=0)[:3]
            coord_max = np.amax(sample_i_points, axis=0)[:3]
            # Appending info to sample placeholder variables
            self.sample_points.append(sample_i_points)
            self.sample_labels.append(sample_i_labels)
            self.sample_coord_min.append(coord_min)
            self.sample_coord_max.append(coord_max)
            sample_num_points.append(sample_i_labels.size)
        
        # Labelweights over entire dataset
        labelweights = labelweights.astype(np.float32)
        # w = N {# of all points} / (2 * N_j) {2 * number of points in class j}
        self.labelweights = np.sum(labelweights) / (2* labelweights)
        
        # CREATION OF BLOCK IDX LIST SHOCASING BELONGING TO SAMPLE
        block_sample_idxs = [] 
        for i in len(sample_num_blocks):
            block_sample_idxs.extend([i] * int(sample_num_blocks[i]))
        self.block_sample_idxs = np.array(block_sample_idxs)
        
        # Print statement
        print("Dataset info:")
        print(f"In total there are {len(sample_split)} samples in {split} set.")
        print(f"Consisting of in total {np.sum(sample_num_blocks)} blocks.")
    
    
    def __getitem__(self, idx):
        # Get needed indices and info for fetching the correct block
        
        # Normalize for safety RGB values in 255.0 range
        
        # Normalizing the xyz columns? Or should this also be done in advance?
        
        # Apply transform given
        
        # Return requested points and labels
        print('Niklas')
    
    
    def __len__(self):
        # Return the length of the created sample_idxs list/ array
        return len(self.sample_idxs)
        
        