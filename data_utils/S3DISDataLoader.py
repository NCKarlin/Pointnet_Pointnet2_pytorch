import os
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from models.pointnet2_utils import pc_normalize
import random

class FracDataset(Dataset):
    def __init__(self, 
                 data_root,
                 split='train', 
                 num_point=4096, 
                 block_size=1.0, 
                 sample_rate=1.0,
                 transform=None):
        super().__init__()
        
        # Setting dataset attributes from Initializationc all
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        #taking files meant for training or testing depending on split argument (checking for 'train' or 'test' in file name)
        files = sorted(os.listdir(data_root)) #list containing the filenames of the scans in the dataset
        if split == 'train':
            #list of filenames to use
            files_split = [file for file in files if not 'test' in file]
        else:
            files_split = [file for file in files if 'test' in file] 
            #files_split: list containing the filenames of the respective files in training or test data
            #TODO: Check format of files_split and what it contains

        # Placeholders for aggregate data from all files combined
        self.file_points, self.file_labels = [], []
        self.file_coord_min, self.file_coord_max = [], []
        num_point_all = [] #list of the number of points in each file
        self.block_center_coordinates = [] #list for block center coords
        self.num_blocks = [] #list for amount of blocks in each file
        self.block_x_len = [] #list for horizontal block length
        self.block_y_len = [] #list for vertical block length
        self.block_mins = [] #list for all minimum points per block
        self.block_maxs =[] #list of all maximum point per block
        self.block_point_indices = [] #list of point indices for corresponding block
        labelweights = np.zeros(2) #total number of points belonging to each label

        # Looping over the individual scans within the train-test-split dataset
        #TODO: Check file_name variable
        for file_name in tqdm(files_split, total=len(files_split)):
            '''
            Within the for-loop these variables have to be checked out concerning their format and content:
            - file_path:
            - file_data:
            - points, labels:
            - tmp:
            - labelweights:
            - file_idxs:
            - block_idxs:
            
            '''
            
            # Getting specific file data from file
            file_path = os.path.join(data_root, file_name)
            file_data = np.load(file_path)  # xyzrgbl, N*7
            points, labels = file_data[:, 0:6], file_data[:, 6]  # xyzrgb, N*6; l, N
            # Currently points and labels of entire scan behind file_path

            tmp, _ = np.histogram(labels, range(3)) #list of how many points are of each label in this file [3637994, 11176]
            labelweights += tmp #adding to total over all files (same shape)
            
            #! Think about whether we actually need the coord_min and coord_max as class attributes
            #TODO: Investigate and potentially delete the max and min coordinates for the file 
            # Getting min and max of all coordinates in this file
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            
            # Calculating number of blocks for respective file
            max_num_blocks = math.floor(np.sum(labels.size) * sample_rate / num_point)
            
            # Creating spacing variables for block center generation
            blocks_x_res = math.floor(np.sqrt(max_num_blocks))
            blocks_y_res = math.floor(np.sqrt(max_num_blocks))  
            num_blocks = (blocks_x_res - 2)**2 #updating actual number of blocks 
            # With padding, not taking edge blocks into account
            blocks_center_x_coord_start = 3 / (2 * blocks_x_res) * coord_max[0]
            blocks_center_y_coord_start = 3 / (2 * blocks_y_res) * coord_max[1]
            blocks_center_x_coord_step = 2 / (2 * blocks_x_res) * coord_max[0]
            blocks_center_y_coord_step = 2 / (2 * blocks_y_res) * coord_max[1]
            
            # Creating block center coordinates
            blocks_center_x_coords = np.linspace(start=blocks_center_x_coord_start,
                                                stop=coord_max[0]-3/2*blocks_center_x_coord_step,
                                                num=blocks_x_res-2)
            blocks_center_y_coords = np.linspace(start=blocks_center_y_coord_start, 
                                                stop=coord_max[1]-3/2*blocks_center_y_coord_step,
                                                num=blocks_x_res-2)
            blocks_center_xx_coords, blocks_center_yy_coords = np.meshgrid(blocks_center_x_coords, 
                                                                           blocks_center_y_coords)
            blocks_center_coords = np.hstack([blocks_center_xx_coords.reshape((-1,1)), 
                                              blocks_center_yy_coords.reshape((-1,1))])
            
            blocks_with_too_few_points = []
            num_points_of_blocks_with_too_few = []
            block_point_idxs = []
            block_mins = [] 
            block_maxs = [] 
            # Checking number of points per block and saving block MIN and MAX
            for index, block_center in enumerate(blocks_center_coords):
                # Determine the block MIN and MAX points
                block_min = block_center - [blocks_center_x_coord_step / 2.0, blocks_center_y_coord_step / 2.0]
                block_max = block_center + [blocks_center_x_coord_step / 2.0, blocks_center_y_coord_step / 2.0]
                # Determining number of points in block
                points_in_block = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
                if points_in_block.size >= num_point:
                    selected_point_idxs = np.random.choice(points_in_block, self.num_point, replace=False).astype(int)
                    block_point_idxs.append(selected_point_idxs)
                    block_mins.append(block_min)
                    block_maxs.append(block_max)
                elif points_in_block.size >= num_point/4:
                    selected_point_idxs = np.zeros((self.num_point))
                    selected_point_idxs[:len(points_in_block)] = points_in_block.astype(int)
                    selected_point_idxs[len(points_in_block):] = np.random.choice(points_in_block,
                                                                            size=(self.num_point - len(points_in_block)), 
                                                                            replace=True)
                    block_point_idxs.append(selected_point_idxs)
                    block_mins.append(block_min)
                    block_maxs.append(block_max)
                # Not enough points in block to subsample from
                else:
                    blocks_with_too_few_points.append(index)
                    num_points_of_blocks_with_too_few.append(points_in_block.size)
                
            # Updating the blocks, so only blocks with enough points are processed
            if len(blocks_with_too_few_points) > 0:
                blocks_center_coords = np.delete(blocks_center_coords, blocks_with_too_few_points, axis=0)
                num_blocks = num_blocks - len(blocks_with_too_few_points)
                    
            # Appending this file data to combined dataset
            self.file_points.append(points)
            self.file_labels.append(labels)
            self.file_coord_min.append(coord_min)
            self.file_coord_max.append(coord_max)
            num_point_all.append(labels.size) # currently 662.525 -> num points before sub-sampling
            self.block_center_coordinates.append(blocks_center_coords)
            self.num_blocks.append(num_blocks)
            self.block_x_len.append(blocks_center_x_coord_step)
            self.block_y_len.append(blocks_center_y_coord_step)
            self.block_point_indices.append(block_point_idxs)
            self.block_mins.append(block_mins)
            self.block_maxs.append(block_maxs)


        # OWN LABELWEIGHTING
        labelweights = labelweights.astype(np.float32)
        pct_labelweights = labelweights / np.sum(labelweights)
        # w = N (# of all points) / [2 * N_j] (# of points of class j)
        self.labelweights = np.sum(labelweights) / (2 * labelweights)
        # "just" Inverse weighing of labels
        #self.labelweights = np.sum(labelweights) / labelweights
        
        # Placeholders for "double indexing"
        file_idxs = [] #creating list/array where the file_idxs points to the file
        block_idxs = [] #creating list/array where block_idx points to block within a file
        
        # Creating double indexing lists 
        for index in range(len(files_split)):
            file_idxs.extend([index] * self.num_blocks[index])
            block_idxs.extend(np.arange(start=0, stop=self.num_blocks[index]))
        
        # Conversion to numpy array for index accessing
        self.file_idxs = np.array(file_idxs) #list of indexes showing which file each block is from
        self.block_idxs = np.array(block_idxs)
        print("In total {} samples in {} set.".format(len(self.file_idxs), split))
        

    def __getitem__(self, idx):
        
        # Pulling correct data according to index
        file_idx = self.file_idxs[idx] #index of the file where the block data is from
        block_idx = self.block_idxs[idx] #index of the block within the file 
        points = self.file_points[file_idx]   # file points N * 6 [ALL POINTS CURRENTLY]
        labels = self.file_labels[file_idx]   # file labels N
        block_point_idxs = self.block_point_indices[file_idx][block_idx] #indices of points within corresponding block
        block_center_coords = self.block_center_coordinates[file_idx][block_idx]
        #TODO: think about whether these are needed or can go for clean up
        block_x_len = self.block_x_len[file_idx]
        block_y_len = self.block_y_len[file_idx]
        coord_min = self.file_coord_min[file_idx] #! Think about removing
        coord_max = self.file_coord_max[file_idx] #! Think about removing
        N_points = points.shape[0] #number of file points
        
        # Fetching corresponding block points 
        block_points = points[block_point_idxs.astype(int), :]
        
        # Creating current points array to be filled
        current_points = np.zeros((self.num_point, 9))
        
        # Fetching original coordinates for post model visualization
        current_coords = block_points[:, 0:3]
        
        # Shifting of x and y coordinate according to block center
        block_points[:, 0] = block_points[:, 0] - block_center_coords[0]
        block_points[:, 1] = block_points[:, 1] - block_center_coords[1]
        
        # Normalizing of RGB values
        block_points[:, 3:6] /= 255.0
        
        # Pulling normalizaed original color values of block points for post model viz
        current_colors = block_points[:, 3:6]
        
        # Normalizing spatial coordinates with maximum values of file
        current_points[:, 6] = block_points[:, 0] / self.file_coord_max[file_idx][0]
        current_points[:, 7] = block_points[:, 1] / self.file_coord_max[file_idx][1]
        current_points[:, 8] = block_points[:, 2] / self.file_coord_max[file_idx][2]
        
        # Coercing values
        current_points[:, 0:6] = block_points
        current_labels = labels[block_point_idxs.astype(int)]
        
        # Additional transforms if any applicable
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        
        return current_points, current_labels, current_coords, current_colors
        
        
        
        # TODO: fetch or create block range for point sampling here
        '''
        After checking the sampling around the true center, if it doesnt meet the minimum 
        points needed, we will create off centers for the block and iterate through those.
        Then point picking/ sampling.
        Then saving of the coordinates and rgb values for display and visualisation or 
        prepare and return these seperately to be saved in train_fragrec.py
        
        DO WE NEED A CHECK FOR AMOUNT OF POINTS WITHIN BLOCK?? IF YES, IT SHOULD BE DONE IN THE INNIT FUNCTION
        '''
        # Pulling center, block center and determine block MIN and MAX
        #pc_center = np.mean(points, axis=0)[:3]
        #! Not needed as of now, is done within the __init__ method
        # block_center = self.block_center_coordinates[file_idx][block_idx]
        # block_min = block_center - [block_x_len / 2.0, block_y_len / 2.0]
        # block_max = block_center + [block_x_len / 2.0, block_y_len / 2.0]
        # point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        
        # Check for minimum points, currently hard coded -> TODO: softcode the minimum points here -> ncentroids[0]
        #if point_idxs.size < 2048:
            #TODO: then skip the block?
            # # Creation of off centers as alternative centers
            # off_centers = np.array([
            #     [0.25 * (block_max[0]-block_min[0]), 0.25 * (block_max[1]-block_min[1]), block_center[2]], #left bottom -> clockwise
            #     [0.25 * (block_max[0]-block_min[0]), 0.5 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.25 * (block_max[0]-block_min[0]), 0.75 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.5 * (block_max[0]-block_min[0]), 0.75 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.75 * (block_max[0]-block_min[0]), 0.75 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.75 * (block_max[0]-block_min[0]), 0.5 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.75 * (block_max[0]-block_min[0]), 0.25 * (block_max[1]-block_min[1]), block_center[2]], 
            #     [0.5 * (block_max[0]-block_min[0]), 0.25 * (block_max[1]-block_min[1]), block_center[2]], 
            # ])
            # points_idxs_list = []
            # points_len_list = []
            # for off_center in off_centers:
            #     block_min = off_center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            #     block_max = off_center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            #     point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #     points_idxs_list.append(point_idxs)
            #     points_len_list.append(len(point_idxs)) 
        
            
        # # (Sub-)Sampling with true mean of the input point cloud
        # #! Are we always taking the same center from the entire input pc??
        # center = np.mean(points, axis=0)[:3]
        # block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        # block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        # point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        # #2048
        # # TODO: double check the point sampling mechanism (point_idxs)
        # if point_idxs.size < 2048:
        #     # Creating off centers
        #     off_centers = np.array([
        #         [0.25 * coord_max[0], 0.25 * coord_max[1], center[2]], 
        #         [0.25 * coord_max[0], 0.5 * coord_max[1], center[2]], 
        #         [0.25 * coord_max[0], 0.75 * coord_max[1], center[2]], 
        #         [0.5 * coord_max[0], 0.25 * coord_max[1], center[2]],  
        #         [0.5 * coord_max[0], 0.75 * coord_max[1], center[2]], 
        #         [0.75 * coord_max[0], 0.25 * coord_max[1], center[2]], 
        #         [0.75 * coord_max[0], 0.5 * coord_max[1], center[2]], 
        #         [0.75 * coord_max[0], 0.75 * coord_max[1], center[2]], 
        #     ])
        #     # Creation of list to pick newly sampled points from
        #     points_idxs_list = []
        #     points_len_list = []
        #     for off_center in off_centers:
        #         block_min = off_center - [self.block_size / 2.0, self.block_size / 2.0, 0]
        #         block_max = off_center + [self.block_size / 2.0, self.block_size / 2.0, 0]
        #         point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        #         points_idxs_list.append(point_idxs)
        #         points_len_list.append(len(point_idxs))
        #     # Checking for any center with more than 1024 points within the block
        #     if any(point_len>round(self.num_point/3) for point_len in points_len_list):
        #         # Grabbing center with most points
        #         points_len_list_max_idx = np.argmax(points_len_list)
        #         point_idxs = points_idxs_list[points_len_list_max_idx]
        #         center = off_centers[points_len_list_max_idx]
        #     else:
        #         # Random point selection as before for now
        #         while (True):
        #             center = points[np.random.choice(N_points)][:3] #taking a completely random point in the room as centre
        #             block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to negative side of centre
        #             block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0] #half a blocksize to positive side of centre
        #             #getting indexes of the points that are within the block xy range
        #             point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
        #             #if there are less than 1024 points in the block then choose a different center
        #             if point_idxs.size > 2048:
        #                 break

        #! This might be neded here in some way potentially
        # np.random.seed(42) # random seed for replicating random selection
        # # If more points available than need in block, sample amount randomly
        # if point_idxs.size > self.num_point:
        #     selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False).astype(int)
        #     num_duplicate_points = 0
        #     print("No duplicate points when sampling blocks, because of more available points.")
        # # Exact match of points and required points
        # elif point_idxs.size == self.num_point:
        #     selected_point_idxs = point_idxs.astype(int)
        #     num_duplicate_points = 0 
        #     print("No duplicate points when sampling blocks, because of exact match with block point amount.")
        # # Not enough points, fill the rest with random indices
        # else:
        #     selected_point_idxs = np.zeros((self.num_point))
        #     selected_point_idxs[:len(point_idxs)] = point_idxs.astype(int)
        #     selected_point_idxs[len(point_idxs):] = np.random.choice(point_idxs,
        #                                                              size=(self.num_point - len(point_idxs)), 
        #                                                              replace=True)
        #     num_duplicate_points = self.num_point - point_idxs.size
        #     print(f"Number of duplicate points: {num_duplicate_points}")
        
        # # normalize
        # #! Is this not the same as above? Should it rather be block_points?
        # selected_points = points[selected_point_idxs.astype(int), :]  # resampled points in the block: num_point * 6
        # current_points = np.zeros((self.num_point, 9))  # num_point * 9
        
        # # Pulling original coords of point sample for post model visualization
        # current_coords = selected_points[:, 0:3]
        
        # #the x and y coordinate is shifted according to the centre of the block
        # selected_points[:, 0] = selected_points[:, 0] - block_center[0]
        # selected_points[:, 1] = selected_points[:, 1] - block_center[1]

        # #normalizing the rgb values
        # selected_points[:, 3:6] /= 255.0
        
        # # Pulling normalized color values of point sample for post model visualisation
        # current_colors = selected_points[:, 3:6]

        # #the new last 3 columns are the xyz columns divided by the corresponding max file coordinate
        # current_points[:, 6] = selected_points[:, 0] / self.file_coord_max[file_idx][0]
        # current_points[:, 7] = selected_points[:, 1] / self.file_coord_max[file_idx][1]
        # current_points[:, 8] = selected_points[:, 2] / self.file_coord_max[file_idx][2]

        # #putting it all together
        # current_points[:, 0:6] = selected_points
        # current_labels = labels[selected_point_idxs.astype(int)]

        # #additional transforms if there are any
        # if self.transform is not None:
        #     current_points, current_labels = self.transform(current_points, current_labels)
        
        # return current_points, current_labels, current_coords, current_colors


    def __len__(self):
        return len(self.file_idxs)


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point: #>4096
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else: #1024-4096
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()