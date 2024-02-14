#! Virtual Environment on 10.15.130.246: dataprepvenv
# IMPORTS
#########################################################################################
import os 
import numpy as np
import math
import yaml 
import open3d as o3d
from data_prep_utils import (
    makePC, 
    get_lowest_corner_coords, 
    create_vec_distances, 
    get_orthonormal_vec_coords,
    rot_mat_from_vecs, 
    rotate_coords_onto_axis, 
    create_rgb_list, 
    lowest_corner_to_all_other_corners_vecs, 
    create_orthogonal_vector_list,
    center_pc
    )

# IMPORT OF RESPECTIVE SAMPLE
raw_input_data_path = "/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data/testdata/data_labelled_int.npy"
raw_input_pc = np.load(raw_input_data_path)


# GENERAL VARIABLES TO BE SET OR READ IN
#########################################################################################
#! Check if the right machine was selected
machine = "DigiLab"
if machine == "DigiLab":
    general_block_saving_path = "/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data"
else:
    general_block_saving_path = "/Users/nk/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data"
dataset_saving_dir_name = "samples"
percentage_boundary = 0.1
# Parsing needed hyperparameters from original config yaml file 
with open('/home/innolidix/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/conf/train/train_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)
    num_point = config_data['hyperparams']['npoint']
    block_size = config_data['hyperparams']['block_size']


# CROPPING OF RAW INPUT PC WITH BOUNDING BOX
#########################################################################################
#TODO: Test whether the cropping makes any difference or not -> Leave it out 
raw_input_pc_coords = raw_input_pc[:, 0:3]
raw_input_rgb_list = create_rgb_list(raw_input_pc)
raw_pc = makePC(raw_input_pc_coords, raw_input_rgb_list) 
cropping_bbox = raw_pc.get_oriented_bounding_box()
raw_pc_cropped = raw_pc.crop(cropping_bbox)


# PREPARATION FOR BACK ROTATION
#########################################################################################
bbox = raw_pc_cropped.get_oriented_bounding_box()
bbox_points = np.round(np.asarray(bbox.get_box_points()), 5)
lowest_corner_coords = get_lowest_corner_coords(bbox_points, axis='z')
lowest_corner_to_corner_vecs = lowest_corner_to_all_other_corners_vecs(bbox_points, 
                                                                       lowest_corner_coords)
lowest_corner_to_corner_dists = create_vec_distances(lowest_corner_to_corner_vecs)
shortest_idx = np.argmin(lowest_corner_to_corner_dists)
shortest_lowest_corner_to_corner_vector_coords = lowest_corner_to_corner_vecs[shortest_idx]
# Determine orthogonal vectors through dot product of their unit-vectors
orthogonal_vec_coords_list = create_orthogonal_vector_list(shortest_lowest_corner_to_corner_vector_coords, 
                                                            lowest_corner_to_corner_vecs)
orthogonal_vec_dists = create_vec_distances(orthogonal_vec_coords_list)
# Delete longest orthogonal vector -> orthognal but diagonal along a face 
longest_idx = np.argmax(orthogonal_vec_dists)
del orthogonal_vec_coords_list[longest_idx]
# Appending shortest lowest corner to corner vec to overall orthogonal list for inclusion
orthogonal_vec_coords_list.append(shortest_lowest_corner_to_corner_vector_coords)
orthonormal_vec_coords = np.array(orthogonal_vec_coords_list)


# FIRST BACK ROTATION
#########################################################################################
z_edge = get_orthonormal_vec_coords(orthonormal_vec_coords, 'shortest')
R_rot1 = rot_mat_from_vecs(z_edge, np.array([0, 0, 1]))
input_pc_rot1 = rotate_coords_onto_axis(raw_input_pc_coords, R_rot1)
orthonormal_vec_coords_rot1 = rotate_coords_onto_axis(orthonormal_vec_coords, R_rot1)


# SECOND BACK ROTATION
#########################################################################################
x_edge = get_orthonormal_vec_coords(orthonormal_vec_coords_rot1, 'longest')
R_rot2 = rot_mat_from_vecs(x_edge, np.array([1, 0, 0]))
input_pc_rot2 = rotate_coords_onto_axis(input_pc_rot1, R_rot2)
orthonormal_vec_coords_rot2 = rotate_coords_onto_axis(orthonormal_vec_coords_rot1, R_rot2)


# CENTERING
#########################################################################################
backrotated_centered_input_pc = center_pc(input_pc_rot2)
# Overriding original coordinates of raw input point cloud 
raw_input_pc[:, 0:3] = backrotated_centered_input_pc


# BLOCK CENTER COORDINATE CREATION
#########################################################################################
# Creation of value ranges for x and y -> x_range, y_range
x_min = np.min(backrotated_centered_input_pc[:,0])
x_max = np.max(backrotated_centered_input_pc[:,0])
y_min = np.min(backrotated_centered_input_pc[:,1])
y_max = np.max(backrotated_centered_input_pc[:,1])
x_range = np.abs(x_min) + x_max
y_range = np.abs(y_min) + y_max
# Maximum block resolution
max_blocks_x_res = math.ceil(x_range / block_size) 
max_blocks_y_res = math.ceil(y_range / block_size)
# Actual block resolution with half block padding on each side
blocks_x_res = max_blocks_x_res - 1
blocks_y_res = max_blocks_y_res - 1
# Center coordinate start in x and y direction
blocks_center_x_start = x_min + block_size
blocks_center_y_start = y_min + block_size
# Center coordinate creation
blocks_center_x_coords = np.linspace(start=blocks_center_x_start, 
                                     stop=x_max-block_size, 
                                     num=blocks_x_res)
blocks_center_y_coords = np.linspace(start=blocks_center_y_start, 
                                     stop=y_max-block_size, 
                                     num=blocks_y_res)
blocks_center_xx_coords, \
    blocks_center_yy_coords = np.meshgrid(blocks_center_x_coords, 
                                          blocks_center_y_coords)
blocks_center_zz_coords = np.zeros((blocks_center_xx_coords.shape))
blocks_center_coords = np.hstack([blocks_center_xx_coords.reshape((-1,1)), 
                                  blocks_center_yy_coords.reshape((-1,1)), 
                                  blocks_center_zz_coords.reshape((-1,1))])


# BLOCKING
#########################################################################################
# Placeholder variables creation for saving vital info
input_block_idxs = []
input_block_points = [] 
not_input_block_idxs = []
# Looping through block center coordinate list to create blocks
for index, block_center in enumerate(blocks_center_coords):
    block_min = block_center[:2] - [block_size / 2.0, block_size / 2.0]
    block_max = block_center[:2] + [block_size / 2.0, block_size / 2.0]
    # Determining indices of points within block
    points_in_block_idxs = np.where(
        (backrotated_centered_input_pc[:,0] >= block_min[0]) & \
        (backrotated_centered_input_pc[:,0] <= block_max[0]) & \
        (backrotated_centered_input_pc[:,1] >= block_min[1]) & \
        (backrotated_centered_input_pc[:,1] <= block_max[1])
        )[0]
    # FILTERING OUT BLOCK WITH TOO FEW POINTS
    if len(points_in_block_idxs) > percentage_boundary*num_point:
        input_block_idxs.append(index)
        input_block_points.append(raw_input_pc[points_in_block_idxs])
    else:
        not_input_block_idxs.append(index)
        continue


# (SUB-)SAMPLING AND SAVING BLOCKS
#########################################################################################
#TODO: Insert shuffling of samples here OR below
#TODO: Think about the furthest points sampling FPS HERE
'''
This includes the following:
1. Shuffle the list of samples
2. Create two lists containing the sample names for both train and test
3. After loading block data, check whether a part of test list
    3.1 If yes, save block to corresponding sample in test directory
    3.2 If no, save block to corresponding sample in train directory
'''
# Creating saving directory 
dataset_saving_dir = os.path.join(general_block_saving_path, dataset_saving_dir_name)
if not os.path.exists(dataset_saving_dir):
    os.mkdir(dataset_saving_dir)
    
# Looping through the included blocks and saving them as new files
for i in range(len(input_block_idxs)):
    # Pull corresponding sample idx 
    
    # Pulling correct information
    block_i_data = input_block_points[i]
    # Creating point indices from pulled point data
    block_i_point_idxs = np.arange(len(block_i_data))
    # Sub-sampling
    if len(block_i_point_idxs) >= num_point:
        sampled_block_i_point_idxs = np.random.choice(block_i_point_idxs, 
                                                      num_point, 
                                                      replace=False)
    else:
        sampled_block_i_point_idxs = list(block_i_point_idxs)
        sampled_block_i_point_idxs.extend(np.random.choice(sampled_block_i_point_idxs, 
                                                           num_point-len(sampled_block_i_point_idxs), 
                                                           replace=True))
    # Selecting sampled points
    sampled_block_i_data = block_i_data[sampled_block_i_point_idxs, :]
    # Creating block saving directory path
    #TODO: Think about training and testing classification, if statement for saving here
    block_saving_dir_path = os.path.join(dataset_saving_dir, 'train', f'sample_0')
    if not os.path.exists(block_saving_dir_path):
        os.makedirs(block_saving_dir_path)
    np.save(os.path.join(block_saving_dir_path, f'block_{i}.npy'), sampled_block_i_data)

#TODO: Insert the reordering of samples into a train and test directory here
'''
This includes the following:
- Shuffle the list of samples names
- Create the sample name split for train and test
- Create train and test directory
- Move the samples accordingly
- Re-number/-name the samples chronologically in the respective train or test folder 
'''