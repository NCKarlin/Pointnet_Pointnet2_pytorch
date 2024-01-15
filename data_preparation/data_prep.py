'''
DATA PREPARATION SCRIPT

Author: Niklas Karlin
Date: 01-2024

This script will contain the actual data preparation for the model. It will take a marked
point cloud as input and subsequently perform the following steps on it:
1. Back rotation and centering for uniform layout of sample
2. Determination of block centers for block creation
3. Determination of valuable blocks for the model
4. Block creation of the sample
5. Saving of block points as separate files 
'''


###### IMPORTS
# GENERAL IMPORTS
import os 
import numpy as np
import math
import open3d as o3d
# FUNCTION IMPORT
from data_prep_utils import (
    makePC, 
    get_lowest_corner_coords, 
    create_vec_distances, 
    get_orthonormal_vec_coords,
    rot_mat_from_vecs, 
    rotate_coords_onto_axis, 
    create_rgb_list, 
    lowest_corner_to_all_other_corners_vecs, 
    create_orthogonal_vector_list
    )
# IMPORT OF RESPECTIVE SAMPLE
raw_input_data_path = "/Users/nk/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data/testdata/data_labelled_int.npy"
raw_input_pc = np.load(raw_input_data_path)


###### GENERAL VARIABLES TO BE SET
#! Think about adjusting it to reading out from the same config YAML file
block_size = 30.0
num_point = 10240
percentage_boundary = 0.1
general_block_saving_path = "/Users/nk/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data"
dataset_saving_dir_name = "test_dataset_test1"


###### CROPPING OF RAW INPUT PC WITH BOUNDING BOX
#TODO: Test wehther the cropping makes any difference or not -> Leave it out 
# Pulling raw input point coordinates
raw_input_pc_coords = raw_input_pc[:, 0:3]
# Creating RGB list for raw input points
raw_input_rgb_list = create_rgb_list(raw_input_pc)
# Creating point cloud
raw_pc = makePC(raw_input_pc_coords, raw_input_rgb_list)
# Cropping raw input pc with oriented bounding box 
cropping_bbox = raw_pc.get_oriented_bounding_box()
raw_pc_cropped = raw_pc.crop(cropping_bbox)


###### PREPARATION FOR BACK ROTATION
# Pulling bounding box of cropped point cloud
bbox = raw_pc_cropped.get_oriented_bounding_box()
# Pulling bounding box corner points
bbox_points = np.round(np.asarray(bbox.get_box_points()), 5)
# Retrieving coordinates of lowest corner in z-direction
lowest_corner_coords = get_lowest_corner_coords(bbox_points, axis='z')
# Construction of all vectors from the lowest corner to all others
lowest_corner_to_corner_vecs = lowest_corner_to_all_other_corners_vecs(bbox_points, lowest_corner_coords)
# Determination of distances to choose shortest to be aligned with z-axis
lowest_corner_to_corner_dists = create_vec_distances(lowest_corner_to_corner_vecs)
shortest_idx = np.argmin(lowest_corner_to_corner_dists)
shortest_lowest_corner_to_corner_vector_coords = lowest_corner_to_corner_vecs[shortest_idx]
# Calculating dot-product to determine orthonormal vectors 
orthogonal_vec_coords_list = create_orthogonal_vector_list(shortest_lowest_corner_to_corner_vector_coords, 
                                                            lowest_corner_to_corner_vecs)
# Calculate all their distances
orthogonal_vec_dists = create_vec_distances(orthogonal_vec_coords_list)
# Delete longest orthogonal vector
longest_idx = np.argmax(orthogonal_vec_dists)
del orthogonal_vec_coords_list[longest_idx]
# Appending shortest lowest corner to corner vec to overall orthogonal list for inclusion
orthogonal_vec_coords_list.append(shortest_lowest_corner_to_corner_vector_coords)
orthonormal_vec_coords = np.array(orthogonal_vec_coords_list)


###### FIRST BACK ROTATION
# Pulling coordinates of shortest orthonormal vector (z-edge)
z_edge = get_orthonormal_vec_coords(orthonormal_vec_coords, 'shortest')
# Retrieving rotation matrix for alignment of z-edge with z-axis
R_rot1 = rot_mat_from_vecs(z_edge, np.array([0, 0, 1]))
# Point rotation
input_pc_rot1 = rotate_coords_onto_axis(raw_input_pc_coords, R_rot1)
# Rotation of orthonormal vectors
orthonormal_vec_coords_rot1 = rotate_coords_onto_axis(orthonormal_vec_coords, R_rot1)


###### SECOND BACK ROTATION
# Pulling coordinates of the longest orthonormal vector (x-edge)
x_edge = get_orthonormal_vec_coords(orthonormal_vec_coords_rot1, 'longest')
# Retrieving rotation matrix for alignment of x-edge with x-axis
R_rot2 = rot_mat_from_vecs(x_edge, np.array([1, 0, 0]))
# Point rotation
input_pc_rot2 = rotate_coords_onto_axis(input_pc_rot1, R_rot2)
# Rotation of orthonormal vectors
orthonormal_vec_coords_rot2 = rotate_coords_onto_axis(orthonormal_vec_coords_rot1, R_rot2)


###### CENTERING
# Center definition of rotated point cloud
pc_center = np.mean(input_pc_rot2, axis=0)
# Centering of backrotated point cloud
backrotated_centered_input_pc = input_pc_rot2 - pc_center
# Setting overall dataset coordinates to backrotated and centered coordinates
raw_input_pc[:, 0:3] = backrotated_centered_input_pc



###### BLOCK CENTER COORDINATE CREATION
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


###### BLOCKING
# Placeholder variables creation for saving vital info
input_block_idxs = []
input_block_points = [] 
not_input_block_idxs = []
# Looping through block center coordinate listto create blocks
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
    #TODO: Read in from YAML and set consistent value
    if len(points_in_block_idxs) > percentage_boundary*num_point:
        input_block_idxs.append(index)
        input_block_points.append(raw_input_pc[points_in_block_idxs])
    else:
        not_input_block_idxs.append(index)
        continue


###### SAVING CORRECT BLOCKS
# Creating saving directory 
dataset_saving_dir = os.path.join(general_block_saving_path, dataset_saving_dir_name)
if not os.path.exists(dataset_saving_dir):
    os.mkdir(dataset_saving_dir)
# Looping through the included blocks and saving them as new files
for i in range(len(input_block_idxs)):
    # Pulling correct information
    block_i_points = input_block_points[i]
    # Creating block saving directory path
    #TODO: Think about training and testing classification
    block_saving_dir_path = os.path.join(dataset_saving_dir, f'sample_1_block_{i}')
    np.save(block_saving_dir_path, block_i_points)