'''
UTILITY FUNCTIONS FOR DATA PREPARATION

Author: Niklas Karlin
Date: 01-2024

This notebook contains all the utility functions that are needed to carry out the data
preparation. 
This means, that a marked sample is taken as input, and it will then rotate back the 
input to a uniform position, block with padding, and after weeding out unwanted blocks
it will then create and save each block as an individual file.
'''
import os 
import numpy as np
import math
import open3d as o3d

# Create RGB list for given input pc
def create_rgb_list(input_pc):
    '''
    Function to create RGB list for PC creation and display. 
    
    Input:
    - input_pc: total information of input points [num_points x 7]
    
    Output:
    - rgb_list: list of normalized colors corresponding to input pc points [num_points x 3]
    '''
    # Assigning color channels
    red_c = input_pc[:, 3]
    green_c = input_pc[:, 4]
    blue_c = input_pc[:, 5]
    # Normalizing color range to [0, 1]
    red_c = (red_c - np.min(red_c)) / (np.max(red_c) - np.min(red_c))
    green_c = (green_c - np.min(green_c)) / (np.max(green_c) - np.min(green_c))
    blue_c = (blue_c - np.min(blue_c)) / (np.max(blue_c) - np.min(blue_c))
    # Stacking to create entire rgb list
    rgb_list = np.stack((red_c, green_c, blue_c), axis=1)
    return rgb_list

# Creating/ Making the point cloud
def makePC(point_data, color_data=np.array([])):
    pcd = o3d.geometry.PointCloud() #Create PC object
    pcd.points = o3d.utility.Vector3dVector(point_data) #Give coordinates
    #Coloring the PC
    if len(color_data) == 0:
        pcd.paint_uniform_color([1, 0, 0])
    else:
        pcd.colors = o3d.utility.Vector3dVector(color_data)
    return pcd

# Retrieve lowest corner of the cuboid/ bounding box
def get_lowest_corner_coords(cuboid_coords, axis):
    ''' 
    inputs:
    - cuboid_coords: 8x3-array of cuboid vertice coordinates
    - axis: string determining in which direction one wants to retrieve the "lowest corner"
    
    return:
    - lowest_corner_coords: 1x3-array with the coordinates of the "lowest" corner in the specified direction
    '''
    if axis == 'z':
        z_val_lowest_corner = np.min(cuboid_coords[:,2])
        # Correction in the case there are several corners with the same value
        if len(np.where(cuboid_coords == z_val_lowest_corner)) == 1:
            lowest_corner_idx = int(np.where(cuboid_coords == z_val_lowest_corner)[0])
        else:
            lowest_corner_idx = int(np.where(cuboid_coords == z_val_lowest_corner)[0][0])
        lowest_corner_coords = cuboid_coords[lowest_corner_idx]
    elif axis == 'y':
        y_val_lowest_corner = np.min(cuboid_coords[:,1])
        # Correction in the case there are several corners with the same value
        if len(np.where(cuboid_coords == y_val_lowest_corner)) == 1:
            lowest_corner_idx = int(np.where(cuboid_coords == y_val_lowest_corner)[0])
        else:
            lowest_corner_idx = int(np.where(cuboid_coords == y_val_lowest_corner)[0][0])
        lowest_corner_coords = cuboid_coords[lowest_corner_idx]
    elif axis == 'x':
        x_val_lowest_corner = np.min(cuboid_coords[:,0])
        # Correction in the case there are several corners with the same value
        if len(np.where(cuboid_coords == x_val_lowest_corner)) == 1:
            lowest_corner_idx = int(np.where(cuboid_coords == x_val_lowest_corner)[0])
        else:
            lowest_corner_idx = int(np.where(cuboid_coords == x_val_lowest_corner)[0][0])
        lowest_corner_coords = cuboid_coords[lowest_corner_idx]
    return lowest_corner_coords

# Determine lowest corner to all other corner vectors of bounding box points
def lowest_corner_to_all_other_corners_vecs(bbox_points, lowest_corner_coords):
    '''
    Function to create all 7 vectors from the lowest corner to all others.
    
    Input:
    - bbox_points: coordinates of all 8 corner points of the bounding box [8 x 3]
    - lowest_corner_coords: coordinates of the lowest corner point [1 x 3]
    
    Output:
    - lowest_corner_to_corner_vectors: array with all vectors from the lowest corner to 
    all others [7 x 3]
    '''
    lowest_corner_to_corner_vectors_list = []
    for i in range(len(bbox_points)):
        vec_i = bbox_points[i] - lowest_corner_coords
        # Skipping vec to same corner aas lowest
        if (vec_i == np.array([0, 0, 0])).all():
            continue
        lowest_corner_to_corner_vectors_list.append(vec_i)
    return np.array(lowest_corner_to_corner_vectors_list)

# Return othogonal vectors from vector list
def create_orthogonal_vector_list(shortest_lowest_corner_to_corner_vector_coords, 
                                  lowest_corner_to_corner_vecs):
    '''
    Function to determine orthogonal vectors from vector list. 
    
    Input: 
    - shortest_lowest_corner_to_corner_vector_coords: self-explanatory [1 x 3]
    - lowest_corner_to_corner_vecs: self-explanatory [7 x 3]
    
    Output:
    - orthogonal_vec_coords_list: list of all vectors, that are orthogonal to shortest
    corner to corner vector
    '''
    orthogonal_vec_coords_list = []
    for i in range(len(lowest_corner_to_corner_vecs)):
        dot_product_result = np.dot(
            unit_vector(shortest_lowest_corner_to_corner_vector_coords), 
            unit_vector(lowest_corner_to_corner_vecs[i])
        )
        if np.round(dot_product_result, 2) <= 0.00001:
            orthogonal_vec_coords_list.append(lowest_corner_to_corner_vecs[i])
    return orthogonal_vec_coords_list

# Create norm of passed vector
def norm_vec(vector):
    '''
    input:
    - vector: 1x3-array of vector to be normed
    
    return:
    - vector norm
    '''
    return np.linalg.norm(vector)

# Creating unit vector for input vector
def unit_vector(vector):
    '''
    input:
    - vector: 1x3-array of vector to be unitized
    
    output:
    - 1x3-array of unit vector from passed vector
    '''
    return vector / np.linalg.norm(vector)

# Create vector array of start and end point array
def create_vectors(start_end_point_array):
    '''
    This function returns an array of 3D vectors from a start and end point array.
    
    input:
    - start_end_point_array: array of starting and end points of vectors
    
    return:
    - vec_return: array of 3D vectors
    '''
    # Creation of empty vector list to be filled
    vec_return = np.zeros((len(start_end_point_array), 3))
    for i in range(len(start_end_point_array)):
        vec_i = start_end_point_array[i][1] - start_end_point_array[i][0]
        vec_return[i] = vec_i
    return vec_return

# Create ascendingly sorted vector distances array
def create_vec_distances(vectors):
    '''
    input:
    - vectors: coordinates of vectors whose distances shall be calculated
    
    return:
    - vector_dists: vector distances
    '''
    vector_dists_list = []
    for i in range(len(vectors)):
        vector_i_dist = norm_vec(vectors[i])
        vector_dists_list.append(vector_i_dist)
    vector_dists = np.array(vector_dists_list)
    return vector_dists

def get_orthonormal_vec_coords(orthonnormal_vec_coords, which):
    '''
    inputs:
    - orthogonal_vec_coords: coordinates of the 3 orthonormal vectors
    - which: string-argument specifying which of the orthonormal vector coordinates to be retrieved
        - longest: x-edge of the bounding box
        - middle: y-edge of the bounding box
        - shortest: z-edge of the bounding box
    
    return:
    - dist_sorted_orthogonal_vec_coords[i]: 1x3-array of corresponding orthonormal vector
    '''
    orthogonal_vec_dists = create_vec_distances(orthonnormal_vec_coords)
    # Sort coordinate list as the distance list 
    dist_sorted_indices = orthogonal_vec_dists.argsort()
    dist_sorted_orthognal_vec_coords = orthonnormal_vec_coords[dist_sorted_indices]
    if which == 'longest':
        return dist_sorted_orthognal_vec_coords[2]
    elif which == 'middle':
        return dist_sorted_orthognal_vec_coords[1]
    elif which == 'shortest':
        return dist_sorted_orthognal_vec_coords[0]
    else:
        print('Argument -which- was specified incorrectly, please correct.')

# Find R for aligning vec1 with vec2
def rot_mat_from_vecs(vec1, vec2):
    '''
    Find rotation matrix aligning vec1 with vec 2 (based in Rodrigues' rotation formula)
    
    inputs:
    - vec1: vector to be aligned | 1x3-array
    - vec2: vector to be aligned with | 1x3-array
    
    return:
    - rot_mat: rotation matrix rotating vec1 onto vec2
    '''
    vec1_uv, vec2_uv = unit_vector(vec1).reshape(3), unit_vector(vec2).reshape(3)
    v = np.cross(vec1_uv, vec2_uv)
    c = np.dot(vec1_uv, vec2_uv)
    s = norm_vec(v)
    kmat = np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * ((1-c) / (s**2))
    return rot_mat

# Rotate coordinates with rotation amtrix for axis alignment
def rotate_coords_onto_axis(coords, R):
    ''' 
    Function that rotates the given points by the given rotation matrix.
    
    input:
    - coords: coordinates of points to be rotated
    - R: rotation amtrix by which the points should be rotated
    
    return:
    coords_rot: rotated point coordinates
    '''
    # Depending on shape of coordinates passed rotation multiplication adjustment
    if coords.shape == (3,):
        coords_rot = R.dot(coords)
    coords_rot_list = []
    for i in range(len(coords)):
        coords_i_rot = R.dot(coords[i])
        coords_rot_list.append(coords_i_rot)
    coords_rot = np.array(coords_rot_list)
    return coords_rot