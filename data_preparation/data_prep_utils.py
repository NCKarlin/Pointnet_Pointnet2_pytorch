import os 
import numpy as np
import math
import open3d as o3d


def create_rgb_list(input_pc):
    '''Function to create RGB-list for point cloud creation and display
    
    The RGB-list has to be handed over individually from the point coordinates for the 
    visualization of a point cloud, hence this function for its individual creation. 
    N: number of points | F: number of features per point
    
    :param input_pc: input point cloud data with RGB-info in columns 3, 4, 5
    :type input_pc: numpy.ndarray [N, F]
    :return rgb_list: list of normalized colors corresponding to input pc points
    :rtype: numpy.ndarray [N, 3]
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


def makePC(point_data, color_data=np.array([])):
    '''Function to create an open3d point cloud object from point cloud data
    
    N: number of points | F: number of features per point
    
    :param point_data: array containing coordinates and RGB data of all points of pc 
    :type point_data: numpy.ndarray [N, F]
    :return pcd: open3d point cloud data object from point_data
    :rtype: open3d.geometry.PointCloud
    '''
    pcd = o3d.geometry.PointCloud() # Create PC object
    pcd.points = o3d.utility.Vector3dVector(point_data) # Give coordinates
    # Coloring the PC
    if len(color_data) == 0:
        pcd.paint_uniform_color([1, 0, 0])
    else:
        pcd.colors = o3d.utility.Vector3dVector(color_data)
    return pcd


def get_lowest_corner_coords(cuboid_coords, axis):
    ''' Retrieving coordinates of lowest cuboid corner in respective direction
    
    :param cuboid_coords: array of coordinates of cuboid corners
    :type cuboid_coords: numpy.ndarray [8, 3]
    :param axis: string of specified axis/ direction for minimum point value
    :type axis: string
    :return lowest_corner_coords: xyz-coordinates of lowest corner in specified direction
    :rtype: numpy.ndarray [1, 3]
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


def lowest_corner_to_all_other_corners_vecs(bbox_points, lowest_corner_coords):
    '''Function to ceate all 7 vectors from the lowest corner to all other cuboid-corners
    
    :param bbox_points: array of coords of all 8 cuboid corner points [8, 3]
    :type bbox_points: numpy.ndarray [8, 3]
    :param lowest_corner_coords: coords of chosen lowest corner of cuboid coords [1, 3]
    :type lowest_corner_coords: numpy.ndarray [1, 3]
    :return lowest_corner_to_corner_vectors_list: array of all vectors from lowest corner
    to all others
    :rtype: numpy.ndarray [7, 3]
    '''
    lowest_corner_to_corner_vectors_list = []
    for i in range(len(bbox_points)):
        vec_i = bbox_points[i] - lowest_corner_coords
        # Skipping vec to same corner as lowest 
        if (vec_i == np.array([0, 0, 0])).all():
            continue
        lowest_corner_to_corner_vectors_list.append(vec_i)
    return np.array(lowest_corner_to_corner_vectors_list)


def create_orthogonal_vector_list(shortest_lowest_corner_to_corner_vector_coords, 
                                  lowest_corner_to_corner_vecs):
    ''' Function to determine orthogonal corner vectors
    
    Given the vectors from the lowest corner to all other corners, as well as the shortest
    lowest corner to corner vectors, this function determines the other two orthogonal 
    vectors, which meet at the lowest corner of the cuboid. 
    
    :param shortest_lowest_corner_to_corner_vector_coords: see name
    :type shortest_lowest_corner_to_corner_vector_coords: numpy.ndarray [1, 3]
    :param lowest_corner_to_corner_vecs: see name 
    :type lowest_corner_to_corner_vecs: numpy.ndarray [7, 3]
    :return orthogonal_vec_coords_list: list of 3 orthogonal vectors at lowest corner
    :rtype: class 'list'
    '''
    orthogonal_vec_coords_list = []
    for i in range(len(lowest_corner_to_corner_vecs)):
        # Orthogonality check through dot product of unit vectors 
        dot_product_result = np.dot(
            unit_vector(shortest_lowest_corner_to_corner_vector_coords), 
            unit_vector(lowest_corner_to_corner_vecs[i])
        )
        if np.round(dot_product_result, 2) <= 0.00001: # numerical instability
            orthogonal_vec_coords_list.append(lowest_corner_to_corner_vecs[i])
    return orthogonal_vec_coords_list


def norm_vec(vector):
    ''' Normalize given vector, this function was mainly crated for readability
    
    :param vector: any 3D-vector/ coords combination
    :type vector: numpy.ndarray [1, 3]
    :return: vector norm
    :rtype: float
    '''
    return np.linalg.norm(vector)


def unit_vector(vector):
    ''' Create unit-vector for given vector, was mainly created for readability
    
    :param vector: 3D-vector / coords combination
    :type vector: numpy.ndarray [1, 3]
    :return: unit vector of given vector
    :rtype: numpy.ndarray [1, 3]
    '''
    return vector / np.linalg.norm(vector)


def create_vectors(start_end_point_array):
    ''' Create vector array from start and end point array
    
    :param start_end_point_array: array of start and end point for vectors 
    :type start_end_point_array: numpy.ndarray [Num_vectors, 2, 3]
    :return vec_return: array of vectors from given start to end point
    :rtype: numpy.ndarray [Num_vectors, 3]
    '''
    vec_return = np.zeros((len(start_end_point_array), 3))
    for i in range(len(start_end_point_array)):
        vec_i = start_end_point_array[i][1] - start_end_point_array[i][0]
        vec_return[i] = vec_i
    return vec_return


def create_vec_distances(vectors):
    ''' Compute vector distances for given array of vectors
    
    :param vectors: array of 3D vectors
    :type vectors: numpy.ndarray [N, 3]
    :return vector_dists: array of vector distances
    :rtype: numpy.ndarray [N, 1]
    '''
    vector_dists_list = []
    for i in range(len(vectors)):
        vector_i_dist = norm_vec(vectors[i])
        vector_dists_list.append(vector_i_dist)
    vector_dists = np.array(vector_dists_list)
    return vector_dists

def get_orthonormal_vec_coords(orthonnormal_vec_coords, which):
    ''' Select specified vector coordinates from orthogonal vector coordinate array
    
    :param orthonormal_vec_coords: list/ array of 3D orthogonal vectors
    :type orthonormal_vec_coords: numpy.ndarray [3, 3]
    :param which: string decoding whether to get 'longest', 'middle' or 'shortest' orthogonal vec
    :type which: string
    :return: 3D vector coordinates of specified orthogonal vector
    :rtype: numpy.ndarray [1, 3]
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
        return None


def rot_mat_from_vecs(vec1, vec2):
    ''' Find rot-mat for aligning vec1 with vec 2 (based on Rodrigues' rotation formula)
    
    :param vec1: 3D vector coordinates of vector to be rotated
    :type vec1: numpy.ndarray [1, 3]
    :param vec2: 3D vector coordinates of vector to be rotated onto
    :type vec2: numpy.ndarray [1, 3] 
    :return rot_mat: Rotation matrix for rotating vec1 onto vec2
    :rtype: numpy.ndarray [3, 3]
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


def rotate_coords_onto_axis(coords, R):
    ''' Function that rotates coords according to given rotation-matrix R
    
    :param coords: coordinates of N points to be rotated
    :type coords: numpy.ndarray [N, 3]
    :param R: rotation-matrix to be applied to coordinates
    :type R: numpy.ndarray [3, 3]
    :return coords_rot: new rotated coordinates of input coordinates
    :rtype: numpy.ndarray [N, 3]
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