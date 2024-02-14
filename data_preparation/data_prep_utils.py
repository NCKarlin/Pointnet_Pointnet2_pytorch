import os 
import numpy as np
import math
import open3d as o3d


def create_rgb_list(input_pc):
    """Function to create RGB-list for point cloud creation and display

    The RGB-list has to be handed over individually from the point coordinates for the 
    visualization of a point cloud, hence this function for its individual creation. 
    N: number of points | F: number of features per point

    Args:
        input_pc (np.ndarray): input point coud with RGB values in column 3-5 [N, F]

    Returns:
        np.ndarray: array of normalized RGB values per point [N, 3]
    """
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
    """Function to create an open3d point cloud object from point cloud data

    Args:
        point_data (np.ndarray): array containing coordinates and RGB data of all points of pc [N, F]
        color_data (np.ndarray): array containing per point RGB data [N, 3]

    Returns:
        open3d.geometry.PointCloud: open3d point cloud data object from input data
    """
    pcd = o3d.geometry.PointCloud() # Create PC object
    pcd.points = o3d.utility.Vector3dVector(point_data) # Give coordinates
    # Coloring the PC
    if len(color_data) == 0:
        pcd.paint_uniform_color([1, 0, 0])
    else:
        pcd.colors = o3d.utility.Vector3dVector(color_data)
    return pcd


def get_lowest_corner_coords(cuboid_coords, axis):
    """Retrieving coordinates of lowest cuboid corner in respective direction

    Args:
        cuboid_coords (np.ndarray): array of coordinates of cuboid corners [8, 3]
        axis (str): string specifying axis/ direction for minimum point value

    Returns:
        np.ndarray: lowest_corner_coords - xyz-coordinates of lowest corner in specified axis/ direction
    """
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
    """Function to ceate all 7 vectors from the lowest corner to all other cuboid-corners

    Args:
        bbox_points (np.ndarray): array of coordinates of all 8 cuboid corner points [8, 3]
        lowest_corner_coords (np.ndarray): coords of chosen lowest corner of cuboid coords

    Returns:
        np.ndarray: lowest_corner_to_corner_vectors_list - array of all vectors from lowest
                    to all others
    """
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
    """Function to determine orthogonal corner vectors

    Given the vectors from the lowest corner to all other corners, as well as the shortest
    lowest corner to corner vectors, this function determines the other two orthogonal 
    vectors, which meet at the lowest corner of the cuboid.

    Args:
        shortest_lowest_corner_to_corner_vector_coords (np.ndarray): [1, 3]
        lowest_corner_to_corner_vecs (np.ndarray): [7, 3] 

    Returns:
        list: orthogonal_vec_coords_list - list of 3 orthogonal vectors at lowest corner
    """
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
    """Normalize given vector, this function was mainly crated for readability

    Args:
        vector (np.ndarray): any 3D-vector/ coordinate combination

    Returns:
        float: vector norm
    """
    return np.linalg.norm(vector)


def unit_vector(vector):
    """Create unit-vector for given vector, was mainly created for readability

    Args:
        vector (np.ndarray): 3D-vector coordinates

    Returns:
        np.ndarray: unit vector of given input vector 
    """
    return vector / np.linalg.norm(vector)


def create_vectors(start_end_point_array):
    """Create vector array from start and end point array

    Args:
        start_end_point_array (np.ndarray): array of start and end points for vectors [N, 2, 3]

    Returns:
        np.ndarrray: array of vectors from given start to end points [N, 3]
    """
    vec_return = np.zeros((len(start_end_point_array), 3))
    for i in range(len(start_end_point_array)):
        vec_i = start_end_point_array[i][1] - start_end_point_array[i][0]
        vec_return[i] = vec_i
    return vec_return


def create_vec_distances(vectors):
    """Compute vector distances for given array of vectors

    Args:
        vectors (np.ndarray): array of 3D-vectors [N, 3]

    Returns:
        np.ndarray: array of vector distances [N, 1]
    """
    vector_dists_list = []
    for i in range(len(vectors)):
        vector_i_dist = norm_vec(vectors[i])
        vector_dists_list.append(vector_i_dist)
    vector_dists = np.array(vector_dists_list)
    return vector_dists

def get_orthonormal_vec_coords(orthonnormal_vec_coords, which):
    """Select specified vector coordinates from orthogonal vector coordinate array

    Args:
        orthonnormal_vec_coords (np.ndarray): array of 3D orthogonal vectors [3, 3]
        which (str): string decoding whether to get 'shortest', 'middle' or 'longest'
                     orthogonal vector

    Returns:
        np.ndarray: 3D vector coordinates of specified orthogonal vector
    """
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
    """Find rot-mat for aligning vec1 with vec 2 (based on Rodrigues' rotation formula)

    Args:
        vec1 (np.ndarray): 3D vectors coordinates of vector to be rotated
        vec2 (np.ndarray): 3D vector coordinates of vector to be rotated onto

    Returns:
        np.ndarray: Rotation matrix for rotating vec1 onto vec2
    """
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
    """Function that rotates coords according to given rotation-matrix R

    Args:
        coords (np.ndarray): 3D coordinates of N points to be rotated [N, 3]
        R (np.ndarray): rotation-matrix ot be applied to coordinates [3, 3]

    Returns:
        np.ndarray: new rotated coordinates of input coordinates [N, 3]
    """
    # Depending on shape of coordinates passed rotation multiplication adjustment
    if coords.shape == (3,):
        coords_rot = R.dot(coords)
    coords_rot_list = []
    for i in range(len(coords)):
        coords_i_rot = R.dot(coords[i])
        coords_rot_list.append(coords_i_rot)
    coords_rot = np.array(coords_rot_list)
    return coords_rot