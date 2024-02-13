import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
import math


def load_specific_blocks(path_to_sample_folder, block_load_list):
    """ Load specific blocks to list of arrays containing points and labels

    Args:
        path_to_sample_folder (str): path to the sample containing the blocks to be loaded
        block_load_list (list): list of integers of blocks to be loaded

    Returns:
        list: block_point_ls - list of arrays containing point data of specified blocks
        list: block_labels_ls - list of arrays containing the point labels of blocks
    """
    block_points_ls = [] 
    block_labels_ls = []
    # Loading specific blocks
    for block_id in block_load_list:
        block_i_path = os.path.abspath(os.path.join(path_to_sample_folder, f'block_{block_id}.npy'))
        block_i_data = np.load(block_i_path)
        block_i_points, block_i_labels = block_i_data[:, 0:6], block_i_data[:, 6]
        block_points_ls.append(block_i_points)
        block_labels_ls.append(block_i_labels)
    return block_points_ls, block_labels_ls


def load_sample(path_to_sample_folder):
    """Load entire sample to list of arrays containing block points and labels

    Args:
        path_to_sample_folder (str): path to the sample containing the blocks to be loaded

    Returns:
        list: sample_points_ls - list of arrays containing all block points of sample
        list: sample_label_ls - list of arrays containing all block labels of sample
    """
    num_blocks = len(os.listdir(path_to_sample_folder))
    sample_points_ls = [] 
    sample_labels_ls = []
    for block_id in range(num_blocks):
        block_i_path = os.path.abspath(os.path.join(path_to_sample_folder, f'block_{block_id}.npy'))
        block_i_data = np.load(block_i_path)
        block_i_points, block_i_labels = block_i_data[:, 0:6], block_i_data[:, 6]
        sample_points_ls.append(block_i_points)
        sample_labels_ls.append(block_i_labels)
    return sample_points_ls, sample_labels_ls


def normalize_rgb(points):
    """Normalize the RGB values of points, provided they are in column 3:6 

    Args:
        points (np.ndarray): array of blocks with points and their features

    Returns:
        np.ndarray: points_normed - array of blocks with points and normalized RGB values
    """
    if isinstance(points, np.ndarray):
        num_blocks = points.shape[0]
        num_features = points.shape[2]
        points_normed = np.reshape(points, (-1, num_features))
        points_normed[:, 3:6] /= 255.0
        points_normed = points_normed.reshape((num_blocks, -1, num_features))
        return points_normed
    elif isinstance(points, list):
        points = np.array(points)
        num_blocks = points.shape[0]
        num_features = points.shape[2]
        points_normed = np.reshape(points, (-1, num_features))
        points_normed[:, 3:6] /= 255.0
        points_normed = points_normed.reshape((num_blocks, -1, num_features))
        return points_normed
    else:
        print("Please convert input to function to list or np.ndarray first.")
        return None
        
    
def create_rgb_list(point_data):
    """Create normalized RGB-color array from points for display

    Args:
        point_data (np.ndarray): array of blocks with points and their features

    Returns:
        np.ndarray: rgb_array - array of per point rgb color values
    """
    # Pulling color channels individually
    red_c = point_data[:,3]
    green_c = point_data[:,4]
    blue_c = point_data[:,5]
    # Shifting/ Normalizing color range to [0,1]
    red_c = (red_c - np.min(red_c)) / (np.max(red_c) - np.min(red_c))
    green_c = (green_c - np.min(green_c)) / (np.max(green_c) - np.min(green_c))
    blue_c = (blue_c - np.min(blue_c)) / (np.max(blue_c) - np.min(blue_c))
    # Creating rgb list array to return
    rgb_array = np.ones((point_data.shape[0], 3))
    rgb_array[:,0] = red_c
    rgb_array[:,1] = green_c
    rgb_array[:,2] = blue_c
    return rgb_array


def pull_xyz_coords(point_data):
    """Separate x-, y- and z-coordinates from point data array

    Args:
        point_data (np.ndarray): array of blocks with points and their features 

    Returns:
        np.ndarray: x_coords - array with all per point x-coordinates
        np.ndarray: y_coords - array with all per point y-coordinates
        np.ndarray: z_coords - array with all per point z-coordinates
    """
    x_coords = point_data[:,0]
    y_coords = point_data[:,1]
    z_coords = point_data[:,2]
    return x_coords, y_coords, z_coords


def create_marked_rgb_array(points, labels, mark_color=[1,0,0]):
    """Mark fracture points of sample in desired color

    Args:
        points (np.ndarray): array of blocks with points and their features 
        labels (np.ndarray): array of per point labels
        mark_color (list, float): list of normalized RGB values for marking. Defaults to [1,0,0].
        mark_point_size (float): float defining point size for marked points. Defaults to 15.

    Returns:
        np.ndarray: marked_rgb_list - color array holding marked and unmarked points color info
        np.ndarray: point_sizes - array holding individual point sizes
    """
    red_c = np.array(points[:,3])
    green_c = np.array(points[:,4])
    blue_c = np.array(points[:,5])
    for i in range(len(labels)):
        if labels[i] == 1.0:
            red_c[i] = mark_color[0]
            green_c[i] = mark_color[1]
            blue_c[i] = mark_color[2]
    return np.stack([red_c, green_c, blue_c], axis=1)


def create_marked_point_sizes(labels, point_size=0.75, marked_point_size=15.0):
    """Create point sizes array for marked display

    Args:
        labels (np.ndarray): array holding labels for points
        point_size (float, optional): size of non frature points. Defaults to 0.75.
        marked_point_size (float, optional): size of fracture points. Defaults to 15.0.

    Returns:
        np.ndarray: point_sizes - array holding point sizes for all points
    """
    point_sizes = np.zeros_like(labels)
    point_sizes.fill(point_size)
    for i in range(len(labels)):
        if labels[i] == 1:
            point_sizes[i] = marked_point_size
    return point_sizes


def create_dataloading_lists(points_ls, labels_ls, batch_size, num_blocks):
    """Create point and label lists for iterative dataloading for model inference

    Args:
        points_ls (list): list of arrays containing per block point data
        labels_ls (list): list of arrays containing per block point labels
        batch_size (int): integer defining batch size
        num_blocks (int): total number of blocks in sample

    Returns:
        list: test_points - list containing batches of point data
        list: test_labels - list containing batches of point labels
    """
    test_points = []
    test_labels = []
    for i in range(num_blocks):
        points = torch.tensor(points_ls[i*batch_size:i*batch_size+4]) # [B, F, N]
        labels = torch.tensor(labels_ls[i*batch_size:i*batch_size+4]) # [B, N]
        points = points.transpose(2, 1)
        test_points.append(points)
        test_labels.append(labels)
    return test_points, test_labels


def generate_predictions(model, test_points_ls, test_labels_ls, loss_function, DEVICE):
    """Generate sample predictions with loaded model

    Args:
        model (fragrec_min.get_model): loaded model used for inerence
        test_points_ls (list): list containing batches of point data
        test_labels_ls (list): list containing batches of point labels
        loss_function (string): description of loss function
        DEVICE (torch.device): device used for calculations (cuda/ pcu)

    Returns:
        np.ndarray: array containing all point predictions
    """
    preds_ls = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_points_ls)):
            points, labels = test_points_ls[i], test_labels_ls[i]
            points, labels = points.float().to(DEVICE), labels.long().to(DEVICE)
            _, pred_probs = model(points, loss_function)
            if loss_function == 'BCE-Loss':
                pred_probs = torch.squeeze(pred_probs.contiguous().view(-1, 1))
                pred_choice = np.zeros(len(pred_probs))
                pred_choice = torch.where(pred_probs >= 0.6, 1, 0).cpu().data.numpy()
                preds_ls.extend(pred_choice)
    return np.array(preds_ls, dtype=np.float32)


def create_and_save_figure(points_data, rgb_arr, point_sizes, savepath, figure_name='marked_preds.png', dpi=500):
    """Function to create and save a marked point cloud for visualization

    Args:
        points_data (np.ndarray): array containing all per point data (features)
        rgb_arr (np.ndarray): array containing the per point normalized RGB-values
        point_sizes (np.ndarray): 1D-array containing per point sizes for display
        savepath (str): path to the target directory for saving
        figure_name (str): title of the figue to be saved. Defaults to 'marked_preds.png'.
        dpi (int): Integer specifying the dpi. Defaults to 500.
    """
    x_coords, y_coords, z_coords = pull_xyz_coords(points_data)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, 
               y_coords, 
               z_coords, 
               c=rgb_arr,
               s=point_sizes)
    ax.view_init(elev=90, azim=90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.savefig(os.path.join(savepath, figure_name), dpi=dpi)
    

def print_stats():
    pass 


def create_and_save_skeleton():
    pass