# For data processing
import torch
import numpy as np
# For 3D visualization
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

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

# Path to input data
input_data_path = "/home/innolidix/Pointnet_Pointnet2_pytorch/data/testdata/data_labelled_int.npy"

# Load the input data
input_data = np.load(input_data_path)

# Assigning the coordinates
input_points = np.stack([input_data[:, 0], 
                         input_data[:, 1], 
                         input_data[:, 2]], axis=1)
# Assigning color channels
red_c   = np.array(input_data[:, 4])
green_c = np.array(input_data[:, 5])
blue_c  = np.array(input_data[:, 6])
# Converting color channels to [0, 1] range
red_c   = (red_c - np.min(red_c)) / (np.max(red_c) - np.min(red_c))
green_c = (green_c - np.min(green_c)) / (np.max(green_c) - np.min(green_c))
blue_c  = (blue_c - np.min(blue_c)) / (np.max(blue_c) - np.min(blue_c))
# Creating color data
color_data = np.stack([red_c, green_c, blue_c], axis=1)

# Make the point cloud to be displayed
viz_pcd = makePC(input_points, color_data)

# Print check for point cloud
print("------ Print check for input point cloud ------")
print(f"The input point cloud consists of {len(viz_pcd.points)} number of points.")
print(f"The minimum coordinates of the point cloud are: {np.amin(np.asarray(viz_pcd.points), axis=0)}")
# Minimum will be [0, 0, 0] as the input data has been shifted already

# Cropping the point cloud using bounding boxes
bbox = viz_pcd.get_oriented_bounding_box()
bbox.color = (0, 1, 0) #bbox in green
viz_pcd_cropped = viz_pcd.crop(bbox)

# Print check for cropped point cloud
print("------ Print check for cropped input point cloud ------")
print(f"The cropped point cloud consists of {len(viz_pcd_cropped.points)} number of points.")
print(f"The bounding box cropping removed {len(viz_pcd.points) - len(viz_pcd_cropped.points)} points.")
print(f"The minimum coordinates of the cropped point cloud are: {np.amin(np.asarray(viz_pcd_cropped.points), axis=0)}")
# Minimum will be [0, 0, 0] as the input data has been shifted already

# Visualizing the point cloud
o3d.visualization.draw_geometries([viz_pcd_cropped])