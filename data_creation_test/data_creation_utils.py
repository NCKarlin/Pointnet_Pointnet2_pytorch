import os
import numpy as np
import open3d as o3d
from shapely.geometry import LineString, mapping
from numpy.lib.stride_tricks import sliding_window_view 


def linestring_to_list(geom):
    m = mapping(geom)
    list_of_tuples = m['coordinates']
    list_of_lists = list(map(list, list_of_tuples))
    return np.array(list_of_lists)


def getShapePoints(gdf, upsample=False):
    step_size_def = 1
    shape_points = np.empty([0, 3])
    for linestring in gdf.geometry:
        line_list = linestring_to_list(linestring)
        if upsample:
            print("Upsampling shape points")
            line_ar = np.array(line_list) #(nr of points in line, 3 coordinates)
            pairs_ar = sliding_window_view(line_ar, (2, 3)).squeeze()
            if len(pairs_ar[:, 1].shape) == 1: #COME UP WITH SOMETHING BETTER HERE
                continue
            pair_dists = np.linalg.norm(pairs_ar[:, 1]-pairs_ar[:, 0], axis=1)
            steps = np.floor(pair_dists/step_size_def).astype(int)+2 #plus two for the edges
            for i, (pair, step) in enumerate(zip(pairs_ar, steps)):
                if i == len(pairs_ar)-1:
                    inter_points = np.linspace(pair[0], pair[1], step, endpoint = True)
                else:
                    inter_points = np.linspace(pair[0], pair[1], step, endpoint = False)
                shape_points = np.vstack([shape_points, inter_points])
        else:
            shape_points = np.vstack([shape_points, np.array(line_list)])
    return shape_points


def makePC(point_data, color_data=np.array([])):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)
    if len(color_data) == 0:
        pcd.paint_uniform_color([1, 0, 0])
    else:
        pcd.colors = o3d.utility.Vector3dVector(color_data)
    return pcd


def save_as_shp(images, shp_file_to_be_saved, ROOT):
    output_filename = "lines_" + "_".join([images[0].split("_")[0], images[0].split("_")[-1], images[-1].split("_")[-1]]) + ".shp"
    print(f'File will be saved as {output_filename}.')
    shp_file_to_be_saved.to_file(os.path.join(ROOT, 'data', 'GEUS', 'shapefiles', 'per_model', output_filename))
    

def save_as_pc(shp_file_path, pc_saving_path):
    shp_file = gpd.read_file(shp_file_path)
    shp_points = getShapePoints(shp_file, upsample=True)
    print('Number of points in the final shapefile point cloud: ', shp_points.shape)
    shp_pc = makePC(shp_points)
    o3d.io.write_point_cloud(pc_saving_path, shp_pc)
    