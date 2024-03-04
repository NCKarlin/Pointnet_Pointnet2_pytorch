import os 
import geopandas as gpd
from data_creation_utils import (
    save_as_shp,
    save_as_pc
)

convert_and_save_pc = False

# Create shapefile path
ROOT = os.getcwd()
shapefile_path_rel = os.path.join(ROOT, 'data', 'GEUS', 'shapefiles', 'Geometries_280623_polylines.shp')
shapefile_path_abs = '/Users/nk/Documents/GitHubRepos/Pointnet_Pointnet2_pytorch/data/GEUS/shapefiles/Geometries_280623_polylines.shp'

# Read in the shapefile
shp = gpd.read_file(shapefile_path_abs)
# Print checks 
print(shp.head())

# Removing one line with NaN values that breaks everything
print('Removing faulty NaN line.')
print(shp.shape)
shp = shp.drop([11513])
print(shp.shape)

# Extracting the lines that correspond to a specific model/ range of images
images = [''] # list of images of a model containing shapefiles 

shp_Nus18 = shp.loc[shp['Model'].str.contains(images[0].split("_")[0])]
shp_Nus18[['Start', 'End']] = shp_Nus18['Model'].str.split(pat=';', n=1, expand=True)
shp_part = shp_Nus18.loc[shp_Nus18['Start'].str.contains('|'.join(images)) & shp_Nus18['End'].str.contains('|'.join(images))]
shp_part.drop(['Start', 'End'], axis=1)

print('Resulting shape with this method is: ', shp_part.shape)

# Save as new shapefile or create the point cloud
save_as_shp(images, shp_part, ROOT)
# if convert_and_save_pc:
#     save_as_pc(filepath_to_extracted_shp_file, 
#                filepath_for_saving_pc)

