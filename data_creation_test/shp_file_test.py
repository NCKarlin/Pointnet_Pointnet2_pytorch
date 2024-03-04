import geopandas as gpd
# Remember installing descartes package
import matplotlib.pyplot as plt

# Read in the shapefile
shp_file = gpd.read_file('path to shp file to display')

# Print checks
print(shp_file.shape)
print(shp_file.head())

# Simply plot the shapefile 
shp_file.plot()
plt.show()