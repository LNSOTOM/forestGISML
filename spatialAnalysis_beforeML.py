############################
# SPATIAL ANALYSIS & SPATIAL AUTOCORRELATION
############################
"""##### I. DATA PREPARATION #####"""
# 1 Load Dataset
# Define relative path to file
EDAsurvey = pd.read_csv('/content/drive/<path file>')
EDAsurvey.head(3)

# don't forget to clean and pre-process the data: check [data_preprocessing.py]

"""##### III. SPATIAL ANALYSIS #####"""
## 3.0 Convert PandasDataFrame to GeoDatrame
import geopandas as gpd

#convert PandasDataFrame to GeoDataFrame (needs shapely object)
#We use geopandas points_from_xy() to transform x=Longitude and y=Latitude into a list of shapely
EDAsurvey_point = gpd.GeoDataFrame(EDAsurvey, geometry=gpd.points_from_xy(EDAsurvey.x, EDAsurvey.y))

EDAsurvey_point.head()

"""## 3.1 Map Projection"""
from pyproj import CRS

# option 2: Defining projection: we assign the GDA94 / MGA zone 55 latitude-longitude CRS (EPSG:28355) to the crs attribute:
EDAsurvey_point.crs = CRS("EPSG:28355")
EDAsurvey_point.crs

"""## 3.2 Plot point geometries"""
import matplotlib.pyplot as plt
# Make subplots that are next to each other
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

# Plot the data in WGS84 CRS
EDAsurvey_point.plot(ax=ax1, color='#ff48a5', edgecolor='#ffcae5', label="Points cloud");

# Add title
ax1.set_title("GDA94 / MGA zone 55");
ax1.legend(title="Tree's Location",loc='upper center', bbox_to_anchor=(1.1, 0.8), frameon=False)

# Remove empty white space around the plot
plt.tight_layout()

"""## 3.3 Write as ESRI Shape File"""
from google.colab import files
EDAsurvey_point.to_file("EDAsurvey_point.shp")

#dowload into local machine
#files.download("EDAsurvey_point.shp")

"""## 3.4 Map Spatial Distribution of site index
### 3.4.1 Map option 1
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify

# 1. MAP 1
## Plot Soil C
# set the range for the choropleth
vmin, vmax = 4, 51

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(6, 7), subplot_kw={'aspect':'equal'})

# Point data
EDAsurvey_point.plot(column='siteindex', scheme='Quantiles', k =7, cmap='Spectral', alpha=1, markersize=30, ax=ax)

# add a title
ax.set_title('Site Index Productivity (m)', fontdict={'fontsize': '20', 'fontweight' : '3'})

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
# You need to import mpl_toolkits.axes_grid1 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Spectral',
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))

# empty array for the data range
sm._A = []

# add the colorbar to the figure
cbar = fig.colorbar(sm, cax=cax)

"""### 3.4.2 Map option 2"""
# 2. MAP 2
fig, ax = plt.subplots(figsize=(14,12), subplot_kw={'aspect':'equal'})

EDAsurvey_point.plot(column="siteindex", scheme='Quantiles', k=5, cmap='Spectral', legend=True, legend_kwds={'title':'Growth Potential at a Site:'}, ax=ax)

# add a title for the plot
ax.set_title("Productivity in Radiata pine \n Quantiles Measurement of the Growth Potential at a Site");

plt.savefig('pointsRadiataTas.jpg', bbox_inches='tight', dpi=300)

"""## 3.5 Save new dataframe with geometry points"""
from google.colab import  drive
#Mounts the google drive to Colab Notebook
drive.mount('/content/drive', force_remount=True)

#Make sure the folder Name is created in the google drive before uploading
EDAsurvey_point.to_csv('/content/drive/My Drive/EDAsurvey_point/EDAsurvey_point.csv', index = False, header=True)

# Define relative path to file
EDAsurvey_point = pd.read_csv('/content/drive/My Drive/EDAsurvey_point/EDAsurvey_point.csv')
EDAsurvey_point.head()

"""## 3.6 Spatial autocorrelation [after PRE-PROCESSING: Target Encoding + Features Selection] + before ML"""
! pip install pysal

"""### 3.6.1 Spatial weights matrix"""
from pysal.lib import weights
from pysal.explore import esda
from pysal.lib import cg as geometry
import seaborn 

# spatial weights matrix. We will use Queen neighbors 
w = weights.contiguity.Queen.from_dataframe(EDAsurvey)

#We will use nearest neighbors 
# w.neighbors

# Row-standardization THE WEIGHTS
w.transform = 'R'

"""### 3.6.2 Spatial Lag"""
# calculate the spatial lag of our variable of interest:
EDAsurvey['w_siteindex'] = weights.spatial_lag.lag_spatial(w, EDAsurvey['siteindex'])

"""### 3.6.3 Moran's (STATISTICS)"""
# 1. Moran's calculation
# the variable of interest is usually standardized by substracting its mean and dividing it by its standard deviation:
EDAsurvey['siteindex_std'] = ( EDAsurvey['siteindex'] - EDAsurvey['siteindex'].mean() )\
                    / EDAsurvey['siteindex'].std()

EDAsurvey['w_siteindex_std'] = ( EDAsurvey['w_siteindex'] - EDAsurvey['w_siteindex'].mean() )\
                    / EDAsurvey['w_siteindex'].std()

!pip install matplotlib==3.1.3

# 2. Plot Moran I
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(12, 6))
# Plot values
seaborn.regplot(x='siteindex_std', y='w_siteindex_std', data=EDAsurvey, ci=None)
plt.ylim([-5, 5])
plt.xlim([-5, 5])

# Display
plt.show()

# 3. retrieve the value of the statistic
moran = esda.moran.Moran(EDAsurvey['siteindex'], w)

#retrieve the value of the statistic
moran.I
# 0.8625947320706182

# 4. P-VALUE
# PySAL performs a simulation and returns a measure of certainty about how likely it is the pattern we observe in our dataset came from a spatially random process
moran.p_sim
#0.001
# Interpretation: A small enough p-value associated with the Moran’s I of a map allows to reject the hypothesis that the map is random.
# we can conclude that the map displays more spatial pattern than we would expect if the values had been randomly allocated to a locations.

#Meaning: if we generated a large number of maps with the same values but randomly allocated over space, 
# and calculated the Moran’s I statistic for each of those maps, only 0.01% of them would display a larger (absolute) value 
# than the one we obtain from the observed data, and the other 99.99% of the random maps would receive a smaller (absolute) value of Moran’s I. 

# => the global autocorrelation analysis can teach us that observations do seem to be positively autocorrelated over space

# 5. Plot Moran I
from splot.esda import plot_moran
#obtain a quick representation of the statistic that combines the Moran Plot (right) with a graphic of the empirical test that we carry out to obtain p_sim (left):
plot_moran(moran);

"""### 3.6.4 Extra"""
# Retrieve, model, analyze, and visualize OpenStreetMap street networks and other spatial data
!pip install osmnx
# to retrieve tile maps from the internet.
!pip install contextily
!pip install rioxarray
# data rasterization pipeline for automating the process of creating meaningful representations of large amounts of data.
!pip install datashader
import pandas
import osmnx
import geopandas
import rioxarray
import xarray
import datashader
import contextily as cx
from shapely.geometry import box
import matplotlib.pyplot as plt

# set up the grid (or canvas, cvs) into which we want to aggregate points
cvs = datashader.Canvas(plot_width=10,
                        plot_height=10
                       )

EDAsurvey_pointFrame = pd.DataFrame(EDAsurvey_point)

# transfer” the points into the grid:
grid = cvs.points(EDAsurvey_pointFrame, 
                  x='x',
                  y='y',                
                 )
grid

grid.attrs

f, axs = plt.subplots(1, 2, figsize=(14, 6))
EDAsurvey_point.plot(ax=axs[0])
grid.plot(ax=axs[1]);
