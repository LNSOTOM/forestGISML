############################
# SPATIAL ANALYSIS & SPATIAL AUTOCORRELATION AFTER ML
############################
"""##### 9 [ Spatial Visualization for Predictions ] #####"""
df_coord=pd.concat([dataTest,X_test_coord], axis=1)
df_coord

"""## 9.1 Convert to Shapefile"""
import geopandas as gpd
from pyproj import CRS

# create data-frame
grid_df_coord = df_coord[['Actual','Predicted','residuals','x','y','siteindex']]

# Convert to spatial point data-frame
grid_df_coord = gpd.GeoDataFrame(
    grid_df_coord, geometry=gpd.points_from_xy(grid_df_coord.x, grid_df_coord.y))

grid_df_coord.crs = CRS("EPSG:28355")
grid_df_coord.crs

#Save as ESRI shape file
from google.colab import files

grid_df_coord.to_file("ML_PRED_Dtree.shp")

"""## 9.2 Read Shapefile"""
#Open GP=state polygon data 
import geopandas as gpd

fp_state="ML_PRED_Dtree.shp"

# read file using gpd.read_file()
ML_PRED_Dtree = gpd.read_file(fp_state) 
ML_PRED_Dtree.head()

"""## 9.3 Plot Shapefile"""
fp_state = ('/content/drive/My Drive/EDAsurvey_point/ML_PRED_Dtree')

import geopandas as gpd
# read file using gpd.read_file()
ML_PRED_Dtree = gpd.read_file(fp_state) 
ML_PRED_Dtree.head()

ML_PRED_Dtree.crs

ML_PRED_Dtree.plot()

"""## 9.4 Analysis Shapefile
### a) 2D Test set in ML \n Quantiles Measurement of Predicted 'siteindex' Values
"""
#fig, ax = plt.subplots(nrows=1, ncols=1,  sharex=False, sharey=False)
plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
plt.scatter(ML_PRED_Dtree.x, ML_PRED_Dtree.y, c=ML_PRED_Dtree.Predicted, cmap='hsv')

#ax.set_xlabel('Longitude') #it's a good idea to label your axes
#ax.set_ylabel('Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

cbar= plt.colorbar()
cbar.set_label('Site Index (m)', labelpad=+1)

plt.title("Site Index Predicted by Location")
#plt.legend(title="Site Index (m)",loc='upper center', bbox_to_anchor=(1.15, 0.8), frameon=False)

plt.savefig('PredictedTestSet2D_dtree.png', bbox_inches='tight', dpi=300)

"""### b) 3D Test set in ML \n Quantiles Measurement of Predicted 'siteindex' Values"""
import seaborn as sns
fig = plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
sns.set(style="whitegrid")

# Data for a three-dimensional line
zline = ML_PRED_Dtree.Predicted
xline = ML_PRED_Dtree.x
yline = ML_PRED_Dtree.y

im = ax.scatter3D(xline, yline, zline, c= ML_PRED_Dtree.Predicted, cmap='hsv')

ax.set_xlabel('Longitude') #it's a good idea to label your axes
ax.set_ylabel('Latitude')

#cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

cbar= fig.colorbar(im, orientation='vertical', shrink=0.6)
cbar.set_label('Site Index (m)', labelpad=+1)

plt.title("Site Index Predicted by Location")
#plt.legend(title="Site Index (m)",loc='upper center', bbox_to_anchor=(1.15, 0.8), frameon=False)

plt.savefig('PredictedTestSet3D_dtree.png', bbox_inches='tight', dpi=300)

"""### c) Test set in ML \n Quantiles Measurement of Predicted 'siteindex' Values"""
#!pip install mapclassify
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify
# 2. MAP 2
fig, ax = plt.subplots(figsize=(14,12), subplot_kw={'aspect':'equal'})

ML_PRED_Dtree.plot(column="Predicted", scheme='Quantiles', k=5, cmap='Spectral', legend=True, legend_kwds={'title':'Growth Potential at a Site:'}, ax=ax)

# add a title for the plot
ax.set_title("Productivity in Radiata pine | Test set in ML \n Quantiles Measurement of Predicted 'siteindex' Values");

plt.savefig('pointsRadiataTas_Predicted_dtree.jpg', bbox_inches='tight', dpi=300)

"""### d) Test set in ML \n Quantiles Measurement of Actual 'siteindex' Values"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify
# 2. MAP 2
fig, ax = plt.subplots(figsize=(14,12), subplot_kw={'aspect':'equal'})

ML_PRED_Dtree.plot(column="Actual", scheme='Quantiles', k=5, cmap='Spectral', legend=True, legend_kwds={'title':'Growth Potential at a Site:'}, ax=ax)

# add a title for the plot
ax.set_title("Productivity in Radiata pine | Test set in ML \n Quantiles Measurement of Observed 'siteindex' Values");

plt.savefig('pointsRadiataTas_Actual_dtree.jpg', bbox_inches='tight', dpi=300)

"""### e) Test set in ML \n Quantiles Measurement of Residuals 'siteindex' Values"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mapclassify
# 2. MAP 2
fig, ax = plt.subplots(figsize=(14,12), subplot_kw={'aspect':'equal'})

ML_PRED_Dtree.plot(column="residuals", scheme='Quantiles', k=5, cmap='Spectral', legend=True, legend_kwds={'title':'Growth Potential at a Site:'}, ax=ax)

# add a title for the plot
ax.set_title("Productivity in Radiata pine | Test set in ML \n Quantiles Measurement of Residuals 'siteindex' Values");

plt.savefig('pointsRadiataTas_Residuals_dtree.jpg', bbox_inches='tight', dpi=300)

"""##### V. SPATIAL AUTOCORRELATION AFTER ML #####"""
# 1 dataTrain
## 1.1 Reinsert coordinates to data TRAIN set
dataTrain_coord

dataTrain_coord_point = gpd.GeoDataFrame(dataTrain_coord, geometry=gpd.points_from_xy(dataTrain_coord.x, dataTrain_coord.y))

dataTrain_coord_point.info()

dataTrain_coord_point.dropna(inplace=True)

dataTrain_coord_point

"""## 1.2 Spatial weights matrix"""
# spatial weights matrix. We will use nearest neighbors 
w = weights.contiguity.Queen.from_dataframe(dataTrain_coord_point)
# w.neighbors

# Row-standardization THE WEIGHTS
w.transform = 'R'

"""## 1.3 Spatial Lag"""
# calculate the spatial lag of our variable of interest:
dataTrain_coord_point['w_Actual'] = weights.spatial_lag.lag_spatial(w, dataTrain_coord_point['Actual'])

# the variable of interest is usually standardized by substracting its mean and dividing it by its standard deviation:
dataTrain_coord_point['Actual_std'] = ( dataTrain_coord_point['Actual'] - dataTrain_coord_point['Actual'].mean() )\
                    / dataTrain_coord_point['Actual'].std()

dataTrain_coord_point['w_Actual_std'] = ( dataTrain_coord_point['w_Actual'] - dataTrain_coord_point['w_Actual'].mean() )\
                    / dataTrain_coord_point['w_Actual'].std()

"""### 1.3.1 Plot Moran I"""
# creating a Moran Plot
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(10, 6))
# Plot values
seaborn.regplot(x='Actual_std', y='w_Actual_std', data=dataTrain_coord_point, ci=None)
# Display
plt.show()

"""### 1.3.2 Calculate Moran values stats"""
# retrieve the value of the statistic
moran = esda.moran.Moran(dataTrain_coord_point['Actual'], w)
#retrieve the value of the statistic
moran.I
#-0.00011751126271195253

moran.p_sim
# 0.407
#0.445

"""### 1.3.3 Moran I plot vs p-value"""
from splot.esda import plot_moran

#obtain a quick representation of the statistic that combines the Moran Plot (right) with a graphic of the empirical test that we carry out to obtain p_sim (left):
plot_moran(moran);

"""# 2 dataTest
## 2.1 Reinsert coordinates to data TEST set
"""
dataTest_coord=pd.concat([dataTest,X_test_coord], axis=1)
dataTest_coord

dataTest_coord_point = gpd.GeoDataFrame(dataTest_coord, geometry=gpd.points_from_xy(dataTest_coord.x, dataTest_coord.y))

dataTest_coord_point.dropna(inplace=True)

dataTest_coord_point.info()

"""## 2.2 Spatial weights matrix"""
# spatial weights matrix. We will use nearest neighbors 
w = weights.contiguity.Queen.from_dataframe(dataTest_coord_point)
# w.neighbors

# Row-standardization THE WEIGHTS
w.transform = 'R'

"""## 2.3 Spatial Lag"""
# calculate the spatial lag of our variable of interest:
dataTest_coord_point['w_Predicted'] = weights.spatial_lag.lag_spatial(w, dataTest_coord_point['Predicted'])

# the variable of interest is usually standardized by substracting its mean and dividing it by its standard deviation:
dataTest_coord_point['Predicted_std'] = ( dataTest_coord_point['Predicted'] - dataTest_coord_point['Predicted'].mean() )\
                    / dataTest_coord_point['Predicted'].std()

dataTest_coord_point['w_Predicted_std'] = ( dataTest_coord_point['w_Predicted'] - dataTest_coord_point['w_Predicted'].mean() )\
                    / dataTest_coord_point['w_Predicted'].std()

"""### 2.3.1 Plot Moran I"""
# creating a Moran Plot
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(10, 6))
# Plot values
seaborn.regplot(x='Predicted_std', y='w_Predicted_std', data=dataTest_coord_point, ci=None)
# Display
plt.show()

"""### 2.3.2 Calculate Moran values stats"""
# retrieve the value of the statistic
moran = esda.moran.Moran(dataTest_coord_point['Predicted'], w)

#retrieve the value of the statistic
moran.I
# 0.0055366336643759735
#0.005536633664375971

moran.p_sim
# 0.177
#0.185

"""### 2.3.3 Moran I plot vs p-value"""
from splot.esda import plot_moran
#obtain a quick representation of the statistic that combines the Moran Plot (right) with a graphic of the empirical test that we carry out to obtain p_sim (left):
plot_moran(moran);

# 3 dataTest Residuals | Polynomial
## 3.1 Resinsert coord to RESIDUAL TEST set
"""
dataTest_coord=pd.concat([dataTest,X_test_coord], axis=1)
dataTest_coord

dataTest_coord_point = gpd.GeoDataFrame(dataTest_coord, geometry=gpd.points_from_xy(dataTest_coord.x, dataTest_coord.y))

dataTest_coord_point.dropna(inplace=True)

dataTest_coord_point

"""## 3.2 Spatial weights matrix"""
# spatial weights matrix. We will use nearest neighbors 
w = weights.contiguity.Queen.from_dataframe(dataTest_coord_point)

# Row-standardization THE WEIGHTS
w.transform = 'R'

"""## 3.3 Spatial Lag"""
# calculate the spatial lag of our variable of interest:
dataTest_coord_point['w_residuals'] = weights.spatial_lag.lag_spatial(w, dataTest_coord_point['residuals'])

# the variable of interest is usually standardized by substracting its mean and dividing it by its standard deviation:
dataTest_coord_point['residuals_std'] = ( dataTest_coord_point['residuals'] - dataTest_coord_point['residuals'].mean() )\
                    / dataTest_coord_point['residuals'].std()

dataTest_coord_point['w_residuals_std'] = ( dataTest_coord_point['w_residuals'] - dataTest_coord_point['w_residuals'].mean() )\
                    / dataTest_coord_point['w_residuals'].std()

"""### 3.3.1 Plot Moran I"""
# creating a Moran Plot
# Setup the figure and axis
f, ax = plt.subplots(1, figsize=(10, 6))
# Plot values
seaborn.regplot(x='residuals_std', y='w_residuals_std', data=dataTest_coord_point, ci=None)
# Display
plt.show()

"""### 3.3.2 Calculate Moran values stats"""
# retrieve the value of the statistic
moran = esda.moran.Moran(dataTest_coord_point['residuals'], w)

#retrieve the value of the statistic
moran.I
# -0.0015285569645116612

moran.p_sim
# 0.419

"""### 3.3.3 Moran I plot vs p-value"""
from splot.esda import plot_moran
#obtain a quick representation of the statistic that combines the Moran Plot (right) with a graphic of the empirical test that we carry out to obtain p_sim (left):
plot_moran(moran);
