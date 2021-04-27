@author: laurasotomayor

# Libraries
"""
# I_OPERATING SYSTEMS libraries
## 1.to upload files
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

## 2.functions for interacting with the operating system
import os

## 3.Commented out IPython magic to ensure Python compatibility.
## Load time
!pip install ipython-autotime
%load_ext autotime

##############################################

# II_GIS libraries
## 1.Python interface to PROJ (cartographic projections and coordinate transformations library)
!pip install pyproj
from pyproj import CRS

## 2.R-Tree spatial index for Python GIS
!pip install Rtree

## 3.library for Choropleth map classification
!pip install mapclassify
import mapclassify

## 4.GeoPandas is a project to add support for geographic data to pandas objects (work with spatial data)
!pip install geopandas
import geopandas as gpd

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################

# III_SPATIAL ANALYSIS & STATISTICS libraries
## 1.Python interface to PROJ (cartographic projections and coordinate transformations library)
!pip install pysal

from pysal.lib import weights
from pysal.explore import esda
from pysal.lib import cg as geometry

from splot.esda import plot_moran


##############################################

# IV_DATA ANALYTICS/EXPLORATION & STATISTICS libraries

## 1.for data
import pandas as pd
import numpy as np

import pandas.util.testing as tm

## 2.for plotting
import matplotlib.pyplot as plt
!pip install matplotlib==3.1.3
import seaborn as sns

## 3.for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

##############################################

# V_MACHINE LEARNING & PRE-PROCESS

## 1.encode categorical data
!pip install category-encoders

## 2.for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

### normalize
from sklearn.preprocessing import MinMaxScaler

### preprocess
from sklearn import preprocessing

## 3.Local Interpretable Model-Agnostic Explanations for machine learning classifiers
!pip install lime

### for explainer
from lime import lime_tabular
