# forestGISML

# 1 Overview
This research is about forest productivity, identifying the conditioning factors such
as: climatic variables derived from rainfall and temperatures, topographic attributes derived from digital
elevation model, edaphic attributes (geology composition and soil attributes) that contribute to explain
the forest growth. In order to identify these factors, different machine learning algorithms methods
have been applied. 
## 1.1 Goal
The goal is to reach the best model that contributes to explain the predicted
observed variable. The variable observed or target variable represents the site index (SI=mean height
dominant tree at a site) values localized by site. The input of the model starts by using the site index
values derived from a non-linear regression model that establish growth canopy potential at a site and
fit the multiple factors into the models, which could explain the productivity by location. 
## 1.2 Data Sourced
The SI was sourced by the forestry company [Timberlands Pacific](https://www.tppl.com.au/).<br />
Supervisors for this research:<br />
∆ [Dr. Matthew J Cracknell](https://www.utas.edu.au/profiles/staff/codes/matthew-cracknell) (Postdoctoral
Research Fellow in Earth Informatics at the ARC Industrial Transformation Research Hub for
Transforming the Mining Value Chain), <br />
∆ Dr. Robert Musk (Data Analyst/Forest Scientist at
Timberlands Pacific) and <br />
∆ Dr. Muhammad Bilal Amin (Computing & Information Systems, [University of Tasmania](https://www.utas.edu.au/technology-environments-design/ict)).<br />
For data collection and processing, GIS processing techniques have been applying to
extract all the attributes and achieved more accurate predictions.
## 1.3 Data Dimension
The survey selected for analysis was collected from five sites with wide variation in landscape
conditioning and climate factors, and a diverse geology and soil attributes related to the area that
influence the productivity of radiata pine across the estate. The data used for the modelling is
comprised by 23 datasets and 953556 observations.
## 1.4 Contribution
This research will contribute by developing a novel ensemble learning base
technique which will produce predictive models to assist a forest manager in
optimal resource utilization with maximization of productivity.

# 2 Scripts Guideline
## 2.1 Getting Started
0. libraries.txt <br />
1. data_preprocessing.py <br />
## 2.2 Machine Learning Algorithms
∆ linear_regression.py <br />
∆ polynomial_regression.py <br />
∆ decisionTree_regression.py <br />
∆ randomForest_regression.py <br />
∆ gradientBoosting_regression.py <br />
∆ deepNeuralNetwork.py <br />
## 2.3 Exploration Data Analysis
∆ EDA.py <br />
--Data visualization: bi-variate plots, bar plots... <br />
--EDA Report (summary)
## 2.4 Statistics 
∆ statistics.py <br />
--Descriptive Statistics <br />
--Correlation Coefficient analysis (Spearman)<br />
--Principal Component Analysis (PCA)<br />
--Regression assumptions: Kernel Density Estimapte plot, Shapiro-Wilk Test, Normal Q-Q plot test Normal distribution Plot <br />
--Confidence Intervals for Regression Accuracy <br />
--Prediction Interval with 95%
## 2.5 Spatial Analysis
∆ spatial_analysis.py <br />
--Spatial Visualization <br />
--Convert to shapefile, raster, add geometry points, assign projection <br />
--Spatial Autocorrelation: spatial weights, Moran's I

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LNSOTOM/forestGISML/master?filepath=regressionModel%20(1).ipynb)
