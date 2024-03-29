# forestGISML: link to [Journal Article](https://www.sciencedirect.com/science/article/pii/S0168169923001928) (Computers and Electronics in Agriculture) and [Master Thesis](https://www.notion.so/Versions_ICTthesis-bfdecbc33eba4473895196fba35d7a38#01923cd692fa41d59b18c056232b070a)
![graphicalAbstract](https://user-images.githubusercontent.com/39131939/232376764-ff945451-77a3-46fa-9bbe-654ed4ee7170.png)

# 1 Overview [ Step 1: Scope ]
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
Timberlands Pacific) <br />
## 1.3 Data Dimension [ Step 2: Data Definition & Baseline ]
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
## 2.2 Machine Learning Algorithms (MLAs) [ Step 3: Modeling: train model ]
∆ linear_regression.py <br />
∆ polynomial_regression.py <br />
∆ decisionTree_regression.py <br />
∆ randomForest_regression.py <br />
∆ gradientBoostedDT_regression.py <br />
## 2.3 Exploratory Data Analysis (EDA)
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
## 2.5 Exploratory Spatial Data Analysis (ESDA)
∆ spatial_analysis.py <br />
∆ spatialAnalysis_beforeML.py <br />
∆ spatialAnalysis_afterML.py <br />
--Spatial Visualization: shapefile, raster, las files <br />
--Convert to shapefile, raster, add geometry points, assign projection <br />
--Terrain analysis (digital elevation model=DEM) <br />
--LiDAR (laz, las format) visualization with python <br />
--Spatial Autocorrelation: spatial weights, Moran's I

## 2.6 Performance and Model Validation [ Step 3: Modeling: perform error analysis ]
∆ validation_models.py <br />
--Summary of ouputs
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LNSOTOM/forestGISML/master?filepath=regressionModel%20(1).ipynb)
