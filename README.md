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
The SI was sourced by the forestry company [Timberlands Pacific](https://www.tppl.com.au/). Supervisors are: [Dr. Matthew J Cracknell](https://www.utas.edu.au/profiles/staff/codes/matthew-cracknell) (Postdoctoral
Research Fellow in Earth Informatics at the ARC Industrial Transformation Research Hub for
Transforming the Mining Value Chain), Dr. Robert Musk (Data Analyst/Forest Scientist at
Timberlands Pacific) and Dr. Muhammad Bilal Amin (Computing & Information Systems, [University of Tasmania](https://www.utas.edu.au/technology-environments-design/ict)).
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
∆ randomForests_regression.py <br />
∆ gradientBoosting_regression.py <br />
∆ deepLearning.py <br />
## 2.3 Statistics and Exploration Data Analysis
## 2.4 Spatial Analysis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LNSOTOM/forestGISML/master?filepath=regressionModel%20(1).ipynb)
