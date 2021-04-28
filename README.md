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
The SI was sourced by the forestry company Timberlands Pacific. My supervisors are: Dr. Muhammad Bilal Amin
(Computing & Information Systems, University of Tasmania), Dr. Matthew J Cracknell (Postdoctoral
Research Fellow in Earth Informatics at the ARC Industrial Transformation Research Hub for
Transforming the Mining Value Chain) and Dr. Robert Musk (Data Analyst/Forest Scientist at
Timberlands Pacific).
For data collection and processing, GIS processing techniques has been applying to
extract all the attributes and achieved more accurate predictions.
## 1.3 Data Dimension
The survey selected for analysis was collected from five sites with wide variation in landscape
conditioning and climate factors, and a diverse geology and soil attributes related to the area that
influence the productivity of radiata pine across the estate. The data used for the modelling is
comprised by 23 datasets and 953556 observations
## 1.4 Contribution
This research will contribute by developing a novel ensemble learning base
technique which will produce predictive models to assist a forest manager in
optimal resource utilization with maximization of productivity.

# 2 Scripts Guideline
∆ linear_regression.py <br />
∆ polynomial_regression.py <br />
∆ decisionTree_regression.py <br />
∆ randomForests_regression.py <br />
∆ gradientBoosted_regression.py <br />
∆ deepLearning.py <br />

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LNSOTOM/forestGISML/master?filepath=regressionModel%20(1).ipynb)
