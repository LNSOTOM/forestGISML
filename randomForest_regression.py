############################
# RANDOM FOREST REGRESSION MODEL
############################
# Reproduce the same scripts than Linear Regression (linear_regression.py)
"""##### 1 [ Split into training ] #####"""
"""##### 2 [ Extract train and test idx for later merge with geography coord ] #####"""

"""##### 3 [ Fit: RANDOM FOREST ] ######"""
## 3.1 Fit: Random Forest Model
###1)Import Model to use
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

###2) Make an instance of Random Forest Model: 100 estimators or  number of trees in the forest.
modelRForest = RandomForestRegressor(n_estimators=100, max_features = 'auto', random_state = 42, criterion='mse')

### 3) Train the Model 
modelRForest.fit(X_train, y_train)

### 3.1.1 Visualize the Tree Model
# Extract single tree
estimator = modelRForest.estimators_[4]

# Visualize Decision Tree
from sklearn import tree
from sklearn.tree import export_graphviz

# Creates dot file named tree.dot
tree.export_graphviz(estimator)

#Good
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,15), dpi=300)

tree.plot_tree(estimator,
               #feature_names = EDAsurvey.columns,  
               #class_names=str(y), 
               #filled = True,
               #fontsize=4);

fig.savefig('randomForestMODEL.png')

## 3.2 Predict Test Results
### 3.2.1 TEST: Make prediction using test set
y_pred = modelRForest.predict(X_test)
y_pred

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

#summary descriptive statistics
dataTest.describe()

### 3.2.2 TRAIN: Make prediction using TRAIN set
y_train_predicted = modelRForest.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

#summary descriptive statistics
dataTrain.describe()

### 3.2.3 Plot Goodness of fit for siteIndex values | Test set
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 6))

ax = sns.regplot(x="Actual", y="Predicted", data=dataTest, label='siteindex predicted', scatter_kws = {'color': 'white', 'alpha': 0.8, 'edgecolor':'blue', 's':10}, line_kws = {'color': '#f54a19'})
ax.set_ylim(0,55)
ax.set_xlim(0,55)
ax.plot([0, 55], [0, 55], 'k--', lw=2)

ax.legend(title="Test set:", frameon=  True, loc='upper left')
#ax.legend(bbox_to_anchor =(0.85, -0.20), ncol = 4) 
plt.title('Goodness-of-fit in Validation Set',fontsize=12)

plt.savefig('actualvsPredicted_RF_testSet.jpg', bbox_inches='tight', dpi=300)

"""##### 4 [ Perfomance and Validation ]#####"""
## 4.1 ACCURACY FOR TRAINING &TEST SET:
print("Accuracy on training set: {:.3f}".format(modelRForest.score(X_train, y_train)))
print("Accuracy on test set:: {:.3f}".format(modelRForest.score(X_test, y_test)))

#Accuracy on training set: 0.978
#Accuracy on test set:: 0.850

## 4.2 Accuracy Measures
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#R2 (explained variance) Train Set: 0.978
#R2 (explained variance) Test set: 0.850
#MAE=Mean Absolute Error: 1.4088428470463965
#MSE=Mean Squared Error: 4.336320852184177
#RMSE=Root Mean Squared Error: 2.0823834546461844

## 4.3 Calculate Squared Error
residSquare = np.square(dataTest['residuals'])
residSquare

### 4.3.1 Plot Squared Errror vs Observed
plt.style.use('seaborn-whitegrid')
fig=plt.figure(figsize = [8, 6])

ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, label='Squared Error', c='white', alpha=0.8, edgecolors='#1b346c', s=10)
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredError_RF.png', bbox_inches='tight', dpi=300)

fig=plt.figure(figsize = [8, 6])

ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Predicted'], y=residSquare, c='#f54a19', label='Squared Error')
ax.set_xlabel("Predicted 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Predicted 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredErrorPredicted_RF.png', bbox_inches='tight', dpi=300)

"""##### 5 [ Prediction Interval (Inference) ]#####"""
#check file [statistics.py]

"""##### 6 [ Evaluation: Explaining Feature Importance ]#####"""
## 6.1 Model Output: Feature Importance
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

featImp = pd.DataFrame({'feature':X_train.columns,'importance':np.round(modelRForest.feature_importances_,3)})

importances = modelRForest.feature_importances_

featImp = featImp.sort_values(by='importance', ascending=0)
featImp

## 6.2 Plot features
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
sns.set(style="whitegrid")
plt.subplot(1, 1, 1) # 1 row, 2 cols, subplot 1

ax = sns.barplot(featImp.feature, featImp.importance)

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=3), (p.get_x()+p.get_width()/2., p.get_height()),
                ha='left',
                va='baseline', 
                #textcoords='offset points',
                rotation='30')
                
#Rotate labels x-axis
plt.xticks(rotation=45, horizontalalignment='right')
plt.ylabel('Feature Importance')
plt.xlabel('Features')
plt.title("Impact of Features on the black-box model performance")

plt.savefig('FI_RF.png', bbox_inches='tight', dpi=300)

## 6.3 Permutation feature importance
!pip install eli5
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(modelRForest, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)

from sklearn.inspection import permutation_importance

r = permutation_importance(modelRForest, X_test, y_test,
                          n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
       print(f"{EDAsurvey.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
             f" +/- {r.importances_std[i]:.3f}")

"""##### 7 [ LIME: Local Interpretable Model-agnostic Explanations ]#####"""
# 0.LIME - Local Interpretable Model-Agnostic
import lime 
import lime.lime_tabular
import seaborn as sns

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["siteindex"],
    training_labels=y_train.values,  
    discretize_continuous=True,
    mode="regression",
)

#explained = explainer.explain_instance(featuresRobu_test[1], model.predict, num_features=25)
row = 42

exp = lime_explainer.explain_instance(X_test.iloc[row], modelRForest.predict, num_features=23)
 
exp.show_in_notebook(show_table=True)

# export LIME to html
exp.save_to_file('lime_RF.html')

# 1. visualize LIME plot
fig.set_size_inches(12, 12)

exp.as_pyplot_figure()

# explore dataframe
pd.DataFrame(exp.as_list())

# Commented out IPython magic to ensure Python compatibility.
# 2. plot LIME improved
# %matplotlib inline
#fig.set_size_inches(14, 20)
#fig = plt.figure(figsize=(14,14))
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

fig = exp.as_pyplot_figure()
plt.savefig('LIME_RF.jpg', bbox_inches='tight', dpi=300)

"""##### 8 [ Fit: Random Forest with Cross Validation ]#####"""
## 8.1 Pipeline-Used RF Model | kfold=10
### 8.1.1 Run Random Forest Model with cv=10
# 1. Run Ranfom Forest Model with cv=10
from sklearn.model_selection import cross_val_score

cv10 = cross_val_score(modelRForest, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(cv10))

#Cross-validation scores: [0.84907694 0.847113   0.85046789 0.85064072 0.84784503 0.85120825
 #0.85022507 0.84781513 0.84795191 0.85013774]
#time: 7h 59min 13s

### 8.1.2 Model Performance
# 2. Report Performance
from numpy import mean
from numpy import std

print("Average cross-validation score: {:.3f}".format(cv10.mean()))
print('MAE: %.4f (%.4f)' % (mean(cv10), std(cv10)))
print("Accuracy: %0.3f (+/- %0.3f)" % (cv10.mean(), cv10.std()))

#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy for 95perc confidence interval: %0.3f (+/- %0.3f)" % (cv10.mean(), cv10.std() * 2))

#outputs:
#Average cross-validation score: 0.849
#MAE: 0.8492 (0.0014)
#Accuracy: 0.849 (+/- 0.001)
#Accuracy for 95perc confidence interval: 0.849 (+/- 0.003)

### 8.1.3 Confidence Intervals for Regression Accuracy
# 3. Confidence Interval 95%
from math import sqrt
#The accuracy of the model was x +/- y at the 95% confidence level.
interval = 1.96 * sqrt( (0.849 * (1 - 0.849)) / 95356)
print('%.3f' % interval)
#OUTPUT: 0.002

##explanation: there is a 95% likehood that the X range to y covers the true model accuracy. 
## The accuracy of the model was +/- 0.002 y at the 95% confidence level
## The classification accuracy of the model is 84.9% +/- 0.2%
## The true classification accuracy of the model is likely between 84.7 and 85.1%.

#The accuracy of the model was x +/- y at the 95% confidence level.
interval = 1.96 * sqrt( (0.849 * (1 - 0.849)) / 1000)
print('%.3f' % interval)

#OUTPUT: 0.022
##the confidence interval drops 2.2% with a sample of 2%

### 8.1.4 Model Validation
# 4. Calculate boundaries for Boxplot | Model Accuracy
import statistics
from scipy import stats
# Median for predicted value
median = statistics.median(cv10)

q1, q2, q3= np.percentile(cv10,[25,50,75])
# IQR which is the difference between third and first quartile
iqr = q3 - q1

# lower_bound is 15.086 and upper bound is 43.249, so anything outside of 15.086 and 43.249 is an outlier.
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr) 

print('upper_bound: %.3f' % upper_bound)
print('Third quartile (q3): %.3f' % q3)
print('Median: %.3f' % median)
print('First quartile (q1): %.3f' % q1)
#print('Median (q2): %.3f' % q2)
print('IQR: %.3f' % iqr)
print('lower_bound: %.3f' % lower_bound)

#OUTPUTS:
#upper_bound: 0.854
#Third quartile (q3): 0.850
#Median: 0.850
#First quartile (q1): 0.848
#IQR: 0.003
#lower_bound: 0.844

# 4.1 plot performance
fig = plt.figure()
fig.suptitle('Model with 10-fold cross-validation')
ax = fig.add_subplot(111)
import matplotlib.pyplot as plt
plt.style.use('classic')

fig.set_size_inches(4, 4)

medianprops = dict(linewidth=1.5, linestyle='-', color='#fc3468')
meanprops =  dict(marker='D', markerfacecolor='indianred', markersize=4.5)

plt.gca().spines['right'].set_color('#D9D8D6')
plt.gca().spines['top'].set_color('#D9D8D6')
plt.gca().spines['left'].set_color('#D9D8D6')
plt.gca().spines['bottom'].set_color('#D9D8D6')

plt.grid(color='grey', linestyle='-', linewidth=0.25)

plt.boxplot(cv10, medianprops=medianprops, meanprops=meanprops, showmeans=True )
ax.set_xticklabels('')
plt.xlabel('Random Forest Regressor')
plt.ylabel('Accuracy Model')
plt.savefig('accuracy_RF.png', bbox_inches='tight', dpi=300)

## 8.2 Pipeline-Used GridSearchCV | kfold=10
### 8.2.1 Run GridSearchCV with cross validation
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline

# create the pre-processing component
#scaler = MinMaxScaler()
    
    # define classifiers
    ## Classifier : Random Forest Classifier
modelRF =  RandomForestRegressor(n_estimators=100, random_state = 42)

params = {
    "criterion":["mse"], 
    "max_features": ['auto'],
}

pipe = Pipeline([('rf_model', modelRF)])

grid_cv = GridSearchCV(modelRF, params)
grid_cv

### 8.2.2 Model Performance
from sklearn.model_selection import cross_val_score
RFcv10 = cross_val_score(modelRForest, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(RFcv10))

#grid_cv.fit(X_train, y_train)  [cv=5]
#ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(grid_cv.score(X_train, y_train)))
#ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(grid_cv.score(X_test, y_test)))

## 8.3 Predict Test Results
### 8.3.1 TEST: Make prediction using test set
predictedNorm = grid_cv.predict(X_test)

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': predictedNorm})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

# sum stats
dataTest.describe()

### 8.3.2 TRAIN: Make prediction using TRAIN set
y_train_predicted = grid_cv.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

### 8.3.3 Plot Predicted vs Observed | Test Set
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))

ax = sns.regplot(x="Actual", y="Predicted", data=dataTest, label='siteindex predicted')
ax.set_ylim(0,55)
ax.set_xlim(0,55)
ax.plot([0, 55], [0, 55], 'k--', lw=2)

ax.legend(title="Test set:", frameon=  True)
#ax.legend(bbox_to_anchor =(0.85, -0.20), ncol = 4) 
plt.title('Features Predicted siteindex (m) in Test Set',fontsize=12)

plt.savefig('actualvsPredictedRFModel_testSet.jpg', bbox_inches='tight', dpi=300)

### 8.3.4 Plot Goodness of fit for siteIndex values | Test set
import numpy as np   # To perform calculations
import matplotlib.pyplot as plt  # To visualize data and regression line
from pylab import rcParams
import seaborn as sns
sns.set(style="whitegrid")

dfTest = dataTest.head(25)
dfTest.plot(kind='bar', figsize=(12,8))

#plt.legend(title="Test set",loc='upper center', bbox_to_anchor=(1.10, 0.8), frameon=False)
plt.legend(title="Test set", frameon=  True)

plt.title('Actual vs Predicted \'siteindex\' Values in Test Set' )
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.xticks(rotation=45, horizontalalignment='right')

plt.savefig('actualvsPredicted_testSet_RF.jpg', bbox_inches='tight', dpi=300)

## 8.4 Performance and Validation
#EVALUATE MODEL
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test Set: {:.3f}".format(metrics.r2_score(y_test, predictedNorm), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictedNorm))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, predictedNorm))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictedNorm)))

"""##### 9 [ Evaluation: Explaining Feature Importance] #####"""
## 9.1 Model Output: Permutation Importance
from sklearn.inspection import permutation_importance
r = permutation_importance(grid_cv, X_test, y_test,
                          n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
       print(f"{EDAsurvey.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
             f" +/- {r.importances_std[i]:.3f}")

grid_cv.best_estimator_.feature_importances_

# create dataframe
d = {'Stats':X.columns,'FI':grid_cv.best_estimator_.feature_importances_}
df = pd.DataFrame(d)
df

# sort values
df = df.sort_values(by='FI', ascending=0)
df

# Plot 1 Bar
ax = df.plot.bar(x='Stats', y='FI', rot=45, width = 0.8)

## 9.2 Plot features
# Commented out IPython magic to ensure Python compatibility.
# Plot 2
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
fig.set_size_inches(14, 10)
sns.set(style="whitegrid")

ax = df.plot.bar(x='Stats', y='FI', alpha=0.9)

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()),
                ha='left',
                va='baseline', 
                #textcoords='offset points',
                rotation='30')

plt.title("Random Forest Features Importance towards dependent variable: 'siteindex' (y)")
plt.ylabel('feature importance', fontsize=12)
plt.xlabel('independent variables (x)', fontsize=12)
plt.xticks(rotation=45, horizontalalignment='right')

plt.savefig('randomForest_FI.png', bbox_inches='tight', dpi=300)

# Plot 3
import plotly.express as px
fig = px.bar_polar(df, r="FI", theta="Stats",
                   color="Stats", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()

#Plot 4 GOOD
#column_names = ['aspect','planCurvature', 'profileCurvature','slope','TPI','TWI_SAGA','Dh_diffuse','Ih_direct','DEM','meanJanRain','meanJulRain','maxJanTemp','minJulTemp','SYMBOL','soil_order','BDw','CLY', 'CFG','ECD','SOC','pHw','SND','SLT']

regression_coefficient = pd.DataFrame(df.sort_values(by='FI', ascending=0))

plt.figure(figsize=(14,8))
g = sns.barplot(x=df.Stats, y=df.FI, data=regression_coefficient, capsize=0.3, palette='spring')
g.set_title("Contribution of features towards dependent variable: 'siteindex' (y)", fontsize=15)
g.set_xlabel("independent variables (x)", fontsize=13)
g.set_ylabel("slope of coefficients (m)", fontsize=13)
plt.xticks(rotation=45, horizontalalignment='right')
g.set_yticks([-1, 0, 1])
g.set_xticklabels(column_names)
for p in g.patches:
    g.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
               textcoords='offset points', fontsize=14, color='black')
    
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.savefig('randomForestModel_10test.png', bbox_inches='tight', dpi=300)

## 9.3 Calculate Squared Error
residSquare = np.square(dataTest['residuals'])
residSquare

### 9.3.1 Plot Squared Errror vs Observed
fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, c='#1b346c', label='Squared Error')
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('Squared Error_RF.png', bbox_inches='tight', dpi=300)

fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Predicted'], y=residSquare, c='#f54a19', label='Squared Error')
ax.set_xlabel("Predicted 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Predicted 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredErrorPredicted_RF.png', bbox_inches='tight', dpi=300)

"""##### 10 [ Spatial Visualization for Predictions] #####"""
#check file [spatialAnalysis_afterML.py]

"""##### 11 [ Regression Assumptions ] #####"""
error = dataTest['Actual'] - dataTest['Predicted']
#error = y_test - predictedStand
#error_info = pd.DataFrame({'y_true': y_test, 'y_pred': predictedStand, 'error': error}, columns=['y_true', 'y_pred', 'error'])

error_info = pd.DataFrame({'y_true': dataTest['Actual'], 'y_pred': dataTest['Predicted'], 'error': error}, columns=['y_true', 'y_pred', 'error'])

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import numpy as np
import matplotlib.pyplot as plt

stats.probplot(error_info.error, dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles | Interpretation: standard deviations", labelpad=15)
plt.title("Probability Plot to Compare Normal Distribution Values to\n Perfectly Normal Distribution", y=1.015)

plt.savefig('probabilityPlot_RF.jpg', bbox_inches='tight', dpi=300)

plt.figure(figsize = [6, 4]) # larger figure size for subplots

# Density Plot and Histogram of all A results
plt.subplot(1, 1, 1) # 1 row, 2 cols, subplot 1
sns.distplot(error_info.error, hist=True, kde=True, 
             bins=int(180/10), color = '#5f90d8', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})

# Plot formatting for  A
plt.legend()
plt.xlabel('Errors')
plt.ylabel('Normalized Errors (density)')
plt.title('Normal and Density Distribution of Errors')

plt.savefig('densityPlotHist_RF.jpg', bbox_inches='tight', dpi=300)

res = error_info.error
fig = sm.qqplot(res, line='s')
plt.show()
