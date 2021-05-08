############################
# DECISION TREES REGRESSION MODEL
############################
# Reproduce the same scripts than Linear Regression (linear_regression.py)
"""##### 1 [ Split into training ] #####"""
"""##### 2 [ Extract train and test idx for later merge with geography coord ] #####"""

"""##### 3 [ Fit: DECISION TREE REGRESSOR ] ######"""
## 3.1 Fit Model: Decision Tree Regression
###1)Import Model to use
from sklearn.tree import DecisionTreeRegressor

###2)Make an instance of Decision Tree Model: try different depht: 15,20,25
modelDTree = DecisionTreeRegressor(max_features = 'auto', random_state = 42, criterion='friedman_mse')

###3)Train the Model
modelDTree.fit(X_train, y_train)

### 3.1.1 Visualize Decision Tree Model
# Visualize Decision Tree
from sklearn import tree
from sklearn.tree import export_graphviz

# Creates dot file named tree.dot
tree.export_graphviz(modelDTree)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
fig = plt.figure()
# Visualize Decision Tree
from sklearn import tree
from sklearn.tree import export_graphviz

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,20), dpi=300)

tree.plot_tree(modelDTree,
               feature_names = X_list,  
               class_names = EDAsurvey['siteindex'],  
               filled = True,
               impurity=False,
               rounded=True,             
               fontsize=10);

fig.savefig('decisionTreeMODEL_1.png')

## 3.2 Predict Test Results
### 3.2.1 TEST: Make prediction using test set
y_pred = modelDTree.predict(X_test)
y_pred

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

#summary descriptive statistics
dataTest.describe()

### 3.2.2 TRAIN: Make prediction using TRAIN set
y_train_predicted = modelDTree.predict(X_train)
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

plt.savefig('actualvsPredicted_DTree_testSet.jpg', bbox_inches='tight', dpi=300)

"""##### 4 [ Perfomance and Validation #####"""
## 4.1 ACCURACY FOR TRAINING & TEST SET:
print("Accuracy on training set: {:.3f}".format(modelDTree.score(X_train, y_train)))
print("Accuracy on test set:: {:.3f}".format(modelDTree.score(X_test, y_test)))

## 4.2 Accuracy Measures
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

## 4.3 Calculate Squared Error
residSquare = np.square(dataTest['residuals'])
residSquare

### 4.3.1 Plot Squared Errror vs Observed
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig=plt.figure(figsize = [8, 6])

ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, label='Squared Error', c='white', alpha=0.8, edgecolors='#1b346c', s=10)
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredError_Dtree.png', bbox_inches='tight', dpi=300)

fig=plt.figure(figsize = [8, 6])

ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Predicted'], y=residSquare, c='#f54a19', label='Squared Error')
ax.set_xlabel("Predicted 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Predicted 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredErrorPredicted_Dtree.png', bbox_inches='tight', dpi=300)

"""##### 5 [ Prediction Interval (Inference) ] #####"""
#check file [statistics.py]

"""##### 6 Evaluation: Explaining Feature Importance #####"""
## 6.1 Model Output: Feature Importance
featImp = pd.DataFrame({'feature':X_train.columns,'importance':np.round(modelDTree.feature_importances_,3)})

importances = modelDTree.feature_importances_

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

plt.savefig('FI_Dtree.png', bbox_inches='tight', dpi=300)

## 6.3 Permutation feature importance
!pip install eli5
import eli5
from eli5.sklearn import PermutationImportance

model = modelDTree.fit(X_train, y_train)
perm = PermutationImportance(model).fit(X_test, y_test)
eli5.show_weights(perm)

perm(X_train.columns, perm.feature_importances_)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(modelDTree, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)

from sklearn.inspection import permutation_importance

r = permutation_importance(modelDTree, X_test, y_test,
                          n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
       print(f"{EDAsurvey.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
             f" +/- {r.importances_std[i]:.3f}")

"""##### 7 LIME: Local Interpretable Model-agnostic Explanations #####"""
# 0.LIME - Local Interpretable Model-Agnostic
import lime 
import lime.lime_tabular
import seaborn as sns

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=y_train.values,
    feature_names=X_train.columns.tolist(),
    #feature_selection="lasso_path",
    class_names=["siteindex"],
    discretize_continuous=True,
    mode="regression",
)

#explained = explainer.explain_instance(featuresRobu_test[1], model.predict, num_features=25)
row = 42

exp = lime_explainer.explain_instance(X_test.iloc[row], modelDTree.predict, num_features=23)
 
exp.show_in_notebook(show_table=True)

# export LIME to html
exp.save_to_file('lime_DTree.html')

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
plt.savefig('LIME_DTree.jpg', bbox_inches='tight', dpi=300)

"""##### 8 [ Fit: Decision Tree with Cross Validation ] #####"""
## 8.1 Pipeline with cv=10
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# create the pre-processing component
#scaler = MinMaxScaler()
    
    # define classifiers
    ## Classifier : Random Forest Classifier
modelDTreeK = DecisionTreeRegressor(#max_depth=5, max_features = 'auto', criterion='mse', random_state=42 )
    max_features = 'auto', random_state = 42, criterion='friedman_mse' )
    
    # define pipeline 
    ## clf_RF
pipe = Pipeline([('rf_model', modelDTreeK)])

params = {
    "splitter":("best", "random"), 
}

grid_cv = GridSearchCV(modelDTreeK, params, n_jobs=-1, verbose=1, cv=10)
grid_cv

grid_cv.fit(X_train, y_train)

## 8.2 Predict Test Results
# TEST: Make prediction using test set
predictedNorm = grid_cv.predict(X_test)

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': predictedNorm})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

#summary descriptive statistics
dataTest.describe()

# TRAIN: Make prediction using TRAIN set
y_train_predicted = grid_cv.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

### 6.2.1 Plot Predicted vs Observed | Test Set
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

plt.savefig('actualvsPredictedmodelDTreeK_testSet.jpg', bbox_inches='tight', dpi=300)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))

ax = sns.regplot(x="Actual", y="Predicted", data=dataTest, label='siteindex predicted', scatter_kws = {'color': 'orange', 'alpha': 0.3}, line_kws = {'color': '#f54a19'})
ax.set_ylim(0,55)
ax.set_xlim(0,55)
ax.plot([0, 55], [0, 55], 'k--', lw=2)

ax.legend(title="Test set:", frameon=  True)
#ax.legend(bbox_to_anchor =(0.85, -0.20), ncol = 4) 
plt.title('Features Predicted siteindex (m) in Test Set',fontsize=12)

plt.savefig('actualvsPredicted_DTreeK_testSet.jpg', bbox_inches='tight', dpi=300)

## 8.3 Performance and Validation
### 8.3.1 Option a)
from sklearn.model_selection import cross_val_score
cv10 = cross_val_score(modelDTree, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(cv10))

# evaluate adaboost algorithm for regressor
from numpy import mean
from numpy import std
# 2. report performance
print("Average cross-validation score: {:.3f}".format(cv10.mean()))
print('MAE: %.3f (%.3f)' % (mean(cv10), std(cv10)))
print("Accuracy: %0.3f (+/- %0.3f)" % (cv10.mean(), cv10.std()))

#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy for 95perc confidence interval: %0.3f (+/- %0.3f)" % (cv10.mean(), cv10.std() * 2))

#Average cross-validation score: 0.700
#MAE: 0.700 (0.004)
#Accuracy: 0.700 (+/- 0.004)
#Accuracy for 95perc confidence interval: 0.700 (+/- 0.009)

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

# 3. plot performance
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
plt.xlabel('Decision Tree Regressor')
plt.ylabel('Accuracy Model')
plt.savefig('accuracy_DTree.png', bbox_inches='tight', dpi=300)

sns.set(style="whitegrid")
fig = plt.figure()

fig.suptitle('Model with 10-fold cross-validation')
ax = fig.add_subplot(111)
import seaborn as sns
sns.set(style="whitegrid")

plt.boxplot(cv10)
ax.set_xticklabels('')
plt.xlabel('Decision Tree Regressor')
plt.ylabel('Accuracy Model')
plt.savefig('accuracy_DTree.png', bbox_inches='tight', dpi=300)

### 8.3.2 Option b)
from sklearn.model_selection import cross_val_score
cv10 = cross_val_score(grid_cv, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(cv10))

#ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(grid_cv.score(X_train, y_train)))
#ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(grid_cv.score(X_test, y_test)))

#EVALUATE MODEL
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test Set: {:.3f}".format(metrics.r2_score(y_test, predictedNorm), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictedNorm))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, predictedNorm))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictedNorm)))

### 6.3.3 Plot Squared Errror vs Observed
residSquare = np.square(dataTest['residuals'])
residSquare

fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, c='#1b346c', label='Squared Error')
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)

plt.savefig('SquaredError_DTreeK.png', bbox_inches='tight', dpi=300)

fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Predicted'], y=residSquare, c='#f54a19', label='Squared Error')
ax.set_xlabel("Predicted 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Predicted 'site index' values")
plt.legend(title="",loc='upper left', frameon=True)
plt.savefig('SquaredErrorPredicted_DTreeK.png', bbox_inches='tight', dpi=300)

best_result = grid_cv.best_score_
print(best_result)

"""##### 9 [ Spatial Visualization for Predictions ] #####"""
#check file [spatialAnalysis_afterML.py]

"""##### 10 [ Regression Assumptions ] #####"""
error = dataTest['Actual'] - dataTest['Predicted']
#error = y_test - predictedStand
#error_info = pd.DataFrame({'y_true': y_test, 'y_pred': predictedStand, 'error': error}, columns=['y_true', 'y_pred', 'error'])

error_info = pd.DataFrame({'y_true': dataTest['Actual'], 'y_pred': dataTest['Predicted'], 'error': error}, columns=['y_true', 'y_pred', 'error'])

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

plt.savefig('densityPlotHist_dtree.jpg', bbox_inches='tight', dpi=300)

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import numpy as np
import matplotlib.pyplot as plt

stats.probplot(error_info.error, dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles | Interpretation: standard deviations", labelpad=15)
plt.title("Probability Plot to Compare Normal Distribution Values to\n Perfectly Normal Distribution", y=1.015)

plt.savefig('probabilityPlot_Dtree.jpg', bbox_inches='tight', dpi=300)
