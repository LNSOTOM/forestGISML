############################
# GRADIENT BOOSTED DECISION TREE (REGRESSION) MODEL
############################
# Reproduce the same scripts than Linear Regression (linear_regression.py)
"""##### 1 [ Split into training ] #####"""
"""##### 2 [ Extract train and test idx for later merge with geography coord ] #####"""

"""##### 3 [ Fit: GRADIENT BOOSTED REGRESSOR ] ######"""
## 3.0 Fit: Base Gradient Boosted Model
###1)Import Model to use
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

# as default=’friedman_mse’
## try criterion: 'mse'
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

params = {'n_estimators': 100,
          'max_depth': 16,
          #'min_samples_split': 5,
          'criterion': 'mse',
          'random_state' : 42,
          'max_features':'auto'}

gradBoost = ensemble.GradientBoostingRegressor(**params)

## 3.1 Tuning parameters
### 3.1.1 max_depht=15
#max_depth=15
gradBoost.fit(X_train, y_train) 

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.937
#Accuracy on test set:: 0.842

### 3.1.2 max_depht=16
#max_depth=16
gradBoost.fit(X_train, y_train) 

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.956
#Accuracy on test set:: 0.845
#time: 1h 25min 51s

### 3.1.3 max_depht=20
#max_depth=20
gradBoost.fit(X_train, y_train) 

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.993
#Accuracy on test set:: 0.843
#1h 41min 29s

## 3.2 Validation error at each stage of training to 
find the optimal number of trees
### 3.2.1 Errors from Model: max_depth=15
#DEPTH=15
from sklearn.metrics import mean_squared_error

errors = [mean_squared_error(y_test, y_pred) for y_pred in gradBoost.staged_predict(X_test)]
errors

#plt.figure(figsize=(8,6))
import matplotlib.pyplot as plt
plt.style.use('classic')
fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax = plt.plot(errors, color='blue', marker='.', markerfacecolor='#b2b2ff', markersize=8)
plt.xlabel('Number of Trees (n_estimators=100)')
plt.ylabel('Error')

plt.grid(color='grey', linestyle='-', linewidth=0.25)
#removing top and right borders
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_color('none')

plt.title(" Validation Error" )
plt.savefig('errors_GBDT.jpg', bbox_inches='tight', dpi=300)

### 3.2.2 Errors from Model: max_depth=16"""
#DEPTH=16
from sklearn.metrics import mean_squared_error

errors = [mean_squared_error(y_test, y_pred) for y_pred in gradBoost.staged_predict(X_test)]
errors

#plt.figure(figsize=(8,6))
import matplotlib.pyplot as plt
plt.style.use('classic')
fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax = plt.plot(errors, color='blue', marker='.', markerfacecolor='#b2b2ff', markersize=8)
plt.xlabel('Number of Trees (n_estimators=100)')
plt.ylabel('Error')

plt.grid(color='grey', linestyle='-', linewidth=0.25)
#removing top and right borders
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_color('none')

plt.title(" Validation Error" )
plt.savefig('errors_GBDT.jpg', bbox_inches='tight', dpi=300)

### 3.2.3 Errors from Model: max_depth=20"""
#DEPTH=20
from sklearn.metrics import mean_squared_error

errors = [mean_squared_error(y_test, y_pred) for y_pred in gradBoost.staged_predict(X_test)]
errors

#plt.figure(figsize=(8,6))
import matplotlib.pyplot as plt
plt.style.use('classic')
fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax = plt.plot(errors, color='blue', marker='.', markerfacecolor='#b2b2ff', markersize=8)
plt.xlabel('Number of Trees (n_estimators=100)')
plt.ylabel('Error')

plt.grid(color='grey', linestyle='-', linewidth=0.25)
#removing top and right borders
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_color('none')

plt.title(" Validation Error" )
plt.savefig('errors_GBDT.jpg', bbox_inches='tight', dpi=300)

## 3.3 predict with best estimatro found for each depht
### 3.3.1 max_depht=15 with best estimator + Performance/validation
bst_n_estimators = np.argmin(errors)
print (bst_n_estimators)
#98

from sklearn.ensemble import GradientBoostingRegressor
#depth=15
gradBoost_best = GradientBoostingRegressor(n_estimators=98, max_depth=15, random_state = 42, max_features = 'auto', criterion= 'mse')

gradBoost_best.fit(X_train, y_train)

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost_best.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost_best.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.937
#Accuracy on test set:: 0.842

#DEPTH=15
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

##OUTPUT
#R2 (explained variance) Train Set: 0.937
#R2 (explained variance) Test set: 0.842
#MAE=Mean Absolute Error: 1.4925601040603598
#MSE=Mean Squared Error: 4.631020417157663
#RMSE=Root Mean Squared Error: 2.1519805801070007

### 3.3.2 max_depht=16 with best estimator + Performance/validation"""
bst_n_estimators = np.argmin(errors)
print (bst_n_estimators)
#99 ---- max=16

from sklearn.ensemble import GradientBoostingRegressor
#depth=16
gradBoost_best = GradientBoostingRegressor(n_estimators=bst_n_estimators, max_depth=16, random_state = 42, max_features = 'auto', criterion= 'mse')
#gradBoost_best = GradientBoostingRegressor(n_estimators=90, max_depth=20, random_state = 42, max_features = 'auto', criterion= 'mse')

gradBoost_best.fit(X_train, y_train)

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost_best.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost_best.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.956
#Accuracy on test set:: 0.845

### 3.3.3 max_depht=20 with best estimator + Performance/validation"""
bst_n_estimators = np.argmin(errors)
print (bst_n_estimators)
#90 --- max=20

from sklearn.ensemble import GradientBoostingRegressor
#depth=20
gradBoost_best = GradientBoostingRegressor(n_estimators=bst_n_estimators, max_depth=20, random_state = 42, max_features = 'auto', criterion= 'mse')
#gradBoost_best = GradientBoostingRegressor(n_estimators=90, max_depth=20, random_state = 42, max_features = 'auto', criterion= 'mse')

gradBoost_best.fit(X_train, y_train)

# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost_best.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost_best.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.993
#Accuracy on test set:: 0.843

"""#####  4 [ Predict Test Results ] #####"""
## 4.1 TEST: Make prediction using test set
# TEST: Make prediction using TEST set
y_pred = gradBoost_best.predict(X_test)
y_pred

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

dataTest.describe()

## 4.2 TRAIN: Make prediction using TRAIN set"""
# TRAIN: Make prediction using TRAIN set
y_train_predicted = gradBoost_best.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

dataTrain.describe()

## 4.3 Plot Goodness of fit for siteIndex values | Test set"""
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 6))

ax = sns.regplot(x="Actual", y="Predicted", data=dataTest, label='siteindex predicted', scatter_kws = {'color': 'white', 'alpha': 0.8, 'edgecolor':'blue', 's':10}, line_kws = {'color': '#f54a19'})
ax.set_ylim(0,55)
ax.set_xlim(0,55)
ax.plot([0, 55], [0, 55], 'k--', lw=2)

ax.legend(title="Test set:", frameon=  True, loc='upper left')
#ax.legend(bbox_to_anchor =(0.85, -0.20), ncol = 4) 
plt.title('Goodness-of-fit in Validation Set',fontsize=12)

plt.savefig('actualvsPredicted_GBDT_testSet.jpg', bbox_inches='tight', dpi=300)

"""##### 5 [ Prediction Interval (Inference) ] #####"""
#check file [statistics.py]

"""##### 6 [ Perfomance and Validation ] #####"""
## 6.1 Accuracy Measures: max depth=16
# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost_best.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost_best.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.956
#Accuracy on test set:: 0.845

#DEPTH=20
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error: {:.3f}'.format(metrics.mean_absolute_error(y_test, y_pred)))  
print('MSE=Mean Squared Error: {:.3f}'.format(metrics.mean_squared_error(y_test, y_pred)))  
print('RMSE=Root Mean Squared Error: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

##OUTPUT
#R2 (explained variance) Train Set: 0.956
#R2 (explained variance) Test set: 0.845
#MAE=Mean Absolute Error: 1.462
#MSE=Mean Squared Error: 4.522
#RMSE=Root Mean Squared Error: 2.126

## 6.2 Accuracy Measures: max depth=20"""
# ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(gradBoost_best.score(X_train, y_train)))

# ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(gradBoost_best.score(X_test, y_test)))

##OUTPUT
#Accuracy on training set: 0.993
#Accuracy on test set:: 0.848

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, gradBoost_best.predict(X_test))
print("The mean squared error (MSE) on test set: {:.3f}".format(mse))

##OUTPUT
#The mean squared error (MSE) on test set: 4.471

#DEPTH=20
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error: {:.3f}'.format(metrics.mean_absolute_error(y_test, y_pred)))  
print('MSE=Mean Squared Error: {:.3f}'.format(metrics.mean_squared_error(y_test, y_pred)))  
print('RMSE=Root Mean Squared Error: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

##OUTPUT
#R2 (explained variance) Train Set: 0.993
#R2 (explained variance) Test set: 0.848
#MAE=Mean Absolute Error: 1.437
#MSE=Mean Squared Error: 4.471
#RMSE=Root Mean Squared Error: 2.115

# evaluate adaboost algorithm for regressor
from numpy import mean
from numpy import std
# 2. report performance
#print("Average cross-validation score: {:.3f}".format(gradBoost_best.mean()))
print('MAE: %.3f (%.3f)' % (mean(gradBoost_best), std(gradBoost_best)))
print("Accuracy: %0.3f (+/- %0.3f)" % (gradBoost_best.mean(), gradBoost_best.std()))

#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy for 95perc confidence interval: %0.3f (+/- %0.3f)" % (cv10.mean(), cv10.std() * 2))

#Average cross-validation score: 0.700
#MAE: 0.700 (0.004)
#Accuracy: 0.700 (+/- 0.004)
#Accuracy for 95perc confidence interval: 0.700 (+/- 0.009)

## 6.3 Calculate Squared Error
residSquare = np.square(dataTest['residuals'])
residSquare

### 6.3.1 Plot Squared Errror vs Observed
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, label='Squared Error', c='white', alpha=0.8, edgecolors='#1b346c', s=10)
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')
plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredError_GBDT.png', bbox_inches='tight', dpi=300)

fig=plt.figure(figsize = [8, 6])
ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Predicted'], y=residSquare, c='#f54a19', label='Squared Error')
ax.set_xlabel("Predicted 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Predicted 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredErrorPredicted_GBDT.png', bbox_inches='tight', dpi=300)

"""##### 7 [ Evaluation: Explaining Feature Importance ] #####"""
## 7.1 Model Output: Feature Importance
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
featImp = pd.DataFrame({'feature':X_train.columns,'importance':np.round(gradBoost_best.feature_importances_,3)})

importances = gradBoost_best.feature_importances_

featImp = featImp.sort_values(by='importance', ascending=0)
featImp

## 7.2 Plot features"""
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
plt.ylabel('Feaceure Importance')
plt.xlabel('Features')
plt.title("Impact of Features on the black-box model performance")


plt.savefig('FI_GBDT.png', bbox_inches='tight', dpi=300)

## 7.3 Permutation feature importance"""
from sklearn.inspection import permutation_importance

r = permutation_importance(gradBoost, X_test, y_test,
                          n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
       print(f"{EDAsurvey.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
             f" +/- {r.importances_std[i]:.3f}")

"""##### 8 [ LIME: Local Interpretable Model-agnostic Explanations ] ######"""
#LIME - Local Interpretable Model-Agnostic=16 max
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

exp = lime_explainer.explain_instance(X_test.iloc[row], gradBoost_best.predict, num_features=23)
 
exp.show_in_notebook(show_table=True)

#LIME - Local Interpretable Model-Agnostic
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

exp = lime_explainer.explain_instance(X_test.iloc[row], gradBoost_best.predict, num_features=23)
 
#exp.show_in_notebook(show_table=True)

print(exp)

# export LIME to html
exp.save_to_file('lime_GBDT.html')

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
plt.savefig('LIME_GBDT.jpg', bbox_inches='tight', dpi=300)

"""##### 9 [ Fit: GBDT with Cross Validation ] #####"""
params = {'n_estimators': 99,
          'max_depth': 20,
          #'min_samples_split': 5,
          'criterion': 'mse',
          'random_state' : 42,
          'max_features':'auto'}

gradBoostCV = ensemble.GradientBoostingRegressor(**params)

#n_estimators=99, max_depth=20, random_state = 42, max_features = 'auto', criterion= 'mse'

from sklearn.ensemble import GradientBoostingRegressor

gradBoost_best = GradientBoostingRegressor(n_estimators=99, max_depth=15, random_state = 42, max_features = 'auto', criterion= 'mse')

# 1. Run Gradient Boosted Model with cv=10
from sklearn.model_selection import cross_val_score

GBDTcv10 = cross_val_score(gradBoost_best, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(GBDTcv10))

## 9.1 Pipeline-Used GBDT Model | kfold=10
### 9.1.1 Run GBDT Model with cv=10
from sklearn.model_selection import cross_val_score

cv10 = cross_val_score(gradBoost_best, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(cv10))

## 9.2 Pipeline-Used GridSearchCV | kfold=10
### 9.2.1 Run GridSearchCV with cross validation
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
gradBoostK = GradientBoostingRegressor(n_estimators=100, criterion='friedman_mse' )
    
    # define pipeline 
    ## clf_RF
pipe = Pipeline([('rf_model', gradBoostK)])

params = {
    "splitter":("best", "random"), 
}

grid_cv = GridSearchCV(gradBoostK, params, n_jobs=-1, verbose=1, cv=10)
grid_cv

grid_cv.fit(X_train, y_train)

## 9.3 Predict Test Results
### 9.3.1 TEST: Make prediction using test set
# TEST: Make prediction using test set
predictedNorm = grid_cv.predict(X_test)

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': predictedNorm})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

dataTest.describe()

### 9.3.2 TRAIN: Make prediction using TRAIN set
# TRAIN: Make prediction using TRAIN set
y_train_predicted = grid_cv.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

### 9.3.3 Plot Predicted vs Observed | Test Set
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

### 9.3.4 Plot Goodness of fit for siteIndex values | Test set"""
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

"""##### 10 [ Performance and Validation ] #####"""
## 10.1 ACCURACY FOR TRAINING &TEST SET:
#depth=20
#ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(grid_cv.score(X_train, y_train)))
#ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(grid_cv.score(X_test, y_test)))

## 10.2 Accuracy Measures
#EVALUATE MODEL
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test Set: {:.3f}".format(metrics.r2_score(y_test, predictedNorm), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictedNorm))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, predictedNorm))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictedNorm)))

## 10.3 Calculate Squared Error
residSquare = np.square(dataTest['residuals'])
residSquare

### 10.3.1 Plot Squared Errror vs Observed
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

"""##### 11 [ Spatial Visualization for Predictions ] #####"""
#check file [spatialAnalysis_afterML.py]

"""##### 12 [ Regression Assumptions ] #####"""
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

plt.savefig('densityPlotHist_GBDT.jpg', bbox_inches='tight', dpi=300)

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import numpy as np
import matplotlib.pyplot as plt

stats.probplot(error_info.error, dist="norm", fit=True, rvalue=True, plot=plt)
plt.xlabel("Theoretical quantiles | Interpretation: standard deviations", labelpad=15)
plt.title("Probability Plot to Compare Normal Distribution Values to\n Perfectly Normal Distribution", y=1.015)

plt.savefig('probabilityPlot_GBDT.jpg', bbox_inches='tight', dpi=300)