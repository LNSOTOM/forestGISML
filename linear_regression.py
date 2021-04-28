############################
# LINEAR MODEL
############################
"""##### 1 [ Split into training ] #####"""
## 1.1 Split into training: test=10%

# A_Target variable: Labels are the values we want to predict
X = EDAsurvey.drop('siteindex', axis = 1)

# Saving feature names for later use
X_list = list(EDAsurvey.columns)

# B_Independent variables: features are the values that help to predict
y = EDAsurvey['siteindex']#.values.reshape(-1,1)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', X_test.shape)
print('Testing Features Shape:', y_train.shape)
print('Testing Labels Shape:', y_test.shape)

EDAsurvey.head(

"""##### 2 [ Extract train and test idx for later merge with geography coord ] #####"""
## 2.1 Extracting train and test idx for later merge with additional data or geography coordinates
test_idx=np.asarray(X_test.index)
train_idx=np.asarray(X_train.index)

X_test_coord=EDAsurvey[[ 'x', 'y', 'siteindex']].iloc[test_idx]
X_test_coord.reset_index(inplace=True,drop=True)

X_train_coord=EDAsurvey[[ 'x', 'y', 'siteindex']].iloc[train_idx]
X_train_coord.reset_index(inplace=True,drop=True)

#test_idx

X_test.shape
#output: (95356, 25)

y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)

## 2.2 Drop coordinates
EDAsurvey.drop(columns=['x', 'y'], inplace= True, axis = 1) 

X_train.drop(columns=['x', 'y'], inplace= True, axis = 1) 
X_test.drop(columns=['x', 'y'], inplace= True, axis = 1)

"""##### 3 [ Fit: Linear Regression ] ######"""
## 3.1 Linear Regression Model | Ordinary Least Squares Method
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

### Create a linear regression object
model = linear_model.LinearRegression()

### TRAIN: Fit the model using the training set
model.fit(X_train, y_train)

## 3.2 Predict Test Results
### 3.2.1 TEST: Make prediction using test set
predictedStand = model.predict(X_test)
predictedStand

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': predictedStand})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

#summary descriptive statistics
dataTest.describe()

### 3.2.2 TRAIN: Make prediction using TRAIN set
y_train_predicted = model.predict(X_train)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

#summary descriptive statistics
dataTrain.describe()

### 3.2.3 Plot Predicted vs Observed | Test Set
import numpy as np   # To perform calculations
import matplotlib.pyplot as plt  # To visualize data and regression line
from pylab import rcParams
import seaborn as sns
sns.set(style="whitegrid")

dfTest = dataTest.head(25)
dfTest.plot(kind='bar', figsize=(12,6))

#plt.legend(title="Train set",loc='upper center', bbox_to_anchor=(1.10, 0.8), frameon=False)
plt.legend(title="Train set", frameon=  True)
plt.title('Actual vs Predicted \'siteindex\' Values in Train Set' )
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.xticks(rotation=45, horizontalalignment='right')

plt.savefig('actualvsPredictedBar_LM_testSet.jpg', bbox_inches='tight', dpi=300)

### 3.2.4 Plot Goodness of fit for siteIndex values | Test set
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

plt.savefig('actualvsPredicted_LM_testSet.jpg', bbox_inches='tight', dpi=300)

"""##### 4 [ Perfomance and Validation ] #####"""
## 4.1 ACCURACY FOR TRAINING & TEST SET:
print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))

## 4.2 Accuracy Measures
print("R2 (explained variance) Train Set: {:.3f}".format(metrics.r2_score(y_train, y_train_predicted), 2))
print("R2 (explained variance) Test set: {:.3f}".format(metrics.r2_score(y_test, predictedStand), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictedStand))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, predictedStand))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictedStand)))

### Confidence Interval for Regression Accuracy
from math import sqrt
interval = 1.96 * sqrt( (0.488 * (1 - 0.488)) / 95356)
print('%.3f' % interval)

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
plt.savefig('SquaredError_LM.png', bbox_inches='tight', dpi=300)

"""##### 5 [ Evaluation: Slope of Coefficients ] #####"""
from sklearn.metrics import mean_squared_error, r2_score
## 5.1 Model Output
# a. Intercept
print("Intercept:", model.intercept_)

# b. Coefficient - the slop of the line
print("Coefficients(slope of the line):", model.coef_)

# c. the error - the mean square error
print("Mean squared error: %.3f"% mean_squared_error(y_test,predictedStand))

# d. R-square -  how well x accout for the varaince of Y
print("R-square: %.3f'" % r2_score(y_test,predictedStand))

## 5.2 Build table to check model output
pred_model = pd.DataFrame(['aspect','planCurvature','profileCurvature','slope','TPI','TWI_SAGA','Dh_diffuse','Ih_direct','DEM','meanJanRain','meanJulRain','maxJanTemp','minJulTemp','SYMBOL','soil_order','BDw','CLY','CFG','ECD','SOC','pHw','SND','SLT'])
coeff = pd.DataFrame(model.coef_, index=['Co-efficient']).transpose()

pd.concat([pred_model,coeff], axis=1, join='inner')

## 5.3 Plot Slopes
column_names = ['aspect','planCurvature', 'profileCurvature','slope','TPI','TWI_SAGA','Dh_diffuse','Ih_direct','DEM','meanJanRain','meanJulRain','maxJanTemp','minJulTemp','SYMBOL','soil_order','BDw','CLY', 'CFG','ECD','SOC','pHw','SND','SLT']

regression_coefficient = pd.DataFrame({'Feature': column_names, 'Coefficient': model.coef_}, columns=['Feature', 'Coefficient'])

### 5.3.1 Display contribution of features towards dependent variable: 'siteindex' (y)
plt.figure(figsize=(14,8))
g = sns.barplot(x='Feature', y='Coefficient', data=regression_coefficient, capsize=0.3, palette='spring')
g.set_title("Contribution of features towards dependent variable: 'siteindex' (y)", fontsize=15)
g.set_xlabel("independent variables (x)", fontsize=13)
g.set_ylabel("slope of coefficients (m)", fontsize=13)
plt.xticks(rotation=45, horizontalalignment='right')
g.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
g.set_xticklabels(column_names)
for p in g.patches:
    g.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), 
               textcoords='offset points', fontsize=14, color='black')
    
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')

plt.savefig('FI_LM.png', bbox_inches='tight', dpi=300)

"""##### 6 [ Regression assumptions ] #####"""
## 6.1 Building Normal and Density Distribution of Errors Graph
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

plt.savefig('densityPlotHist.jpg', bbox_inches='tight', dpi=300)

## 6.2 KDE Plot of Normal Distribution of Values
plus_one_std_dev = np.mean(error_info.error) + np.std(error_info.error)
minus_one_std_dev = np.mean(error_info.error) - np.std(error_info.error)

import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize = [8, 8]) 

kde = stats.gaussian_kde(error_info.error)
pos = np.linspace(min(error_info.error), max(error_info.error), 50000)
plt.plot(pos, kde(pos), color='purple')
shade = np.linspace(minus_one_std_dev, plus_one_std_dev, 300)
plt.fill_between(shade, kde(shade), alpha=0.5, color='purple',)
plt.text(x=0.25, y=.0085, horizontalalignment='center', fontsize=10, 
         s="68% of values fall within\n this shaded area of\n plus or minus 1 standard\n deviation from the mean", 
         bbox=dict(facecolor='whitesmoke', boxstyle="round, pad=0.25"))
plt.title("KDE Plot of Normal Distribution of Values", fontsize=10, y=1.012)
plt.xlabel("values", labelpad=15)
plt.ylabel("probability", labelpad=15);

plt.savefig('kdePlot.jpg', bbox_inches='tight', dpi=300)

## 6.2.1 Display values from KDE Plot
std_dev = round(np.std(error_info.error), 1)
median = round(np.median(error_info.error), 1)

print("normal_distr_values has a median of {0} and a standard deviation of {1}".format(median, std_dev))

mean = round(np.mean(error_info.error), 1)
mean

for number_deviations in [-3, -2, -1, 1, 2, 3]:
    value = round(np.mean(error_info.error) + number_deviations * np.std(error_info.error), 1)
    print("{0} is {1} standard deviations from the mean".format(value, number_deviations))

## 6.3 Probability Plot to Compare Normal Distribution Values to Perfectly Normal Distribution
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

fig, ax = plt.subplots(1, 1)
x = stats.probplot(error_info.error, dist="norm", fit=True, rvalue=True, plot=plt)

ax.get_lines()[0].set_markerfacecolor('pink')
ax.get_lines()[0].set_markeredgecolor('blue')
ax.get_lines()[0].set_markersize(6)

plt.xlabel("Theoretical quantiles | Interpretation: standard deviations", labelpad=15)
plt.title("Probability Plot to Compare Normal Distribution Values to\n Perfectly Normal Distribution", y=1.015)

plt.savefig('probabilityPlot.jpg', bbox_inches='tight', dpi=300)

### 6.3.1 Display Shapiro-Wilk Test values from previous plot
#Shapiro-Wilk Test to test Normal Distribution (slow way)
w, pvalue = stats.shapiro(error_info.error)
print(w, pvalue)

## 6.4 Normal Q-Q plot test Normal distribution Plot
#Normal Q-Q plot test Normal distribution
#From the above figure, we see that all data points lie to close to the 45-degree line 
#and hence we can conclude that it follows Normal Distribution.
res = error_info.error
fig = sm.qqplot(res, line='s')
plt.show()

### 6.4.1 Values from previous plot
import scipy.stats as stats
stats.describe(error_info.error)

t, pvalue = stats.ttest_1samp(error_info.error, 0.010480123194236406)
print(t, pvalue)

"""##### 7 [ Fit Model: Linear model | K-fold Cross Validation ] #####"""
"""## 7.1 Model with 10-fold cross-validation with all features ##"""

"""###  Option 1 ###"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#lm = LinearRegression()

### 1. evaluate the model
scores10 = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(scores10))

### 2. report performance
print("Average cross-validation score: {:.3f}".format(scores10.mean()))
print('MAE: %.3f (%.3f)' % (mean(scores10), std(scores10)))
print("Accuracy: %0.3f (+/- %0.3f)" % (scores10.mean(), scores10.std()))

#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy for 95perc confidence interval: %0.3f (+/- %0.3f)" % (scores10.mean(), scores10.std() * 2))

#### 2.1 Measures for boxplots
import statistics
from scipy import stats
# Median for predicted value
median = statistics.median(scores10)

q1, q2, q3= np.percentile(scores10,[25,50,75])
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

### 3. plot performance model with 10-fold cross-validation
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

plt.boxplot(scores10, medianprops=medianprops, meanprops=meanprops, showmeans=True)
ax.set_xticklabels('')
plt.xlabel('Linear Regression')
plt.ylabel('Accuracy Model')

# Show the grid lines as light grey lines
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.savefig('accuracy_LM.png', bbox_inches='tight', dpi=300)

"""### Option 2 ###"""
# step-1: create a cross-validation scheme
from sklearn.model_selection import StratifiedKFold, KFold
folds = KFold(n_splits = 10, shuffle = True, random_state = 42)
#folds = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

# step-2: specify range of hyperparameters to tune, consider all available #features
params = [{'n_features_to_select': list(range(1, len(X_list)+1))}]
params

# step-3: perform grid searchf
from sklearn.feature_selection import RFE
## 3.1 specify model
#Create a linear regression object
lm = linear_model.LinearRegression()

rfe = RFE(lm)

## 3.2 building GridSearchCV
from sklearn.model_selection import GridSearchCV
model_cv = GridSearchCV(estimator = rfe, param_grid = params, scoring= 'r2', cv = folds, verbose = 1, return_train_score=True)
model_cv

# 4. fit the model KFold=10
model_cv.fit(X_train, y_train)

# 5. Accuracy model for KFold=10
## 5.1 ACCURACY FOR TRAINING SET:
print("Accuracy on training set: {:.3f}".format(model_cv.score(X_train, y_train)))
## 5.2 ACCURACY FOR TEST SET:
print("Accuracy on test set:: {:.3f}".format(model_cv.score(X_test, y_test)))

## 5.3 Predicting the Test set results
y_pred = model_cv.predict(X_test)
y_pred

# 6. EVALUATE MODEL: R2 | KFold=10
print("R2 (explained variance): {:.3f}".format(metrics.r2_score(y_test, y_pred), 2))
print('MAE=Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE=Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE=Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

##6.1 Feature Importance evaluation
from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(rf, X_train, y_train):
    return r2_score(y_train, model_cv.predict(X_train))

perm_imp_rfpimp = permutation_importances(model_cv, X_train, y_train, r2)
perm_imp_rfpimp

"""## 7.2 Evaluating Linear Regression Models ##"""
from sklearn.model_selection import cross_val_score
# function to get cross validation scores
def get_cv_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=10,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')

### 7.2.1 Linear Model | Ordinary Least Squares
lm = LinearRegression()
# get cross val scores
get_cv_scores(lm)
   
