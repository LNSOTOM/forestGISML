############################
# NON LINEAR DATA MODEL: POLYNOMIAL REGRESSION
############################
# Reproduce the same scripts than Linear Regression (linear_regression.py)
"""##### 1 [ Split into training ] #####"""
"""##### 2 [ Extract train and test idx for later merge with geography coord ] #####"""

"""##### 3 [ Fit: Polynomial Regression ] ######"""
## 3. 1 Fit Model: Polynomial Regression
### Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()

### Train model
pol_reg.fit(X_poly, y_train)

## 3.2 Predict Test Results
### 3.2.1 TEST: Make prediction using test set
y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))
y_pred

dataTest = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataTest['residuals']=dataTest['Actual'] - dataTest['Predicted']
dataTest

#summary descriptive statistics
dataTest.describe()

### 3.2.2 TRAIN: Make prediction using TRAIN set
y_train_predicted = pol_reg.predict(X_poly)
y_train_predicted

dataTrain = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_predicted})
dataTrain['residuals']=dataTrain['Actual'] - dataTrain['Predicted']
dataTrain

#summary descriptive statistics
dataTrain.describe()

### 3.2.3 Plot Predicted vs Observed | Test Set
### Plot A
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

plt.savefig('actualvsPredicted_PolyReg_testSet.jpg', bbox_inches='tight', dpi=300)

### Plot B
ax = sns.regplot(x="Actual", y="Predicted", data=dataTest,
                 scatter_kws = {'color': 'orange', 'alpha': 0.3}, line_kws = {'color': '#f54a19'},
                 x_estimator=np.mean, logx=True)

ax.set_ylim(0,55)
ax.set_xlim(0,55)
ax.plot([0, 55], [0, 55], 'k--', lw=2)

### Plot C
ax = sns.regplot(x="Actual", y=y_pred, data=dataTest,
                 scatter_kws={"s": 80},
                 order=2, ci=None)

"""##### 4 [ Perfomance and Validation #####"""
## 4.1 ACCURACY FOR TRAINING & TEST SET:
print("Accuracy on train set:: {:.3f}".format(pol_reg.score(X_poly, y_train)))
print("Accuracy on test set:: {:.3f}".format(pol_reg.score(poly_reg.fit_transform(X_test), y_test)))

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
plt.style.use('seaborn-whitegrid')
fig=plt.figure(figsize = [8, 6])

ax = fig.add_subplot(111)

ax.scatter(x=dataTest['Actual'], y=residSquare, label='Squared Error', c='white', alpha=0.8, edgecolors='#1b346c', s=10)
ax.set_xlabel("Observed 'site index' values") #it's a good idea to label your axes
ax.set_ylabel('Squared Error')

plt.title("Squared Error vs Observed 'site index' values")
plt.legend(title="",loc='upper right', frameon=True)
plt.savefig('SquaredError_PolyReg.png', bbox_inches='tight', dpi=300)

"""##### 5 [ Evaluation: Slope of Coefficients ] #####"""
#pol_reg.coef_
from sklearn.metrics import mean_squared_error, r2_score
## 5.1 Model Output
# a. Intercept
print("Intercept:", pol_reg.intercept_)

for coef, col in enumerate(X_train.columns):
    print(f'{col}:  {pol_reg.coef_[coef]}')

## 5.2 Build table to check model output
pred_model = pd.DataFrame(['aspect','planCurvature','profileCurvature','slope','TPI','TWI_SAGA','Dh_diffuse','Ih_direct','DEM','meanJanRain','meanJulRain','maxJanTemp','minJulTemp','SYMBOL','soil_order','BDw','CLY','CFG','ECD','SOC','pHw','SND','SLT'])
coeff = pd.DataFrame(pol_reg.coef_)

df = pd.concat([pred_model,coeff], axis=1, join='inner')
df

## 5.3 Plot Slopes
# adding column name to the respective columns 
df.columns =['Features', 'Coefficients'] 
  
# displaying the DataFrame 
print(df)

df = df.sort_values(by='Coefficients', ascending=0)
df

### 5.3.1 Display contribution of features towards dependent variable: 'siteindex' (y)
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
sns.set(style="whitegrid")
plt.subplot(1, 1, 1) # 1 row, 2 cols, subplot 1

ax = sns.barplot(df.Features, df.Coefficients)

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()),
                ha='left',
                va='baseline', 
                #textcoords='offset points',
                rotation='30')
                
#Rotate labels x-axis
plt.xticks(rotation=45, horizontalalignment='right')
plt.ylabel('independent variables (x)')
plt.xlabel('Coefficents')
plt.title("Contribution of features towards dependent variable: 'siteindex' (y)")

plt.savefig('polyreg_FI.png', bbox_inches='tight', dpi=300)

## 5.4 Feature Importance
from sklearn.inspection import permutation_importance
r = permutation_importance(polyreg, X_test, y_test,
                          n_repeats=30,
                           random_state=0)

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
       print(f"{EDAsurvey.columns[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
             f" +/- {r.importances_std[i]:.3f}")

"""##### 6 [ Fit Model: Polynomial regression | K-fold Cross Validation ] #####"""
"""## Model with 10-fold cross-validation with all features ##"""
"""### Option 1 | Assessing Quality of Regression Model ###"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()

# Train model
pol_reg.fit(X_poly, y_train)

cv_scores_Kfold10 = cross_val_score(pol_reg,X_poly, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(cv_scores_Kfold10))
print("Average cross-validation score: {:.3f}".format(cv_scores_Kfold10.mean()))

"""### Option 2 | Polynomial Regression | cv=10 ###"""
from sklearn.model_selection import cross_val_score
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

###1. evaluate the model
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()

poly10 = cross_val_score(pol_reg , X_poly, y_train, cv=10, scoring='r2')
print("Cross-validation scores: {}".format(poly10))

### 2. report performance
from numpy import mean
from numpy import std

print("Average cross-validation score: {:.3f}".format(poly10.mean()))
print('MAE: %.3f (%.3f)' % (mean(poly10), std(poly10)))
print("Accuracy: %0.3f (+/- %0.3f)" % (poly10.mean(), poly10.std()))

#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy for 95perc confidence interval: %0.3f (+/- %0.3f)" % (poly10.mean(), poly10.std() * 2))

# 2.1 Measure for boxplots
import statistics
from scipy import stats
# Median for predicted value
median = statistics.median(poly10)

q1, q2, q3= np.percentile(poly10,[25,50,75])
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
# Cool Boxplot
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

plt.boxplot(poly10, medianprops=medianprops, meanprops=meanprops, showmeans=True )
ax.set_xticklabels('')
plt.xlabel('Polynomial Regression')
plt.ylabel('Accuracy Model')
plt.savefig('accuracy_polyReg.png', bbox_inches='tight', dpi=300)

# Boring boxplot
fig = plt.figure()
fig.suptitle('Model with 10-fold cross-validation')
ax = fig.add_subplot(111)
import seaborn as sns
sns.set(style="whitegrid")


plt.boxplot(poly10)
ax.set_xticklabels('')
plt.xlabel('Polynomial Regression')
plt.ylabel('Accuracy Model')
plt.savefig('accuracy_polyReg.png', bbox_inches='tight', dpi=300)