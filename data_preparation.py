
############################
#  DATA PREPARATION 
############################
## 1 [ Load Dataset ]
### 1.1 Define relative path to file
EDAsurvey = pd.read_csv('/content/drive/<path of my file>')
EDAsurvey.head()

### 1.2 check database dimensions
print('Examples:{}\nFeatures: {}'.format(EDAsurvey.shape[0], EDAsurvey.shape[1]))
#OUTPUT: (953556, 32)

## 2 [ Cleaning: drop RCODE and DESCRIPT ]
### 2.1 Remove column  RCODE and DESCRIPT for ML
EDAsurvey.drop(['RCODE', 'DESCRIPT'], inplace = True, axis = 1) 
EDAsurvey.head()

### 2.2 Check info of dataset
EDAsurvey.info()

############################
#  PRE-PROCESSING 
############################
##### 1 CATEGORICAL DATA #####
# 1.1 [ Target Encoding ]
def calc_mean(EDAsurvey, by, on, m):
    # Compute the global mean
    mean = EDAsurvey[on].mean()

    # Compute the number of values and the mean of each group
    agg = EDAsurvey.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    TargEnc = (counts * means) / (counts)

    # Replace each value by the according smoothed mean
    return EDAsurvey[by].map(TargEnc)

# 1.2 Add new values to dataframe
EDAsurvey.loc[:, 'SYMBOL'] = calc_mean(EDAsurvey, by='SYMBOL', on='siteindex', m=0)
EDAsurvey.loc[:, 'soil_order'] = calc_mean(EDAsurvey, by='soil_order', on='siteindex', m=0)
EDAsurvey.head()

##### 2 FEATURES SELECTION #####
# 2.1 [ Correlation Analysis: Spearman ]
## 2.1.1 Create a copy of dataframe
EDAsurveyCORR = EDAsurvey.copy()
EDAsurveyCORR.head()

## 2.1.2 Correlation with Spearman
spearmancorr = EDAsurveyCORR.corr(method='spearman')
spearmancorr

# 2.1.3 Correlation with Spearman Improved
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(16,8))

corr_matrix = EDAsurveyCORR.corr(method="spearman")

sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Spearman correlation")
#Rotate labels x-axis
plt.xticks(rotation=45, horizontalalignment='right')

plt.savefig('Spearmancorr_EDAsurvey.png', bbox_inches='tight', dpi=300)

# 2.2 [ Drop features high correlated ]
# 2.2.1 Remove 4 columns: less feature importance
EDAsurvey.drop(columns=['catchmentArea_SAGA', 'SVF_simplified', 'Gh_total', 'distFromCoast'], inplace= True, axis = 1) 
EDAsurvey.head()

## 2.2.2 Check dimension
EDAsurvey.shape
#OUTPUT: (953556, 24)

##### 3 SCALE DATA #####
# 3.1 [ Normalize ]
from sklearn.preprocessing import MinMaxScaler

y = EDAsurveyCORR['siteindex']
EDAsurveyCORR = EDAsurveyCORR.loc[:, ~EDAsurveyCORR.columns.isin(['siteindex'])]

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(EDAsurveyCORR)
EDAsurveyCORR.loc[:,:] = scaled_values

EDAsurveyCORR['siteindex'] = y

# 3.2 [ Plot Boxplots ]
from sklearn import preprocessing
import numpy as np
# Get column names first
names = EDAsurveyCORR.columns

# Create the Scaler object
scaler2 = preprocessing.MinMaxScaler()

# Fit your data on the scaler object
EDAsurvey_normalize = scaler2.fit_transform(EDAsurveyCORR)
EDAsurvey_normalize = pd.DataFrame(EDAsurvey_normalize, columns=names)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style("white")

plt.style.use('classic')

# Seaborn visualization library
medianprops = dict(linewidth=1.5, linestyle='-', color='#fc3468')
meanprops =  dict(marker='D', markerfacecolor='indianred', markersize=4.5)
flierprops = dict(marker='o', markerfacecolor='#7fcdbb', markersize=3, alpha=0.5)

box = EDAsurvey_normalize.boxplot(return_type='axes', figsize=(10,12), vert=False, medianprops=medianprops, 
                                  meanprops=meanprops, flierprops = flierprops,
                                  notch=False,  showmeans=True)
                           
# Show the grid lines as light grey lines
plt.grid(color='white', linestyle='-', linewidth=0.25, alpha=0.5)
plt.gca().spines['left'].set_color('none')
plt.gca().spines['right'].set_color('none')

plt.savefig('boxplot_EDAsurvey_norm.png', bbox_inches='tight', dpi=300)
