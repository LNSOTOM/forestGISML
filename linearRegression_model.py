#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: laurasotomayor

"""""""""""""""""""""""""""""""""""""""""""""""""""""


"""
# DATA PREPARATION """
""""""""""""""""""""""""""""""""""""""""""""""""""
## 1 Load Dataset

### 1.1 Define relative path to file
EDAsurvey = pd.read_csv('/content/drive/<path of my file>')
EDAsurvey.head()

### 1.2 check database dimensions
print('Examples:{}\nFeatures: {}'.format(EDAsurvey.shape[0], EDAsurvey.shape[1]))
#OUTPUT: (953556, 32)


## 2 Cleaning: drop RCODE and DESCRIPT

### 2.1 Remove column  RCODE and DESCRIPT for ML
EDAsurvey.drop(['RCODE', 'DESCRIPT'], inplace = True, axis = 1) 
EDAsurvey.head()

### 2.2 Check info of dataset
EDAsurvey.info()
"""""""""""""""""""""""""""""""""""""""""""""""""""""



# PRE-PROCESSING
"""""""""""""""""""""""""""""""""""""""""""""""""""""
# 1 Categorical Data

## 1.1 Target Encoding
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