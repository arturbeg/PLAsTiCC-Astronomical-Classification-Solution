'''
Task 3 -- Data Augmentation Approach
CS342 Assignment 2 Solutions
u1610375
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import logging
import time
from sklearn.preprocessing import minmax_scale, StandardScaler

# read csv files
train       = pd.read_csv('/modules/cs342/Assignment2/training_set.csv', header=0)
meta_train  = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv', header=0)

# Place the target in the train timeseries
## so later the newly created observations have a target associated with them
targets = meta_train[['object_id','target']]

# So, now the newly created objects will have a target
train = train.merge(
                right=targets,
                how='outer',
                on='object_id'
            )

# Window (Time) Warping algorithm
def time_warping(train,  meta_train):
    unique_objects = meta_train['object_id'].unique()
    
    train_warped = pd.DataFrame()
    # Observations stretched by a given scaling factor are associated with a specific bathc
    for batch, scaling_factor in enumerate([0.01, 0.05, 0.1, 0.2]):
        for new_local_obj_id, obj_id in enumerate(unique_objects):
            # For each object in each scaling factor batch, conduct window warping
            # Get one object
            obj = train[train['object_id']==obj_id].copy()
            
            # Get the range and count
            obj_max = obj['mjd'].max()
            obj_min = obj['mjd'].min()
            obj_range = obj_max - obj_min
            
            count = obj['mjd'].count()
            
            # Stretch the time series using the minmax_scale
            # the scaling factor determines the new feature range for the MJD variable
            # the furthre away the observations are, the larger the feature range is
            new_time = minmax_scale(
                    np.array(obj['mjd']),
                    feature_range=(obj_min - scaling_factor*obj_range, obj_max + scaling_factor*obj_range)
            )
            
            # Assign new times to object
            obj['mjd'] = new_time
            new_obj_id = train['object_id'].max() + train['object_id'].count()*batch + new_local_obj_id
            obj['object_id'] = new_obj_id

            # Add the object into the dataframe of newly created (warped) train-series
            train_warped = pd.concat([train_warped, obj])
        
    return train_warped

# Generate the time_warped dataframe
train_warped = time_warping(train, meta_train)
# Merge the warped values with the original train
final_train = pd.concat([train_warped, train])
final_train.to_csv('./modules/cs342/Assignment2/data_augmentation.csv')

# Potential method
'''
Could potentially implement data augmentation on
the simple functions of the data and not on the
raw time series itself (idea for the future)
'''