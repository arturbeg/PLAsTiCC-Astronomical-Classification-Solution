'''
Task 4 -- Random Forest and Multi-Layer Classifiers
using raw time series values (or simple functions of them) as inputs
to classify the time series
CS342 Assignment 2 Solutions
u1610375
'''
# Dependencies
import numpy as np
import pandas as pd
import gc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# read_csv files
train_series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv',header=0)
train_metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv',header=0)

# simple functions of flux to be used by the classifiers
simple_features = train_series.groupby(
    ['object_id', 'passband'])['flux'].agg(
    ['mean', 'max', 'min', 'std']).unstack('passband')


clf = RandomForestClassifier(n_estimators=200, criterion='gini',\
                       oob_score=True, n_jobs=4, random_state=42,\
                      verbose=1, class_weight='balanced', max_features='sqrt')

# Uses a set of binary classifiers to conduct multiclass classification
clf = OneVsRestClassifier(clf)

# Makes sure the probabilities predicted by the classifier are in line with the
## distribution of the training set
calibrator = CalibratedClassifierCV(clf, cv=3)

# function to normalise the data in the simple features dataframe
def normalize(ts):
    return (ts-ts.mean()) / ts.std()

simple_features = normalize(simple_features)

# Extracting class target data
Y_train = train_metadata['target']

calibrator.fit(simple_features, Y_train)

### MLP For the Simple Features (NO FEATURE ENGINEERING) ###
# Multi Layer Perceptron Classifier
# Dependencies
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from collections import Counter 

## Inspired by: https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
# Defining simple model in Keras
K.clear_session()
def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 512
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=simple_features.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//2,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    
    model.add(Dense(14, activation='softmax'))
    return model 

unique_y = np.unique(Y_train)
class_map = dict()
for i, val in enumerate(unique_y):
    class_map[val] = i
print(class_map)

y_map = np.zeros((Y_train.shape[0],))    

y_map = np.array([class_map[val] for val in Y_train])

# converting the target vector into a one-hot encoding form
# produces a matrix of height: number of objects and width of the number of classes
y_categorical  = to_categorical(y_map)

# Calculating the class weights
y_count = Counter(y_map)
print(y_count)

wtable = np.zeros((len(unique_y),))

for i in range(len(unique_y)):
    wtable[i] = y_count[i] / y_map.shape[0]

mlp = build_model()
# Compiling the model with an Adam optimizer (more in the report)
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

mlp.fit(simple_features, y_categorical, epochs=10, batch_size=32)




