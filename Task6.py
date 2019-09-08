'''
Task 6 -- CNN based classification
CS342 Assignment 2 Solutions
u1610375
'''
### CNN Classification of Light Curves ###
'''
Inspired by: https://www.kaggle.com/higepon/updated-keras-cnn-use-time-series-data-as-is
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import gc
import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Lambda
from keras.layers import GRU, Dense, Activation, Dropout, concatenate, Input, BatchNormalization
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import time
from tensorflow.python.client import timeline
import re
import time

train = pd.read_csv('/modules/cs342/Assignment2/training_set.csv', header=0)

# Uncomment the lines below to use the augmented_train instead of the original train dataset in the CNN
# To generate the data_augmentation.csv, need to run the data_augmentation.py script
# augmentred_train = pd.read_csv('data_augmentation.csv', header=0)
# train = augmented_train.copy()

### Standardize the input ###
ss1 = StandardScaler()
train[['mjd', 'flux', 'flux_err']] = ss1.fit_transform(train[['mjd', 'flux', 'flux_err']])

### Sort train data before we group them ###
train = train.sort_values(['object_id', 'passband', 'mjd'])

### Time Series Transformation ###
'''
Transforming train data into 2D data 

[num_passbands, len(flux) + len(flux_err) + len(det)]
as below

So, for each object_id we have one monotone image which has

width: num_passbands
height: len(flux) + len(flux_err) + len(detected)

'''
train_timeseries = train.groupby(['object_id', 'passband'])['flux', 'flux_err', 'detected'].apply(lambda df: df.reset_index(drop=True)).unstack()
train_timeseries.fillna(0, inplace=True)

# rename column names
train_timeseries.columns = ['_'.join(map(str,tup)).rstrip('_') for tup in train_timeseries.columns.values]
print(train_timeseries.head(7))

num_columns = len(train_timeseries.columns)
print(num_columns)

# We reshape the data into [None, num_columns, num_passbands]
X_train = train_timeseries.values.reshape(-1, 6, num_columns).transpose(0, 2, 1)

# Load metadata and construct target value one-hot vector

meta_train = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
print(meta_train.head())

classes = sorted(meta_train.target.unique())
print(classes)

# generate the classmap
class_map = dict()
for i, val in enumerate(classes):
	class_map[val] = i
print(class_map)

# Example for timeseries for passband 0
train_timeseries0 = train_timeseries.reset_indedx()
train_timeseries0 = train_timeseries0[train_timeseries0.passband==0]
# print(train_timeseries0.head())

merged_meta_train = train_timeseries0.merge(meta_train, on="object_id", how="left")
merged_meta_train.fillna(0, inplace=True)

y = merged_meta_train.target
classes = sorted(y.unique())

class_weight = {
	c: 1 for c in classes
}

for c in [64, 15]:
	class_weight[c] = 2

print('Unique classes : ', classes)	

targets = merged_meta_train.target
target_map = np.zeros((targets.shape[0],))
target_map = np.array([class_map[val] for val in targets])
# convert targets into a dummy variable matrix (one-hot encoding) form
Y = to_categorical(target_map)

### The CNN design begins here ###

batch_size = 256

def weight_variable(shape, name=None):
	return np.random.normal(scale=0.01, size=shape)

def build_model():
	input = Input(shape=(X_train.shape[1], 6), dtype='float32', name='input0')
    output = Conv1D(256,
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001))(input)

	output = BatchNormalization()(output)
	output = Activation('relu')(output)                

	output = MaxPooling1D(pool_size=4, strides=None)(output)

    output = Conv1D(256,
             kernel_size=3,
             strides=1,
             padding='same',
             kernel_initializer='glorot_uniform',
             kernel_regularizer=regularizers.l2(l=0.0001))(output)

	output = BatchNormalization()(output)
	output = Activtion('relu')(output)       

	output = MaxPooling1D(pool_size=4, strides=None)(output)
	output = Lambda(lambda x: K.mean(x, axis=1))(output) # Same as GAP for 1D Conv Layer

	output = Dense(len(classes), activation="softmax")(output)
	model  = Model(inputs=input, outputs=output)
	return model


# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss

epochs = 1000 # number of times to go through each training observation when updating the weights of the model
y_count = Counter(target_map)
wtable = np.zeros((len(classes),))
for i in range(len(classes)):
    wtable[i] = y_count[i] / target_map.shape[0]

y_map = target_map
y_categorical = Y
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
start = time.time()
clfs = []
oof_preds = np.zeros((len(X_train), len(classes)))

model_file = "model.weigths"

for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    checkPoint = ModelCheckpoint(model_file, monitor='val_loss',mode = 'min', save_best_only=True, verbose=0) # saving the model

    x_train, y_train = X_train[trn_], Y[trn_]
    x_valid, y_valid = X_train[val_], Y[val_]
    
    model = build_model()  
    # Generating an Adam optimiser (extension of the original stochastic gradient descent) with a learning rate (step) of 0.001  
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto') # early stopping to stop training the network once the validation loss gets too high

    model.compile(loss=mywloss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                        batch_size=batch_size,
                    shuffle=False,verbose=1,callbacks=[checkPoint, stopping])           
    plot_loss_acc(history)
    
    print('Loading Best Model')
    model.load_weights(model_file)
    # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict(x_valid,batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict(x_valid,batch_size=batch_size)))
    clfs.append(model)
    
print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(Y,oof_preds))

elapsed_time = time.time() - start
print("elapsed_time:", elapsed_time)


### Train on 6274 samples, validate on 1574 samples

### Test set predictions using the model were made using a Kaggle Kernel ###









