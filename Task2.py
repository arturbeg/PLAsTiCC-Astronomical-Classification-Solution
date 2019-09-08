'''
Task 2 -- Feature Engineering Approaches
CS342 Assignment 2 Solutions
u1610375
'''
### Dependencies ###
import numpy as np
import pandas as pd
import gc
from tsfresh.feature_extraction import extract_features

# read csv files
train_series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv',header=0)
train_metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv',header=0)

# Uncomment the lines below to use the augmented_train instead of the original train_series dataset
# To generate the data_augmentation.csv, need to run the data_augmentation.py script
# augmentred_train = pd.read_csv('data_augmentation.csv', header=0)
# train_series = augmented_train.copy()
# simple features for baseline models in Task 4
# simple_features = train_series.groupby(
#     ['object_id', 'passband'])['flux'].agg(
#     ['mean', 'max', 'min', 'std']).unstack('passband')
# simple_features.to_csv('./simple_features.csv')


## Mjd_diff for detected==1 -- Feature Engineering 1 ##
train = train_series.copy()
# Inspired by: https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135/notebook
train_detected = train[train['detected']==1]
train_detected['mjd_min']  = train_detected.groupby('object_id').mjd.transform('min')
train_detected['mjd_max']  = train_detected.groupby('object_id').mjd.transform('max')
train_detected['mjd_diff'] = train_detected['mjd_max'] - train_detected['mjd_min']
mjd_diff_df = train_detected[['object_id', 'mjd_diff']]
mjd_diff_df = mjd_diff_df.drop_duplicates()
mjd_diff_df = mjd_diff_df.reset_index()
del mjd_diff_df['index']


### Generating Absolute Magnitude Feature - Feature Engineering 2 ###
train_flux = train.copy()
train_flux['flux_min']  =  train_flux.groupby('object_id').flux.transform('min')
train_flux['flux_max']  =  train_flux.groupby('object_id').flux.transform('max')
train_flux['flux_diff'] =  train_flux['flux_max'] - train_flux['flux_min'] + 0.001 # specify why add the 0.001
flux_diff_df = train_flux[['object_id', 'flux_diff']]
flux_diff_df = flux_diff_df.drop_duplicates()
flux_diff_df = flux_diff_df.reset_index()
del flux_diff_df['index']
## Log transform the flux_diff and multiply by -2.5 according to the formula of magnitude##
flux_diff_df['flux_diff'] = np.log10(flux_diff_df['flux_diff'])
flux_diff_df['flux_diff'] = flux_diff_df['flux_diff'] * -2.5

## Collect the distmod data for each object_id ##
distmod_df = train_metadata.copy()
distmod_df = distmod_df[['object_id', 'distmod']]
distmod_df = distmod_df.fillna(0.0)

## Calculate the absolute magnitude ##
absolute_mag = flux_diff_df['flux_diff'] - distmod_df['distmod']

### Combine the 2 features generated above into one pandas dataframe ###
train_fe = flux_diff_df.copy()
train_fe['absolute_mag'] = absolute_mag
train_fe['mjd_diff'] = mjd_diff_df['mjd_diff']
del train_fe['flux_diff']

## Generate the Phase Curve Data using LombScargle algorithm - Feature Engineering 3 ##
## Inspired by: https://www.kaggle.com/rejpalcz/feature-extraction-using-period-analysis
from scipy.signal import lombscargle
import math
from tqdm import tqdm

## Helper functions, angular frequency to period
def freq2Period(w):
    return 2 * math.pi / w
def period2Freq(T):
    return 2 * math.pi / T

gc.enable()
gc.collect()

# get data and normalize bands
def processData(train, object_id):
    
    #load data for given object
    X = train.loc[train['object_id'] == object_id]
    x = np.array(X['mjd'].values)
    y = np.array(X['flux'].values)
    passband = np.array(X['passband'].values)
    
    # normalize bands
    for i in np.unique(passband):
        yy = y[np.where(passband==i)]
        mean = np.mean(yy)
        std = np.std(yy)
        y[np.where(passband==i)] = (yy - mean)/std
    
    return x, y, passband

# Calculate periodogram
def getPeriodogram(x, y, steps=10000, minPeriod=None, maxPeriod=None):
    if not minPeriod:
        minPeriod = 0.1 # for now, let's ignore very short periodic objects
        
    if not maxPeriod:
        maxPeriod = (np.max(x) - np.min(x))/2 # you cannot detect P > half of your observation period
        
    maxFreq = np.log2(period2Freq(minPeriod))
    minFreq = np.log2(period2Freq(maxPeriod))
    
    f = np.power(2, np.linspace(minFreq, maxFreq, steps))
    p = lombscargle(x, y, f, normalize=True)
    
    return f, p

def findBestPeaks(x, y, F, P, threshold=0.3, n=5):
    
    # find peaks above threshold
    indexes = np.where(P>threshold)[0]
    # if nothing found, look at the highest peaks anyway
    if len(indexes) == 0:
        q = np.quantile(P, 0.9995)
        indexes = np.where(P>q)[0]
    
    peaks = []
    start = 0
    end = 0
    for i in indexes:
        if i - end > 10:
            peaks.append((start, end))
            start = i
            end = i
        else:
            end = i
    
    peaks.append((start, end))
        
    
    # increase accuracy on the found peaks
    results = []
    for start, end in peaks:
        if end > 0:
            minPeriod = freq2Period(F[min(F.shape[0]-1, end+1)])
            maxPeriod = freq2Period(F[max(start-1, 0)])
            steps = int(100 * np.sqrt(end-start+1)) # the bigger the peak width, the more steps we want - but sensible (linear increase leads to long computation)
            f, p = getPeriodogram(x, y, steps = steps, minPeriod=minPeriod, maxPeriod=maxPeriod)
            results.append(np.array([freq2Period(f[np.argmax(p)]), np.max(p)]))

    # sort by normalized periodogram score and return first n results
    if results:
        results = np.array(results)
        results = results[np.flip(results[:,1].argsort())]
    else:
        results = np.array([freq2Period(F[np.argmax(P)]), np.max(P)]).reshape(1,2)
    return results[0:n]

# Multiprocessing
from multiprocessing import Pool
import multiprocessing as mp

def getFeatures(object_id):
    x, y, passband = processData(train_series, object_id)
    f, p = getPeriodogram(x, y)
    peaks = findBestPeaks(x, y, f, p)
    features = np.zeros((5,2))
    features[:peaks.shape[0], :peaks.shape[1]] = peaks
    
    return np.append(np.array([object_id]), features.reshape(5*2))    

object_ids  = train_series['object_id'].unique()

p = Pool(CORES)
results = p.map(getFeatures, object_ids)

# Calculating training set
object_ids = train_series['object_id'].unique()
columns = np.array(['id'])
for i in range(5):
    period_str = 'period_'+str(i+1)
    power_str = 'power_'+str(i+1)
    columns = np.append(columns, np.array([period_str, power_str]))

results = p.map(getFeatures, object_ids)    
output = pd.DataFrame(results, columns=columns)
output['id'] = output['id'].astype(np.int32)
output.to_csv('./modules/cs342/Assignment2/train-periods.csv')


train_periods = output.copy()
train_periods = train_periods.rename(index=str, columns={"id": "object_id"})
final_fe = pd.merge(train_fe, train_periods, on="object_id")
final_fe.to_csv('./modules/cs342/Assignment2/final_fe.csv')

#### BONUS SIMPLE FEATURES (Not Featrue Engineering)####
# Inspired by: https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
### Aggregation Feature Engineering from Simple NN kernel ###

train = train_series.copy()
train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

aggs = {
    'mjd': ['min', 'max', 'size'],
    'passband': ['min', 'max', 'mean', 'median', 'std'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

agg_train = train.groupby('object_id').agg(aggs)
new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]
agg_train.columns = new_columns
agg_train['mjd_diff'] = agg_train['mjd_max'] - agg_train['mjd_min']
agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

del agg_train['mjd_max'], agg_train['mjd_min']
agg_train.head()

del train
gc.collect()

agg_train.to_csv('/modules/cs342/Assignment2/agg_train.csv')


model_features = agg_train.reset_index().merge(
    right=final_fe,
    how='outer',
    on='object_id'
)



#### Another set of bonus features from ideas from kernels kernel####
# Inspired by: https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss

'''
Per passband features with tsfresh library
fft features capture periodicity
'''
from tsfresh.feature_extraction import extract_features

gc.enable()
train = train_series.copy()

# fft stands for fast fourier transform, the coefficients of the sine and cosine terms
## are extracted by the featureize function using the extract_features method of tsfresh
## more info can be found in the report
fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None} 

def featureize(df):
    agg_df_ts = extract_features(df, column_id="object_id", column_sort="mjd", column_kind='passband',
                                 column_value="flux", default_fc_parameters=fcp, n_jobs=4)
    # tsfresh returns a dataframe with an index name="id"
    agg_df_ts.index.rename('object_id', inplace=True)
    return agg_df_ts

tsfresh_train = featureize(train)

tsfresh_train.to_csv('./modules/cs342/Assignment2/tsfresh_train.csv')
print(tsfresh_train.head())
















