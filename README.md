# PLAsTiCC Astronomical Classification Challenge

PLAsTiCC is a large data challenge that attempts to classify astronomical objects by analysing the time series measurements of the ‘light curves’ data (intensity of photon flux) emitted by
cosmological objects using six different astronomical passbands . The flux may be decreasing or 1 increasing over time, the pattern of the changes in its brightness acts as a good indicator of the
underlying object. Each object in the set of training data belongs to one of the 14 classes, in the test data there is one additional 15th class that is meant to capture “novelties” (objects that are
hypothesised to exist).

Throughout the analysis it was identified that domain specific feature engineering is of high relevance to identifying cosmological objects. The absolute magnitude of a flux has proved to
be a very significant feature that lets us compare cosmological events that might be at different distances from the telescope. The difference between the maximum and minimum MJD values
for detected observations of flux is an extremely useful domain specific feature that can separate a “one event” from a “cyclic event”. Not to mention, a fourier transform of the time
series data is yet another feature engineering approach that proved to be very useful in the analysis. Light Gradient Boosting framework that incorporates the feature engineering has
proved to be the best approach to tackle the challenge.

## Model Performance Progression

![shot 1](progression_graph.png?raw=true)
