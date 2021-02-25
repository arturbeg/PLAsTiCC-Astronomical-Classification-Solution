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

The lowest score on the progression graph is based on a naive approach of classifying objects based on a single variable: the boolean for being a galactic object (redshift == 0), this approach yielded a multiclass log-loss of 2.158. CNN approach was an improvement, however the log-loss value was still relatively high, this could be because the dataset is too small for a CNN to be able to extract robust features, moreover, given the sparse nature of time series data and its
differences in length, data augmentation did not fix the issue. A simple neural network that used simple functions of the light curve data alongside the metadata resulted in a noticeable improvement of the log-loss score (1.368). However, after conducting the necessary feature engineering outlined above, there has been further progression in the accuracy of results (log-loss score of 1.135). Finally, the best score was achieved using an LGBM classifier that 21 used previously engineered features (hyperparameters were truncated from the best-performer, calculated and selected by Bayesian optimisation).
