# regression predictive modelling problem using the boston housing dataset
# full walkthrough and explanation here: https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

import numpy as np
# import pandas
from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import boston_housing

# data processing + analysis
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

(X,Y), (x_test, y_test) = boston_housing.load_data(test_split=0.2)

def baseline_model():
    # create model scaffold
    model = Sequential()
    model.add(Dense(13, input_dim=13, activation="relu"))
    model.add(Dense(1))

    # compile model with choice off error function (here, mse)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# use fixed random seed for reproducibility
seed=5
np.random.seed(seed)

# pull model and do training (80% of dataset)
model = baseline_model()
estimators=[]
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline=Pipeline(estimators)

kfold=KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X,Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# estimator = model.fit(X, Y, epochs=100, batch_size=5, verbose=0)



# evaluation:
loss_and_metrics=model.evaluate(x_test, y_test, batch_size=5)

# predict
# classes = model.predict(x_test, batch_size=5)

print ("done")
