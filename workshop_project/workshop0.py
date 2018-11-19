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

def mean(ar):
    return sum(ar) / len(ar)

class BostonModel:
    def __init__(self):
        self.run_type = "standard"
        self.run_type = "deep"
        self.run_type = "wide"

        self.prepare_data = True

        (self.X,self.Y), (self.x_test, self.y_test) = boston_housing.load_data(test_split=0.2)

    def baseline_model(self):
        # create model scaffold
        model = Sequential()
        # note: kernel_Initializer default is glorot_uniform.
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation="relu"))
        model.add(Dense(1))

        # compile model with choice off error function (here, mse)
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def mse(self, prediction, actual):
        # todo: use numpy arrays
        errors = []
        for i in range(0, len(prediction)):
            errors[i] = pow(prediction[i]-actual[i], 2)
        
        return mean(errors)

    def predict_outputs(self, estimator, x_test):
        estimator.fit(self.X,self.Y)
        prediction = estimator.predict(x_test)

        print("prediction: %.2f" % self.mse(prediction, self.y_test))

def main():
    print("Running BostonHousing model")
    m = BostonModel()

    # use fixed random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # pull model and do training (80% of dataset)
    model = m.baseline_model()
    modelHandle = m.baseline_model

    X = m.X
    Y = m.Y
    x_test = m.x_test
    y_test = m.y_test

    if m.prepare_data:
        estimators=[]
        estimators.append(('standardize', StandardScaler()))
        regressor = KerasRegressor(build_fn=m.baseline_model, epochs=50, batch_size=5, verbose=0)
        estimators.append(('mlp', regressor))
        pipeline=Pipeline(estimators)

        kfold=KFold(n_splits=10, random_state=seed)
        results = cross_val_score(pipeline, X,Y, cv=kfold)
        print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    else:
        estimator = KerasRegressor(build_fn=m.baseline_model, epochs=100, batch_size=5, verbose=0)
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(estimator, X, Y, cv=kfold)
        
        # also try a prediction:
        # m.predict_outputs(estimator, x_test)

    print("Output:: %.2f (%.2f) MSE" % (abs(results.mean()), results.std()))

    
    # estimator = model.fit(X, Y, epochs=100, batch_size=5, verbose=0)

    # evaluation:
    # loss_and_metrics=model.evaluate(x_test, y_test, batch_size=5)

    # predict
    # classes = model.predict(x_test, batch_size=5)

    print ("done")

if __name__ == "__main__":
    main()