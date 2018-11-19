import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist

# globals:
debugim=False

# pre-shuffled data from the keras data loader...
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print (X_train.shape)

from matplotlib import pyplot as plt
# show a first image in dataset
if debugim:
	plt.imshow(X_train[0])
	plt.show()

a = X_test.shape[1]
X_train = X_train.reshape(X_train.shape[0], a, a, 1)
X_test = X_test.reshape(X_test.shape[0],a, a, 1)
print(X_test.shape)

# need brightness vales to be float32:s in [0,1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# print(y_train[:10])

# change number output to a bool-vector hit output (better?)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
## --> [0, 0, 1, 0, 0, 0 ..., 0] == 2

model = Sequential()
# note: step size == 1,1 can be changed using 'subsample' property
model.add(Convolution2D(32, 3, 3, 
					activation='relu', 
					input_shape=(a, a, 1)))

# DEBUG
# print(model.output_shape)

model.add(Convolution2D(32,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)



