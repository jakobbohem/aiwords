from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Activation
# create call option:
# create call option:
# model=Sequential([
# 	Dense(32, input_shape=(784,)),
# 	Activation('relu'),
# 	Dense(10),
# 	Activation('softmax'),
# ])

model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))

# multi-class classification problem
model.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])

#binary classification problem:
model.compile(optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['accuracy'])

# for a mean square error regression problem:
model.compile(optimizer='rmsprop',
	loss='mse')



# custom metrics:
import keras.backend as K

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

## create the model
model.compile(optimizer='rmsprop',
	loss='binary_crossentropy',
	metrics=['accuracy', mean_pred])

### TRAINING
model.fit(data, labels, epochs=10, batch_size=32	)


# model.compile(loss=keras.losses.categorical_crossentropy,
	# optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=5, batch_size=32)
