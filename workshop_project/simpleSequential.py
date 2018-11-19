from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K

# test data includes...
from keras.datasets import boston_housing

# create call option:
# model=Sequential([
def main():
	model = Sequential()
	relu = "relu"
	softmax ="softmax"
	debug_output = False

	model.add(Dense(units=32, activation=relu, input_dim=13))
	model.add(Dense(units=1, activation=relu))
	# model.add(Dense(units=10, activation=softmax))

	# check the layers..
	l1 = model.layers[0]
	l2 = model.layers[1]

	# 'configure learning process':
	# multi-class classification problem (still true for rmsprop/mse)??
	model.compile(optimizer='rmsprop',
		loss='mean_squared_error',
		metrics=['accuracy'])	
	
	# model.compile(optimizer='categorical_crossentropy',
	# 	loss='sgd',
	# 	metrics=['accuracy'])

	#load example data
	(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=113)
	data = x_train
	labels = y_train

	if debug_output:
		print(data)
		print(labels)

	#train network
	model.fit(x_train, y_train, epochs=20, batch_size=5, verbose=0)

	# optionally, train on batch:
	# model.train_on_batch(x_batch, y_batch)

	#convenience method for saving model structure/data:
	path = "json_model.json"
	with open(path, 'w') as outfile:
		print(model.to_json(), file=outfile)

	loss_and_metrics = model.evaluate(x_test, y_test, batch_size=5)

	# classes = model.predict(x_test, batch_size=128)

if __name__ == "__main__":
	main()