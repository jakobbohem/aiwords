from keras.models import Sequential
from keras.layers import Dense, Activation

# test data includes...
from keras.datasets import boston_housing

# create call option:
# model=Sequential([
def main():
	model = Sequential()
	relu = "relu"
	softmax ="softmax"
	debug_output = False

	model.add(Dense(units=1, activation=relu, input_dim=13))
	# model.add(Dense(units=10, activation=softmax))

	# multi-class classification problem
	model.compile(optimizer='rmsprop',
		loss='mean_squared_error',
		metrics=['accuracy'])

	(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
	data = x_train
	labels = y_train

	if debug_output:
		print(data)
		print(labels)

	model.fit(data, labels, epochs=10, batch_size=32)

	# optionally, train on batch:
	# model.train_on_batch(x_batch, y_batch)
	path = "json_model.json"
	with open(path, 'w'):
		print(model.to_json())

	loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

	classes = model.predict(x_test, batch_size=128)

if __name__ == "__main__":
	main()