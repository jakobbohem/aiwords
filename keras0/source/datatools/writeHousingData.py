from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

def tostring(l):
	for i in range(len(l)):
		l[i] = str(l[i])
	return l

with open("housingData.csv", 'w') as f:
	i = 0
	for row in x_train:
		l = row.tolist()
		l.append(y_train[i])
		# should be lambda-able!
		print(";".join(tostring(l)), file=f)

		i += 1 # for with index?