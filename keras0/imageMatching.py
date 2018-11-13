from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

from matplotlib import pyplot as plt

### testing simple matplot lib
import numpy as np
x = np.linspace(0,2 ,100)

# plt.show()

plt.plot(x,x,label = 'linear')
plt.imshow(X_train[0])
plt.show()