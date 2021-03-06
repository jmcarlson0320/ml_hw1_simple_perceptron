import pcn
import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# init the perceptron and input data
p = pcn.pcn(784, 10)
x_train = p.pre_process(x_train)
x_test = p.pre_process(x_test)
print(np.shape(x_train))

cnf = p.confusion_matrix(x_train, y_train)
initial_accuracy_on_test = p.accuracy(cnf)
print(f'initial accuracy on test data: {initial_accuracy_on_test}')

learning_rate = 0.001
num_epochs = 20
results = []
print('training perceptron...')
for i in range(num_epochs):
    print('epoch ' + str(i + 1) + ' of ' + str(num_epochs))
    p.train(x_train, y_train, learning_rate, 1)

    cnf = p.confusion_matrix(x_train, y_train)
    training_results = p.accuracy(cnf)

    cnf = p.confusion_matrix(x_test, y_test)
    test_results = p.accuracy(cnf)
    results.append((training_results, test_results))
    print('accuracy on training set: ' + str(training_results))
    print('accuracy on test set: ' + str(test_results))
    print('')
    
    '''
    if i > 0:
        diff = training_results - results[i - 1][0]
        if abs(diff) < 0.001:
            break
    '''

print('results:')
print(results)
np.set_printoptions(suppress='True')
print(cnf)
plt.plot(results)
plt.show()
