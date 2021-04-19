import numpy as np


'''
weights update algorithm using matrix operations
run the input vector through the perceptron
extract the predicted output class
if out not equal to target
    calculate the threshold of the output
    convert target to one_hot vector
    compute error vector
    wrap error in matrix
    transpose error vector
    wrap input in matrix
    matrix multiply error vector and input vector (gives a matrix)
    scale by learning rate
    add this delta matrix to the weights matrix
'''


class pcn:

    # sets perceptron parameters and initializes weights matrix
    def __init__(self, num_input_nodes, num_output_nodes):
        self.num_in = num_input_nodes + 1
        self.num_out = num_output_nodes
        self.weights = np.random.default_rng().uniform(-0.05, 0.05, (self.num_out, self.num_in))


    # data = pre_process(data)
    def pre_process(self, input_data):
        # flatten each data point to a 1d array
        dim = np.shape(input_data)
        input_data = np.reshape(input_data, (dim[0], dim[1] * dim[2]))

        # scale each data point to be a value between 0 and 1
        scale = 1.0 / 255.0
        input_data = input_data * scale

        # add bias node to beginning of each data point
        bias = np.ones((dim[0], 1))
        input_data = np.concatenate((bias, input_data), axis=1)
        return input_data


    def train(self, input_data, output_data, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            count = 1
            for data, target in zip(input_data, output_data):
                y = np.dot(self.weights, data)
                prediction = np.argmax(self.one_hot(y))
                if prediction != target:
                    y = self.threshold(y)
                    t = np.zeros(self.num_out)
                    t[target] = 1
                    error = t - y
                    error = np.array([error])
                    error = np.transpose(error)
                    x = np.array([data])
                    d_w = np.dot(error, x)
                    d_w = d_w * learning_rate
                    self.weights += d_w
                print('\r# input data: ' + str(count), end = '')
                count += 1
            # calculate accuracy???
            # do other per-epoch things
        print('')


    def confusion_matrix(self, input_data, target_labels):
        cnf = np.zeros((10, 10))
        outputs = np.dot(input_data, np.transpose(self.weights))
        for y, t in zip(outputs, target_labels):
            val = self.interpret(self.one_hot(y))
            cnf[val][t] += 1
        return cnf


    def accuracy(self, confusion_matrix):
        total = 0;
        correct = 0
        for i in range(self.num_out):
            for j in range(self.num_out):
                total += confusion_matrix[i][j]
                if i == j:
                    correct += confusion_matrix[i][j]
        return correct / total


    # returns unthresholded output vector
    def recall(self, input_vector):
        return np.dot(self.weights, input_vector)


    def threshold(self, output_vector):
        return np.where(output_vector > 0, 1, 0)


    def one_hot(self, raw_output):
        dim = np.shape(raw_output)
        out = np.zeros(dim[0])
        index = np.argmax(raw_output)
        out[index] = 1
        return out


    def interpret(self, one_hot_output):
        return np.argmax(one_hot_output)
