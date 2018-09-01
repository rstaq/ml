import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def plot_rand_digits(X, y):
    for i in range(10):
        index = np.random.randint(0, len(X))
        image = np.reshape(X[index, :], (20, 20))
        plt.imshow(image, cmap='gray')
        plt.title(y[index])
        plt.show()


def rand_init_theta(rows, cols, epsilon=0.12):
    return np.random.rand(rows, cols) * (2 * epsilon) - epsilon


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def convert_y(y):
    """
    Converts the predicted class in an np.array of shape 10x1
    where the predicted output is 1 and the other values are 0
    :param y: predicted output
    :return: np.array of shape 10x1
    """
    y_ = np.zeros(10)
    if y == 10:
        y_[0] = 1
    else:
        y_[y] = 1
    return y_


def cost_function(X, y, theta_1, theta_2, lambda_=0):
    m = len(X)
    J = 0
    theta_1_grad = np.zeros((len(theta_1), len(theta_1.T)))
    theta_2_grad = np.zeros((len(theta_2), len(theta_2.T)))
    X = np.c_[np.ones(len(X)), X]
    for i in range(m):
        y = convert_y(y[i])

        a1 = X[i, :]
        z2 = np.dot(a1, theta_1.T)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis=0)
        z3 = np.dot(a2, theta_2.T)
        a3 = sigmoid(z3)
        J += (1 / m) * (np.dot(-y, np.log(a3).T) - np.dot(1 - y, np.log(1 - a3).T))

        delta3 = a3 - y
        delta2 = np.dot(delta3, theta_2)
        delta2 = np.delete(delta2, 0)
        delta2 = delta2 * sigmoid_gradient(z2)

        theta_1_grad += np.outer(delta2, a1)
        theta_2_grad += np.outer(delta3, a2)


def main():
    input_layer_size = 400
    hidden_layer_size = 25
    output_layer_size = 10

    data = sio.loadmat("data.mat")
    X = data.get('X')
    y = data.get('y')
    # plot_rand_digits(X, y)

    theta_1 = rand_init_theta(hidden_layer_size, input_layer_size + 1)
    theta_2 = rand_init_theta(output_layer_size, hidden_layer_size + 1)

    J, grad = cost_function(X, y, theta_1, theta_2)


if __name__ == '__main__':
    main()
