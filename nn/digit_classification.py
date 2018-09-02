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


def plot_cost_function(J_costs):
    plt.plot(J_costs, 'b-')
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
    y_[y - 1] = 1
    return y_


def cost_function(X, y, theta_1, theta_2, lambda_=0):
    m = len(X)
    J = 0
    theta_1_new = np.zeros((len(theta_1), len(theta_1.T)))
    theta_2_new = np.zeros((len(theta_2), len(theta_2.T)))

    X = np.c_[np.ones(len(X)), X]
    for i in range(m):
        y_ = convert_y(y[i])

        a1 = X[i, :]
        z2 = np.dot(a1, theta_1.T)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis=0)
        z3 = np.dot(a2, theta_2.T)
        a3 = sigmoid(z3)
        J += (1 / m) * (np.dot(-y_, np.log(a3).T) - np.dot(1 - y_, np.log(1 - a3).T))

        delta3 = a3 - y_
        delta2 = np.dot(delta3, theta_2)
        delta2 = np.delete(delta2, 0)
        delta2 = delta2 * sigmoid_gradient(z2)

        theta_1_new += np.outer(delta2, a1)
        theta_2_new += np.outer(delta3, a2)

    theta_1_new /= m
    theta_2_new /= m

    # regularization
    J += (lambda_ / (2 * m)) * ( np.sum(np.power(theta_2[:,1:], 2)) + np.sum(np.power(theta_1[:,1:], 2)))
    theta_1_reg = np.array(theta_1, copy=True)
    theta_2_reg = np.array(theta_2, copy=True)
    theta_1_reg[:,0] = 0
    theta_2_reg[:,0] = 0
    theta_1_new = np.add(theta_1_new, lambda_ / m * theta_1_reg)
    theta_2_new = np.add(theta_2_new, lambda_ / m * theta_2_reg)

    return J, theta_1_new, theta_2_new


def main():
    input_layer_size = 400
    hidden_layer_size = 25
    output_layer_size = 10
    max_iter = 50

    data = sio.loadmat("data.mat")
    X = data.get('X')
    y = data.get('y')
    # plot_rand_digits(X, y)

    theta_1 = rand_init_theta(hidden_layer_size, input_layer_size + 1)
    theta_2 = rand_init_theta(output_layer_size, hidden_layer_size + 1)

    for i in range(max_iter):
        J, theta_1, theta_2 = cost_function(X, y, theta_1, theta_2, 1)
        print("Iteration {} cost {}".format(i, J))


if __name__ == '__main__':
    main()
