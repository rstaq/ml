import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import time


def plot_rand_digits(X, y, number_pictures=3):
    for i in range(number_pictures):
        index = np.random.randint(0, len(X))
        image = np.reshape(X[index, :], (20, 20), order='F')
        plt.imshow(image, cmap='gray')
        plt.title(y[index])
        plt.gray()
        plt.show()


def plot_cost_function(J_costs):
    plt.plot(J_costs, 'k-', linewidth=1)
    plt.title("J(θ) after every iteration")
    plt.xlabel("number iteration")
    plt.ylabel("J(θ)")
    plt.show()


def rand_init_theta(rows, cols, epsilon=0.12):
    """
    randomly initialize theta in the range of [-epsilon, +epsilon]
    :param rows: number of rows of the output matrix
    :param cols: number of columns of the output matrix
    :param epsilon: range of each value
    :return: rows x cols matrix of random initialized values
    """
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
    theta_1_gradient = np.zeros((len(theta_1), len(theta_1.T)))
    theta_2_gradient = np.zeros((len(theta_2), len(theta_2.T)))

    X = np.c_[np.ones(len(X)), X]
    for i in range(m):
        y_ = convert_y(y[i])

        # forward-propagation
        a1 = X[i, :]
        z2 = np.dot(a1, theta_1.T)
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis=0)
        z3 = np.dot(a2, theta_2.T)
        a3 = sigmoid(z3)
        J += (1 / m) * (np.dot(-y_, np.log(a3).T) - np.dot(1 - y_, np.log(1 - a3).T))

        # back-propagation
        delta3 = a3 - y_
        delta2 = np.dot(delta3, theta_2)
        delta2 = np.delete(delta2, 0)
        delta2 = delta2 * sigmoid_gradient(z2)

        theta_1_gradient += np.outer(delta2, a1)
        theta_2_gradient += np.outer(delta3, a2)

    theta_1_gradient /= m
    theta_2_gradient /= m

    # regularization
    J += (lambda_ / (2 * m)) * ( np.sum(np.power(theta_2[:,1:], 2)) + np.sum(np.power(theta_1[:,1:], 2)))
    theta_1_reg = np.array(theta_1, copy=True)
    theta_2_reg = np.array(theta_2, copy=True)
    theta_1_reg[:,0] = 0
    theta_2_reg[:,0] = 0
    theta_1_gradient = np.add(theta_1_gradient, lambda_ / m * theta_1_reg)
    theta_2_gradient = np.add(theta_2_gradient, lambda_ / m * theta_2_reg)

    return J, theta_1_gradient, theta_2_gradient


def predict(X, theta_1, theta_2):
    p = np.zeros((len(X), 1))
    X = np.c_[np.ones(len(X)), X]
    h1 = sigmoid(np.dot(X, theta_1.T))
    h1 = np.c_[np.ones(len(h1)), h1]
    h2 = sigmoid(np.dot(h1, theta_2.T))
    p = np.argmax(h2, axis=1)
    return p


def check_predictions(y, p):
    wrong_pred_index = []
    for i in range(len(y)):
        if p[i] + 1 != y[i]:
            wrong_pred_index.append(i)

    print("Number of misclassified examples: {} -> {}%".format(len(wrong_pred_index), 100 / len(y) * len(wrong_pred_index)))
    return wrong_pred_index


def main(max_iter=400, lambda_=1.0):
    input_layer_size = 400
    hidden_layer_size = 25
    output_layer_size = 10

    data = sio.loadmat("data.mat")
    X = data.get('X')
    y = data.get('y')
    # plot_rand_digits(X, y)

    print("Randomly initializing theta")
    theta_1 = rand_init_theta(hidden_layer_size, input_layer_size + 1)
    theta_2 = rand_init_theta(output_layer_size, hidden_layer_size + 1)
    J_costs = []

    # train theta
    for i in range(max_iter):
        J, theta_1_gradient, theta_2_gradient = cost_function(X, y, theta_1, theta_2, lambda_)
        theta_1 -= theta_1_gradient
        theta_2 -= theta_2_gradient
        J_costs.append(J)
        print("Iteration {} cost {}".format(i, J))
    # plot_cost_function(J_costs)

    p = predict(X, theta_1, theta_2)
    p_false = check_predictions(y, p)
    return len(p_false)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--search', help="searches for the best values of max_iter and lambda",
                        action='store_true', dest='search')
    args = parser.parse_args()
    if args.search:
        start = time.time()
        prev_false = 100000
        best_iter = 0
        best_lambda = 0
        max_iter = [300, 600, 1000, 1300, 1600, 2000]
        lambda_ = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        for iter in max_iter:
            for l in lambda_:
                start_inner = time.time()
                curr_false = main(iter, l)
                if curr_false < prev_false:
                    prev_false = curr_false
                    best_iter = iter
                    best_lambda = l
                end_inner = time.time()
                print("Duration for {} iteration and lambda {}: {}".format(iter, l, end_inner - start_inner))
        print("Best number of iteration: {}\nBest value of lambda: {}".format(best_iter, best_lambda))
        end = time.time()
        print("Duration of the search: {}".format(end - start))
    else:
        main()
