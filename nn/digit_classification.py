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


if __name__ == '__main__':
    main()
