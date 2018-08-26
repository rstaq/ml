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


def main():
    data = sio.loadmat("data.mat")
    X = data.get('X')
    y = data.get('y')
    # plot_rand_digits(X, y)


if __name__ == '__main__':
    main()
