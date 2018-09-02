from unittest import TestCase
from nn.digit_classification import cost_function
import scipy.io as sio


class TestDigitClassification(TestCase):
    def load_data(self):
        w = sio.loadmat("weights.mat")
        data = sio.loadmat("../data.mat")
        X = data.get('X')
        y = data.get('y')
        theta1 = w.get("Theta1")
        theta2 = w.get("Theta2")
        return X, y, theta1, theta2

    def test_cost_function_without_regularization(self):
        expected = 0.2876
        delta = 0.0001
        X, y, theta1, theta2 = self.load_data()
        J, theta1, theta2 = cost_function(X, y, theta1, theta2)
        TestCase.assertAlmostEqual(self, first=J, second=expected, delta=delta)

    def test_cost_function_with_regularization(self):
        expected = 0.3837
        delta = 0.0001
        X, y, theta1, theta2 = self.load_data()
        J, theta1, theta2 = cost_function(X, y, theta1, theta2, lambda_=1)
        TestCase.assertAlmostEqual(self, first=J, second=expected, delta=delta)