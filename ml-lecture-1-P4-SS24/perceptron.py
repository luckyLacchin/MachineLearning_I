import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
import sys

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
from plot_functions import init_2D_linear_separation_plot, update_2D_linear_separation_plot

class Perceptron:
    def __init__(self, n_inputs, w=None, threshold=0):
        if w is None:
            self.w = np.zeros(n_inputs)
        else:
            assert len(self.w) == n_inputs, "Error, number of inputs must be the same as the number of weights."
            self.w = w
        self.threshold = threshold

    # P4.1 :Implement a forward-pass function for the perceptron
    def forward(self, x):
        """
        :param x: The input values
        :return: The output
        """
        y = 0 # TODO: overwrite this y with the correct value
        # <START Your code here>

        # <END Your code here>
        return y

    # P4.2: Implement a single learning step for a single data point
    def learn(self, data_sample, C, NotC):
        """
        :param: A data sample
        :param: The set of positive data points
        :param: The set of negative data points
        :return: True if the weights of the perceptron have changed, and false otherwise
        """
        weights_changed = False
        # <START Your code here>

        # <END Your code here>
        return weights_changed

    # P4.3: Implement a single learning step for a single data point.
    # Also update the weights if no update is necessary for classification but if this would
    # lead to a larger max. margin gamma.
    def get_current_margin(self, C, NotC):
        D = np.concatenate((C, NotC))
        smallest_dist = sys.float_info.max
        for data in D:
            dist = np.abs(inner_prod(self.w, data)) / np.linalg.norm(self.w)
            smallest_dist  = min(dist, smallest_dist)
        return smallest_dist



    def get_accuracy(self, D, C, NotC):
        accuracy = 0
        # <START your code here>

        # <END your code here>
        return accuracy

if __name__ == '__main__':

    # Generate a dataset
    n_samples = 100
    C, NotC = gen_lin_sep_dataset(n_samples=n_samples, guarantee_separable=True)

    # Compute radius of a dataset, i.e., the distance from the origin to the farthest point.
    # Therefore, complete the code for get_radius in helper_functions.py
    D = np.concatenate((C, NotC))
    np.random.shuffle(D) # Randomize the contents of D.
    R = get_radius(D)

    # P4.1 - Implement the perceptron convergence algorithm by Rosenblatt

    # # Instantiate the perceptron with input dimension 2
    p = Perceptron(2)
    #
    # Initialize the plot
    figure_data = init_2D_linear_separation_plot(C, NotC, p.w, R)

    corrections_ctr = 0
    max_iter = 100

    for iter_ctr in range(1, max_iter):
        # <START your code here>

        # <END your code here>
        if np.linalg.norm(p.w) > 0: # Condition is required because otherwise the plotting is not correct.
            update_2D_linear_separation_plot(figure_data, p.w, R)

        print(f"Iterations: {iter_ctr} \t Corrections: {corrections_ctr}")


    test_accuracy_no_max_margin = 0
    test_accuracy_max_margin = 0
    train_accuracy_no_max_margin = 0
    train_accuracy_max_margin = 0

    # P4.2 - Evaluate the Rosenblatt algorithm
    # Instantiate a new perceptron with input dimension 2
    p = Perceptron(2)
    accuracy = 0.0
    C_train = np.zeros_like(D)
    C_test = np.zeros_like(D)
    NotC_train = np.zeros_like(D)
    NotC_test = np.zeros_like(D)
    D_train = np.zeros_like(D)
    D_test = np.zeros_like(D)

    # <START your code here>
    # TODO: Overwrite C_train, NotC_train, C_test and NotC_test and D_train, D_test with appropriate split data.


    # <END your code here>



    # Train the algorithm on the training set
    corrections_ctr = 0
    max_iter = 200
    figure_data = init_2D_linear_separation_plot(C_train, NotC_train, p.w, R)
    for iter_ctr in range(1, max_iter):
        # <START your code here>

        # <END your code here>
        if np.linalg.norm(p.w) > 0 and weights_changed: # Condition is required because otherwise the plotting is not correct.
            update_2D_linear_separation_plot(figure_data, p.w, R)

        gamma = p.get_current_margin(C_train, NotC_train)
        print(f"Iterations: {iter_ctr} \t Corrections: {corrections_ctr} \t Gamma: {gamma}")
    # Show the plot for training

    # Evaluate the algorithm on the training set
    train_accuracy = p.get_accuracy(D_train, C_train, NotC_train)
    train_accuracy_no_max_margin += train_accuracy
    print(f"Percentage of correctly classified points in the training set: {train_accuracy * 100}%")
    # Evaluate the algorithm on the testing set
    test_accuracy = p.get_accuracy(D_test, C_test, NotC_test)
    print(f"Percentage of correctly classified points in the testing set: {test_accuracy * 100}%")

    # Show the plot for testing
    figure_data = init_2D_linear_separation_plot(C_test, NotC_test, p.w, R)



