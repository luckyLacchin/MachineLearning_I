import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random

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

    # P3.2 :Implement a forward-pass function for the perceptron
    def forward(self, x):
        """
        :param x: The input values
        :return: The output
        """
        y = 0 # TODO: overwrite this y with the correct value
        # <START Your code here>

        # <END Your code here>
        return y

    # P3.3: Implement a single learning step for a single data point
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

if __name__ == '__main__':

    # Generate a dataset
    n_samples = 100
    C, NotC = gen_lin_sep_dataset(n_samples=n_samples, guarantee_separable=True)

    # Compute radius of a dataset, i.e., the distance from the origin to the farthest point.
    # Therefore, complete the code for get_radius in helper_functions.py
    D = np.concatenate((C, NotC))
    np.random.shuffle(D) # Randomize the contents of D.
    R = get_radius(D)

    # P3.2 - Implement the perceptron convergence algorithm by Rosenblatt

    # Instantiate the perceptron with input dimension 2
    p = Perceptron(2)

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



    # P3.3 - Evaluate the Rosenblatt algorithm
    # Instantiate a new perceptron with input dimension 2
    p = Perceptron(2)
    accuracy = 0.0
    # <START your code here>
    # Perform the train/test split

    # <END your code here>

    # Initialize the plot for training
    # figure_data = init_2D_linear_separation_plot(C_train, NotC_train, p.w, R)

    # Train the algorithm on the training set
    corrections_ctr = 0
    max_iter = 100

    for iter_ctr in range(1, max_iter):
        # <START your code here>


        # <END your code here>
        if np.linalg.norm(p.w) > 0: # Condition is required because otherwise the plotting is not correct.
            update_2D_linear_separation_plot(figure_data, p.w, R)

        print(f"Iterations: {iter_ctr} \t Corrections: {corrections_ctr}")

    # Evaluate the algorithm on the testing set
    # Initialize the plot for testing
    # figure_data = init_2D_linear_separation_plot(C_test, NotC_test, p.w, R)
    # <START your code here>

    # <END your code here>

    print(f"Percentage of correctly classified points in the testing set: {accuracy * 100}%")