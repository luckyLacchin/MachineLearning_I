import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
from numpy.random import default_rng
from tqdm import tqdm
from collections import OrderedDict
from mcp import MCPNeuron, generate_boolean_functions, gen_lin_sep_dataset
import json
from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
from plot_functions import init_2D_linear_separation_plot, update_2D_linear_separation_plot, plot_3d_surface

class MCPNetwork:
    def __init__(self, n_inputs, n_hidden, w=None, threshold=0):
        # Set weights of output neuron (not of hidden neurons)
        if w is None:
            self.w = np.zeros(n_hidden)
        else:
            assert len(self.w) == n_hidden, "Error, number of inputs must be the same as the number of hidden units."
            self.w = w
        # Set threshold of output neuron (not of hidden neurons)
        self.threshold = threshold

        # <START Your code here>
        # Add the hidden units
        # Create a list of hidden MCP units according to n_inputs and n_hidden.

        # <END Your code here>
        return

    def forward(self, x):
        """
        :param x: The input values
        :return: The output in the interval [-1,1]
        """
        # <START Your code here>

        # <END Your code here>

    # Implement a function to randomize the weights and the threshold
    def set_random_params(self):
        # <START Your code here>

        # <END Your code here>
        return

    # Implement a function to check whether the MCP neuron represents a specific Boolean Function
    def is_bf(self, bf):
        fail = False
        # <START Your code here>

        # <END Your code here>
        return not fail


if __name__ == '__main__':
    # Implement an evaluation script to estimate how many Boolean functions can be approximated with a MCP network, in dependence on the number of hidden units.
    n_inputs_to_test = range(1,5)
    n_hidden_to_test = range(1,10)

    # For n=3 inputs: h_n = 3/(n+2) * 2^n = 3/5 * 2^3 = 4.8 hidden units
    # For n=4 inputs: h_n = 3/(n+2) * 2^n = 3/6 * 2^4 = 1/2 *  16 = 8 hidden units

    # TODO: Overwrite this succ_rates dictionary with the correct success rate values. This dictionary should contain, for each number of inputs and for each number of hidden units,
    #  the fraction of linear threshold functions, i.e., the number of Boolean functions that can be approximated
    #  with a MCP network.
    succ_rates = {n_inputs: {n_hidden: np.nan for n_hidden in n_hidden_to_test} for n_inputs in n_inputs_to_test}
    avg_guesses = {n_inputs: {n_hidden: np.nan for n_hidden in n_hidden_to_test} for n_inputs in n_inputs_to_test}
    max_samples_to_test = 200
    max_guesses = 150000
    os.makedirs('mcp_data', exist_ok=True)

    data = {}
    for n_inputs in n_inputs_to_test:
        # <START Your Code Here>

        # <END Your Code Here>

        # Dump data as json
        with open('mcp_data/succ_rates.json', 'w') as fp:
            json.dump(succ_rates, fp)
        with open('mcp_data/avg_guesses.json', 'w') as fp:
            json.dump(avg_guesses, fp)
        # Plot the success rates and number of guesses as 3D surface plot.
        try:
            for z_ang_hori in [0, 30, 60, 90]:
                for z_ang_vert in [0, 45]:
                    plot_3d_surface(succ_rates, filename=f'mcp_data/succ_rates_{z_ang_hori}_{z_ang_vert}.png', x_label='# hidden', y_label='# inputs', z_label='fraction of functions represented', rotate_z=(z_ang_hori, z_ang_vert))
                    plot_3d_surface(avg_guesses, filename=f'mcp_data/avg_guesses_{z_ang_hori}_{z_ang_vert}.png', x_label='# hidden', y_label='# inputs', z_label='average guesses of parameters until success', rotate_z=(z_ang_hori, z_ang_vert))
        except Exception as e:
            print(f"Error: {e}")

    print("Done")


