import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
from numpy.random import default_rng
from tqdm import tqdm
from collections import OrderedDict

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
from plot_functions import init_2D_linear_separation_plot, update_2D_linear_separation_plot

class MCPNeuron:
    def __init__(self, n_inputs, w=None, threshold=0):
        if w is None:
            self.w = np.zeros(n_inputs)
        else:
            assert len(self.w) == n_inputs, "Error, number of inputs must be the same as the number of weights."
            self.w = w
        self.threshold = threshold

    # P2.1 - Implement a forward-pass function for a simple MCP neuron
    def forward(self, x):
        """
        :param x: The input values
        :return: The output in the interval [-1,1]
        """
        # <START Your code here>
        y = np.inner(self.w, x) - self.threshold
        return np.sign(y)
        # <END Your code here>
        # return 0 # TODO: Overwrite this with your code

    # P.2.3 - Implement a function to randomize the weights and the threshold
    def set_random_params(self):
        # <START Your code here>
        self.w = np.random.uniform(-1, 1, len(self.w))
        self.threshold = np.random.uniform(-1, 1)
        # <END Your code here>


    # P.2.4 - Implement a function to check whether the MCP neuron represents a specific Boolean Function
    def is_bf(self, bf):
        fail = False
        for x in bf.keys():
            y = bf[x]
            pred = self.forward(x)
            if y != pred:
                fail = True
                break
        return not fail


# P2.2 - Implement a function that generates sets of random Boolean functions given a dimension.
def generate_boolean_functions(dim, max_samples=0):
    """
    :param dim: The input dimension, i.e., the number of Boolean inputs
    :param max_samples: The max. number of functions to return.
    This value is bounded by the possible number of functions for a given input dimension.
    For example, for dim=2, there can only be 16 Boolean functions.
    :return: The functions to return as dictionaries, where keys are tuples of inputs and values are outputs.
    """
    #bf = []
    # <START Your Code Here>
    bf = {}

    # Generate all combinations of inputs
    input_combinations = generate_random_list(dim)

    # Generate random output for each input combination
    for inputs in input_combinations:
        output = random.choice([-1, 1])  # Randomly choose either -1 or 1
        bf[tuple(inputs)] = output
    # <END Your Code Here>
    return bf

def generate_random_list (dim):
    if dim == 1:
        return [[-1], [1]]
    else:
        smaller_combinations = generate_random_list(dim - 1)
        combinations = []
        for combination in smaller_combinations:
            combinations.append(combination + [-1])
            combinations.append(combination + [1])
        return combinations
#i didn't use the hint given about "the index 14 in binary is 1110", in that case i would have used the formula
if __name__ == '__main__':
    # P.2.1 - Test the forward function
    mcp_1 = MCPNeuron(1)
    mcp_1.w = np.array([1])
    mcp_1.threshold = 1
    mcp_1.forward([-1])

    mcp_2 = MCPNeuron(2)
    mcp_2.w = np.array([1, 1])
    mcp_2.threshold = 1
    mcp_2.forward([-1, 1])

    mcp_3 = MCPNeuron(3)
    mcp_3.w = np.array([1, 1, 1])
    mcp_3.threshold = 1
    mcp_3.forward([-1, 1, -1])

    #P.2.2.
    # Test the function by sampling 200 random functions for dim=2 and dim=3
    dim = 2
    num_samples = 200
    sampled_functions_dim_2 = generate_boolean_functions(dim, num_samples)

    dim = 3
    sampled_functions_dim_3 = generate_boolean_functions(dim, num_samples)

    # Print the first few sampled functions for each dimension
    print("Sampled Boolean Functions for Dimension 2:")
    for i, function in enumerate(sampled_functions_dim_2):
        print(f"Function {i + 1}: {function}")

    print("\nSampled Boolean Functions for Dimension 3:")
    for i, function in enumerate(sampled_functions_dim_3):
        print(f"Function {i + 1}: {function}")

    #P2.3
    mcp_debug = MCPNeuron(3)
    mcp_debug.set_random_params();
    print(f"2.3 weights: {mcp_debug.w}")
    print(f"2.3 threshold: {mcp_debug.threshold}")

    # P.2.4 - Test your is_bf function
    bf_1 = ({(-1, -1): -1, (-1, 1): -1, (1,-1): -1, (1, 1): 1})
    bf_2 = ({(-1, -1): 1, (-1, 1): 1, (1, -1): -1, (1, 1): 1})
    mcp_1 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_1 represents bf_1.
    #  That is, for all four inputs the output must be correct.
    mcp_1.w = np.array([1,1])
    mcp_1.threshold = 1
    for k in bf_1.keys():
        out = mcp_1.forward(k)
        if out == bf_1[k]:
            print("Correct!")
        else:
            print("Incorrect")

    mcp_2 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_2 represents bf_2
    #  That is, for all four inputs the output must be correct.
    mcp_2.w = np.array([-1,1])
    mcp_2.threshold = -1
    for k in bf_2.keys():
        out = mcp_2.forward(k)
        if out == bf_2[k]:
            print("Correct!")
        else:
            print("Incorrect")
#primo e terzo
    print(mcp_1.is_bf(bf_1))
    print(mcp_2.is_bf(bf_2))

    # P2.5 - Implement an evaluation script to estimate how many Boolean functions can be approximated with a MCP neuron.
    n_inputs_to_test = range(1,5)
    succ_rates = []
    for i in n_inputs_to_test:
        successes = 0
        bf = generate_boolean_functions(i,200)
        mcp_neurons = MCPNeuron(i)
        for _ in range(5000):
            mcp_neurons.set_random_params()
            print(mcp_neurons.is_bf(bf))
            if mcp_neurons.is_bf(bf):
                successes += 1
        success_rate = successes/ 5000
        succ_rates.append(success_rate)
    # TODO: Overwrite this succ_rates list. This list should contain, for each number of inputs,
    #  the fraction of linear threshold functions, i.e., the number of Boolean functions that can be approximated
    #  with a MCP neuron. For comparison, the numbers provided in the lecture are given in the succ_rates_lecture list.
    #succ_rates = [0] * len(n_inputs_to_test)
    print(succ_rates)
    succ_rates_lecture = [1, 0.88, 0.5, 0.06][:len(n_inputs_to_test)]
    # <START Your Code Here>

    # <END Your Code Here>

    # Plot the approximation success rates from your experiments and from the lecture
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(n_inputs_to_test, succ_rates, label='Estimated Success Rate')
    plt.plot(n_inputs_to_test, succ_rates_lecture, label = "Lecture Success Rate")
    plt.xlabel('Number of Inputs')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.show()
    #plt.savefig('approx.png')

    # Not part of the exercise but interesting: Plot how many guesses were required to find a MCP parameterization.
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.plot(n_inputs_to_test, avg_guesses)
    # # plt.show()
    # plt.savefig('guesses.png')
    # print("Done")


