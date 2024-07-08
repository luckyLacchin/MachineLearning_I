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
        weighted_sum = inner_prod(self.w, x, use_numpy=True) - self.threshold
        y = np.sign(weighted_sum)
        if y == 0:
            y = 1
        return y
        # <END Your code here>

    # P.2.3 - Implement a function to randomize the weights and the threshold
    def set_random_params(self):
        # <START Your code here>
        self.w = np.random.random(self.w.shape)
        self.w *= 2
        self.w -= 1
        self.threshold = np.random.random(1)[0]
        self.threshold *= 2
        self.threshold -= 1
        # <END Your code here>


    # P.2.4 - Implement a function to check whether the MCP neuron represents a specific Boolean Function
    def is_bf(self, bf):
        fail = False
        # <START Your code here>
        for x in bf.keys():
            y = bf[x]
            pred = self.forward(x)
            if y != pred:
                fail = True
                break
        # <END Your code here>
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
    # <START Your Code Here>
    bf = []
    def gen_binary_permutations(n, max_samples=0):
        perm = []
        n_samples = min(len(np.arange(2 ** n, 2 ** (n + 1))), max_samples)
        if n_samples > 0:
            rng = default_rng()
            if n < 5:
                numbers = rng.choice(np.arange(2 ** n, 2 ** (n + 1)), replace=False, size=n_samples)
            else:
                number_range = 2 ** (n + 1) - 2 ** n
                numbers = np.random.random(size=n_samples)
                numbers *= number_range
                numbers += 2 ** n
                numbers = numbers.astype(np.int64)
        else:
            numbers = range(2 ** n, 2 ** (n + 1))
        for i in numbers:
            bin_str = bin(i)[3:]
            bin_list = [int(bin_str[i]) for i in range(len(bin_str))]
            perm.append(bin_list)
        perm = np.array(perm)
        perm *= 2
        perm -=1
        return perm

    in_data = gen_binary_permutations(dim)
    # print(in_data)
    out_data = gen_binary_permutations(len(in_data), max_samples=max_samples)
    # print(out_data)
    n_bool_func = len(out_data)
    print(f"Number of Boolean functions for dim={dim}: {n_bool_func}")
    for y in out_data:
        this_f = OrderedDict()
        for i, x in enumerate(in_data):
            this_f[tuple(x)] = y[i]
        bf.append(this_f)
    # <END Your Code Here>
    return bf

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

    # P.2.4 - Test your is_bf function
    bf_1 = ({(-1, -1): -1, (-1, 1): -1, (1,-1): -1, (1, 1): 1})
    bf_2 = ({(-1, -1): 1, (-1, 1): 1, (1, -1): -1, (1, 1): 1})
    mcp_1 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_1 represents bf_1.
    #  That is, for all four inputs the output must be correct.
    mcp_1.w = np.array([1,1])
    mcp_1.threshold = 1
    correct = mcp_1.is_bf(bf_1)
    print(f"mcp_1 represents bf_1: {correct}")

    mcp_2 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_2 represents bf_2
    #  That is, for all four inputs the output must be correct.
    mcp_2.w = np.array([-1,1])
    mcp_1.threshold = 0
    correct = mcp_2.is_bf(bf_2)
    print(f"mcp_2 represents bf_2: {correct}")

    # P2.5 - Implement an evaluation script to estimate how many Boolean functions can be approximated with a MCP neuron.
    n_inputs_to_test = range(3,5)

    # TODO: Overwrite this succ_rates list. This list should contain, for each number of inputs,
    #  the fraction of linear threshold functions, i.e., the number of Boolean functions that can be approximated
    #  with a MCP neuron. For comparison, the numbers provided in the lecture are given in the succ_rates_lecture list.
    succ_rates = [0] * len(n_inputs_to_test)
    succ_rates_lecture = [1, 0.88, 0.5, 0.06][:len(n_inputs_to_test)]
    # <START Your Code Here>
    max_samples_to_test = 70000
    max_guesses = 5000
    data = {}
    for n_inputs in n_inputs_to_test:
        data[n_inputs] = {}
        max_samples = max_samples_to_test
        data[n_inputs] = {"all_data": [], "successes": 0, "avg_guesses_if_succ": 0, "succ_rate": 0}
        bf = generate_boolean_functions(n_inputs, max_samples=max_samples)

        for func in tqdm(bf):
            mcp = MCPNeuron(n_inputs)
            n_guesses = 0
            while n_guesses < max_guesses:
                mcp.set_random_params()
                is_bf = mcp.is_bf(func)
                n_guesses += 1
                if is_bf:
                    break
            success = n_guesses < max_guesses
            result = {"bf": bf, "n_guesses": n_guesses, "success": success}
            data[n_inputs]["all_data"].append(result)
            data[n_inputs]["successes"] += success
            data[n_inputs]["avg_guesses_if_succ"] += (n_guesses * success)

        if data[n_inputs]["successes"] > 0:
            data[n_inputs]["avg_guesses_if_succ"] /= data[n_inputs]["successes"]
        else:
            data[n_inputs]["avg_guesses_if_succ"] = 0
        data[n_inputs]["succ_rate"] = data[n_inputs]["successes"] / len(data[n_inputs]["all_data"])

    succ_rates = [data[n_inputs]["succ_rate"] for n_inputs in n_inputs_to_test]
    avg_guesses = [data[n_inputs]["avg_guesses_if_succ"] for n_inputs in n_inputs_to_test]
    # <END Your Code Here>

    # Plot the approximation success rates from your experiments and from the lecture
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(n_inputs_to_test, succ_rates)
    plt.plot(n_inputs_to_test, succ_rates_lecture)
    plt.savefig('approx.png')

    # Not part of the exercise but interesting: Plot how many guesses were required to find a MCP parameterization.
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(n_inputs_to_test, avg_guesses)
    # plt.show()
    plt.savefig('guesses.png')
    print("Done")


