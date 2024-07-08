import math
import sys
import os
import time

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from conti_nn_util import MagicPytorchContiNN
import torch.optim as optim


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


########################################
# P5.1 - Implement 5 example functions #
########################################
def function_1(x):
    y = 0 # TODO: Overwrite y appropriately
    # <START Your code here>
    y = sigmoid(x)
    # <END Your code here>
    return np.array([y])


def function_2(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = math.sin(x)
    # <END Your code here>
    return np.array([y])


def function_3(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = sigmoid(x) + math.sin(x)
    # <END Your code here>
    return np.array([y])


def function_4(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = math.pow(x / 6, 2) - math.pow(x / 8, 4)
    # <END Your code here>
    return np.array([y])


def function_5(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = math.sin(x) * math.sqrt(abs(x))
    # <END Your code here>
    return np.array([y])


##############################
# P5.2 - A continuous neuron #
##############################
class ContinuousNeuron:

    def __init__(self, n_inputs, w=None, threshold=0):
        self.n_inputs = n_inputs
        if w is None:
            self.w = np.zeros(n_inputs)
        else:
            assert len(self.w) == n_inputs, "Error, number of inputs must be the same as the number of weights."
            self.w = w
        self.threshold = threshold
        self.__name__ = f"ContinuousNeuron_{n_inputs}"
        self.set_random_params()

    # Implement a forward-pass function for a continuous neuron
    def forward(self, x):
        y = 0 # TODO: Overwrite y appropriately
        # <START Your code here>
        weighted_sum = inner_prod(self.w, x, use_numpy=True) - self.threshold
        y = sigmoid(weighted_sum)
        # <END Your code here>

        return y

    # Implement a function to randomize the weights and the threshold
    def set_random_params(self, magnitude=1.0):
        """
        :param magnitude: The range of the weight w and threshold values goes from -magnitude to +magnitude.
        :return:
        """
        # <START Your code here>
        self.w = np.random.random(self.w.shape)
        self.w *= 2 * magnitude
        self.w -= 1 * magnitude

        self.threshold = np.random.random(1)[0]
        self.threshold *= 2 * magnitude
        self.threshold -= 1 * magnitude
        # <END Your code here>

    def get_params(self):
        return copy.deepcopy(self)

    def set_params(self, params):
        for item in dir(params):
            if item.startswith("__"):
                continue
            val = params.__getattribute__(item)
            self.__setattr__(item, val)
        return

    # Implement a function to guess the neuron's weight and threshold parameters given a function
    def fit(self, func, n_updates, x_start, x_end, n_train_datapoints, magnitude=50):
        """
        :param func: The function to approximate
        :param n_updates: The number of parameter updates
        :param x_start: The interval start of the range to train
        :param x_end: The interval end of the range to train
        :param n_train_datapoints: Number of datapoints in the interval
        :return: The L2-error for the best parameters
        """
        best_params_err = 0  # TODO: Overwrite the 0
        # <START Your code here>
        best_param_err = np.inf
        best_params = self.get_params()
        x_step = (x_end - x_start) / n_train_datapoints
        for n_guess in tqdm(range(n_updates)):
            self.set_random_params(magnitude=magnitude)
            step_errors = []
            for x in np.arange(x_start, x_end, x_step):
                y_approximator = self.forward([x])
                y_orig = func(x)
                step_errors += list(y_approximator - y_orig)
            err = np.linalg.norm(np.array(step_errors))
            if err < best_param_err:
                best_param_err = err
                best_params = self.get_params()
        self.set_params(best_params)
        # <END Your code here>
        return best_params_err

    def display_parameters(self):
        print(f"w: {self.w}")
        print(f"b: {self.threshold}")


###############################
# P5.3 - A continuous network #
###############################
class ContinuousNetwork:
    def __init__(self, n_inputs, n_hidden, w=None, threshold=0):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        if w is None:
            self.w = np.zeros(n_hidden)
        else:
            assert len(self.w) == n_hidden, "Error, number of inputs must be the same as the number of weights."
            self.w = w
        self.threshold = threshold
        # Add hidden layer
        self.hidden = []
        for _ in range(n_hidden):
            self.hidden.append(ContinuousNeuron(n_inputs))
        self.__name__ = f"ContinuousNetwork_{n_inputs}_{n_hidden}"
        self.set_random_params()
        return

    def forward(self, x):
        """
        :param x: The input values
        :return: The output
        """
        y = 0 # TODO: Replace y with the actual forward pass values
        # <START Your code here>
        hidden_out = [h.forward(x) for h in self.hidden]
        y = inner_prod(self.w, hidden_out, use_numpy=True) + self.threshold
        # <END Your code here>
        return y

    def set_random_params(self, magnitude=1.0):
        """
        :param magnitude: The range of the weight w and threshold values goes from -magnitude to +magnitude.
        :return:
        """
        # <START Your code here>
        self.w = np.random.random(self.w.shape)
        self.w *= 2 * magnitude
        self.w -= 1 * magnitude
        self.threshold = np.random.random(1)[0]
        self.threshold *= 2 * magnitude
        self.threshold -= 1 * magnitude

        for h in range(self.n_hidden):
            self.hidden[h].set_random_params(magnitude=magnitude)
        # <END Your code here>

    def get_params(self):
        return copy.deepcopy(self)

    def set_params(self, params):
        for item in dir(params):
            if item.startswith("__"):
                continue
            val = params.__getattribute__(item)
            self.__setattr__(item, val)
        return

    def fit(self, func, n_updates, x_start, x_end, n_train_datapoints, magnitude=50):
        """
        :param func: The function to approximate
        :param n_updates: The number of parameter updates
        :param x_start: The interval start of the range to train
        :param x_end: The interval end of the range to train
        :param n_train_datapoints: Number of datapoints in the interval
        :return: The L2-error for the best parameters
        """
        best_params_err = 0 # TODO: Overwrite the 0
        # <START Your code here>
        x_step = (x_end - x_start) / n_train_datapoints
        best_param_err = np.inf
        best_params = self.get_params()
        for n_guess in tqdm(range(n_updates)):
            self.set_random_params(magnitude=magnitude)
            step_errors = []
            for x in np.arange(x_start, x_end, x_step):
                y_approximator = self.forward([x])
                y_orig = func(x)
                step_errors += list(y_approximator - y_orig)
            err = np.linalg.norm(np.array(step_errors))
            if err < best_param_err:
                best_param_err = err
                best_params = self.get_params()
        self.set_params(best_params)
        # <END Your code here>
        return best_params_err

    def display_parameters(self):
        print(f"w: {self.w}")
        print(f"b: {self.threshold}")
        for h in self.hidden:
            h.display_parameters()


def eval_approximator(func, function_approximator, x_start, x_end, n_eval_steps, figure_filename, figure_path):
    step_errors = []
    xs = []
    ys_approx = []
    ys_orig = []
    x_step = (x_end - x_start) / n_eval_steps
    for x in np.arange(x_start, x_end, x_step):
        y_approx = function_approximator.forward([x])
        # Detaching is required for torch tensors.
        try:
            y_approx = y_approx.detach().numpy()
        except:
            pass
        y_orig = func(x)
        xs.append(x)
        ys_approx.append(y_approx)
        ys_orig.append(y_orig)
        step_errors += list(y_approx - y_orig)
    err = np.linalg.norm(np.array(step_errors))
    # Plot the approximation results
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(xs, ys_approx, label="Approximator")
    plt.plot(xs, ys_orig, label="Original function")
    plt.legend()
    plt.title(f'{func.__name__} - {function_approximator.__name__} - err: {err}')
    plt.savefig(os.path.join(figure_path, figure_filename))
    plt.close()
    return err


if __name__ == '__main__':
    # Code for P5.2 - P5.3
    n_updates = 500
    n_eval_steps = 100
    n_train_datapoints = n_eval_steps
    x_start = - 4 * math.pi
    x_end = 4 * math.pi
    fpath = 'conti_nn_data'
    os.makedirs(fpath, exist_ok=True)
    functions = [function_1, function_2, function_3, function_4, function_5]
    approximatorClassesArgs = []
    approximatorClassesArgs.append([ContinuousNeuron, 1])

    for n_hidden in [1, 2, 4, 8, 16]: # If your machine is too slow, feel free to remove the larger values.
        approximatorClassesArgs.append([ContinuousNetwork, 1, n_hidden])
        approximatorClassesArgs.append([MagicPytorchContiNN, 1, n_hidden])
        pass
    for func in functions:
        print(f"approximating function {func.__name__}")
        best_err = np.inf
        for approxClassArg in approximatorClassesArgs:
            approx = approxClassArg[0](*tuple(approxClassArg[1:]))
            print(f"using approximator  {approx.__name__}")
            fname = f'{str(approx.__name__)}_{str(func.__name__)}.png'
            # if os.path.exists(os.path.join(fpath,fname)):
            #     continue
            approx.fit(func, n_updates, x_start, x_end, n_train_datapoints)
            err = eval_approximator(func, approx, x_start, x_end, n_eval_steps, figure_path=fpath, figure_filename=fname)
            if err < best_err:
                print(f"The approximator {str(approx.__name__)} ist the best so far for function {str(func.__name__)}, with the evaluation error: {err}.")
                best_err = err
                err = eval_approximator(func, approx, x_start, x_end, n_eval_steps, figure_path=fpath,
                                        figure_filename=f"best_{str(func.__name__)}.png")

    # Code for P5.4
    x_start = 0
    x_end = 2 * math.pi

    ##################################################
    ### Approximating sin(x) with a single neuron ###
    ##################################################

    # Learn the sin function again with a single neuron
    guessing_neuron = ContinuousNeuron(1)
    guessing_neuron.fit(function_2, n_updates, x_start, x_end, n_train_datapoints)
    fname = f'guessing_sin_neuron.png'
    err = eval_approximator(function_2, guessing_neuron, x_start, x_end, n_eval_steps, figure_path=fpath,
                            figure_filename=fname)
    # print(f"The error for a single learned neuron to approximate the sin function is {err}")


    manual_neuron = ContinuousNeuron(1)
    # <START Your code: Assign appropriate values to the weight and threshold (replace the 0s)>
    manual_neuron.w[0] = -math.pi
    manual_neuron.w[0] = -2

    manual_neuron.threshold = -8.2
    manual_neuron.threshold = -6
    # <END Your code>
    fname = f'manual_sin_neuron.png'
    err = eval_approximator(function_2, manual_neuron, x_start, x_end, n_eval_steps, figure_path=fpath, figure_filename=fname)
    print(f"The error for a single manually defined neuron to approximate the sin function is {err}")

    ########################################################################
    ### Approximating sin(x) with a network with a single hidden neuron ###
    ########################################################################
    # Learn the sin function again by guessing with a network that has a single hidden unit
    guessing_network_1 = ContinuousNetwork(1,1)
    guessing_network_1.fit(function_2, n_updates, x_start, x_end, n_train_datapoints)
    fname = f'guessing_sin_network_1.png'
    err = eval_approximator(function_2, guessing_network_1, x_start, x_end, n_eval_steps, figure_path=fpath,
                            figure_filename=fname)
    print(f"The error for a randomly guessed network with a single hidden unit is {err}")

    # Learn the sin function again by gradient descent with a network that has a single hidden unit
    sgd_network_1 = MagicPytorchContiNN(1, 1)
    sgd_network_1.fit(function_2, n_updates, x_start, x_end, n_train_datapoints)
    fname = f'sgd_sin_network_1.png'
    err = eval_approximator(function_2, sgd_network_1, x_start, x_end, n_eval_steps, figure_path=fpath,
                            figure_filename=fname)
    print(f"The error for a sgd-learned network with a single hidden unit is {err}")

    manual_network_1 = ContinuousNetwork(1,1)
    # <START Your code: Assign appropriate values to the weights and thresholds (replace the 0s)>
    manual_network_1.w[0] = -2
    manual_network_1.threshold = 1
    manual_network_1.hidden[0].w[0] = math.pi
    manual_network_1.hidden[0].threshold = 3 * math.pi
    # <END Your code>
    fname = f'manual_sin_network_1.png'
    err = eval_approximator(function_2, manual_network_1, x_start, x_end, n_eval_steps, figure_path=fpath,
                            figure_filename=fname)
    print(f"The error for a manually defined network with a single unit in the hidden layer to approximate the sin function is {err}")
    print("Done")


