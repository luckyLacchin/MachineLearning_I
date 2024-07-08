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
    y = np.sin(x)
    # <END Your code here>
    return np.array([y])


def function_3(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = sigmoid(x) + np.sin(x)
    # <END Your code here>
    return np.array([y])


def function_4(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = (x / 6) ** 2 - (x / 8) ** 4
    # <END Your code here>
    return np.array([y])


def function_5(x):
    y = 0  # TODO: Overwrite y appropriately
    # <START Your code here>
    y = np.sin(x) * np.sqrt(x)
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
        return y
        # <END Your code here>

    # Implement a function to randomize the weights and the threshold
    def set_random_params(self, magnitude=1.0):
        """
        :param magnitude: The range of the weight w and threshold values goes from -magnitude to +magnitude.
        :return:
        """
        # <START Your code here>
        self.w = np.random.uniform(-magnitude, magnitude, len(self.w))
        self.threshold = np.random.uniform(-magnitude, magnitude)
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
        best_params_err = float('inf')  # Initialize the best error to a large value
        best_params = None

        # Generate 100 homogeneously chosen values in the interval [x_start, x_end]
        x_values = np.linspace(x_start, x_end, n_train_datapoints)

        for _ in range(n_updates):
            # Randomly initialize weights and threshold
            self.set_random_params(magnitude)

            # Compute the neuron's output for the x_values
            neuron_outputs = np.array([self.forward(x) for x in x_values])

            # Compute the original function's output for the x_values
            func_outputs = np.array([func(x) for x in x_values])

            # Compute the L2 norm between the original function and the neuron
            l2_error = np.linalg.norm(neuron_outputs - func_outputs)

            # Check if this is the best error
            if l2_error < best_params_err:
                best_params_err = l2_error
                best_params = self.get_params()

        # Set the parameters to the best found parameters
        self.set_params(best_params)
        # <END Your code here>
        return best_params_err

    def display_parameters(self):
        print(f"w: {self.w}")
        print(f"b: {self.threshold}")


###############################
# P5.2 - A continuous network #
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
        weighted_sum = inner_prod(self.w, hidden_out, use_numpy=True) - self.threshold
        # <END Your code here>
        y = sigmoid(weighted_sum)
        return y

    def set_random_params(self, magnitude=1.0):
        """
        :param magnitude: The range of the weight w and threshold values goes from -magnitude to +magnitude.
        :return:
        """
        # <START Your code here>
        self.w = np.random.uniform(-magnitude, magnitude, len(self.w))
        self.threshold = np.random.uniform(-magnitude, magnitude)
        for h in self.hidden: #it's empty, for this reason there is no recursion!
            h.set_random_params()
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
            # Initialize the best parameters and error
        best_params_err = float('inf')  # Start with an infinitely large error
        best_params = self.get_params()  # Store the initial parameters

        # Generate training data
        x_train = np.linspace(x_start, x_end, n_train_datapoints)  # Create evenly spaced training data points in the interval
        y_train = np.array([func(x) for x in x_train])  # Compute the target function values for the training data

        # Set random initial parameters
        self.set_random_params(magnitude)

        learning_rate = 0.01  # Learning rate for gradient descent

        for update in range(n_updates):  # Loop over the number of updates
            total_error = 0  # Initialize the total error for this update

            # Update weights for each training sample
            for i in range(n_train_datapoints):  # Loop over each training data point
                x = x_train[i]  # Get the input value
                y_true = y_train[i]  # Get the corresponding true output value
                y_pred = self.forward(x)  # Get the predicted output from the network

                # Compute the error
                error = y_true - y_pred  # Calculate the difference between the true and predicted outputs
                total_error += error ** 2  # Accumulate the squared error

                # Compute gradients for weights and threshold
                hidden_outputs = [neuron.forward(x) for neuron in self.hidden]  # Compute the hidden layer outputs
                d_w = -2 * error * np.array(hidden_outputs)  # Compute the gradient for the weights
                d_threshold = 2 * error  # Compute the gradient for the threshold

                # Update weights and threshold
                self.w -= learning_rate * d_w  # Update the weights using the gradient
                self.threshold -= learning_rate * d_threshold  # Update the threshold using the gradient

                # Update hidden neurons' parameters
                for j, neuron in enumerate(self.hidden):
                    neuron.update_params(learning_rate, error, self.w[j])

            # Update best parameters if current error is smaller
            if total_error < best_params_err:
                best_params_err = total_error  # Update the best error
                best_params = self.get_params()  # Store the best parameters

        # Set the best parameters found
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

    # <END Your code>
    fname = f'manual_sin_network_1.png'
    err = eval_approximator(function_2, manual_network_1, x_start, x_end, n_eval_steps, figure_path=fpath,
                            figure_filename=fname)
    print(f"The error for a manually defined network with a single unit in the hidden layer to approximate the sin function is {err}")
    print("Done")


