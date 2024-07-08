import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from matplotlib import pyplot
from math import cos, sin, atan
import math
import sys
import os
import time
import cv2
# Python program showing
# abstract base class work

from abc import ABC, abstractmethod

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from conti_nn_util import MagicPytorchContiNN
import torch.optim as optim
# from approx_err_util import PrepareData
from torch.utils.data import Dataset, DataLoader
from collections import deque


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def l2Loss(yhat, y):
    n_data = len(y)
    errs = yhat - y
    sqErrs = errs ** 2
    loss = np.sum(sqErrs) / n_data
    return loss


# A  general parent class for function approximation. Inherits from Abstract Base Class (ABC).
class FunctionApproximator(ABC):

    def __init__(self):
        self.loss_function = l2Loss
        self.lr = 0.1

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_gradients(self, x, y):
        pass

    @abstractmethod
    def update_parameters(self, this_lr):
        pass

    def plots_to_vid(self, image_folder, video_prefix='vid', del_images=True):
        approx_images = [img for img in os.listdir(image_folder) if img.endswith("_approx.png")]
        loss_images = [img for img in os.listdir(image_folder) if img.endswith("_loss.png")]
        grad_images = [img for img in os.listdir(image_folder) if img.endswith("_grad.png")]
        weight_images = [img for img in os.listdir(image_folder) if img.endswith("_weights.png")]

        n_frames = len(approx_images)
        vid_duration = 15
        fps = max(n_frames // vid_duration, 1)

        assert len(approx_images) == len(loss_images) == len(grad_images) == len(weight_images)

        approx_frame = cv2.imread(os.path.join(image_folder, approx_images[0]))
        loss_frame = cv2.imread(os.path.join(image_folder, loss_images[0]))
        grad_frame = cv2.imread(os.path.join(image_folder, grad_images[0]))
        weight_frame = cv2.imread(os.path.join(image_folder, weight_images[0]))
        joint_frame = cv2.hconcat([cv2.vconcat([approx_frame, loss_frame]), cv2.vconcat([grad_frame, weight_frame]) ])
        joint_height, joint_width, joint_layers = joint_frame.shape

        joint_video = cv2.VideoWriter(os.path.join(image_folder, f"{video_prefix}_joint.avi"), 0, fps,
                                     (joint_width, joint_height))
        for frame_idx in range(n_frames):
            approx_frame = cv2.imread(os.path.join(image_folder, approx_images[frame_idx]))
            grad_frame = cv2.imread(os.path.join(image_folder, grad_images[frame_idx]))
            loss_frame = cv2.imread(os.path.join(image_folder, loss_images[frame_idx]))
            weight_frame = cv2.imread(os.path.join(image_folder, weight_images[frame_idx]))
            joint_frame = cv2.hconcat([cv2.vconcat([approx_frame, loss_frame]), cv2.vconcat([grad_frame, weight_frame]) ])
            joint_video.write(joint_frame)
        cv2.destroyAllWindows()
        joint_video.release()

        if del_images:
            for image in approx_images:
                os.remove(os.path.join(image_folder, image))
            for image in loss_images:
                os.remove(os.path.join(image_folder, image))
            for image in grad_images:
                os.remove(os.path.join(image_folder, image))
            for image in weight_images:
                os.remove(os.path.join(image_folder, image))

    def plot_eval(self, x_interval, func, iteration, n_datapoints, loss_hist, grad_hist, weight_hist, figure_path=None, show_plot=False):
        interval_step = (x_interval[1] - x_interval[0]) / n_datapoints
        xs = np.expand_dims(np.arange(x_interval[0], x_interval[1], interval_step), 1)
        ys = np.array([func(x) for x in xs])[:,0]

        yshat = self.forward(xs)
        losses = self.loss_function(ys, yshat)
        loss_metric = np.mean(losses)
        loss_hist.append(loss_metric)
        iter_str = f"{iteration}".zfill(5)
        fig_prefix = f"iteration {iter_str}__loss {loss_metric:.5f}__func {func.__name__} {self.__name__}"

        figsize = (8, 5)

        figure_filename = f"{fig_prefix}_approx.png"
        figure_title = f"{fig_prefix}"

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(xs, yshat, label="Approximator")
        plt.plot(xs, ys, label="Original function")
        plt.legend()
        plt.title(figure_title)
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, figure_filename))
        if show_plot is True:
            plt.show()
        plt.close('all')

        fig, ax = plt.subplots(figsize=figsize)
        plt.yscale("log")
        plt.plot(np.arange(len(loss_hist)), loss_hist, label="L2 Loss")
        plt.legend()
        plt.title(f'{figure_title}')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, figure_filename[:-4] + "_loss.png"))
        if show_plot is True:
            plt.show()
        plt.close('all')

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(np.arange(len(grad_hist)), grad_hist, label="Mean weight gradients")
        plt.legend()
        plt.title(f'{figure_title}')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, figure_filename[:-4] + "_grad.png"))
        if show_plot is True:
            plt.show()
        plt.close('all')

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(np.arange(len(weight_hist)), weight_hist, label="Mean weights")
        plt.legend()
        plt.title(f'{figure_title}')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, figure_filename[:-4] + "_weights.png"))
        if show_plot is True:
            plt.show()
        plt.close('all')
        return loss_metric

    # Implement a function to update the neuron's weight and threshold parameters using GD
    def fit_osgd(self, func, x_start, x_end, n_train_datapoints, figure_path=None, loss_threshold=0.0, plot_every_n_data=5, min_lr=None):
        """
        :param func: The function to approximate
        :param n_updates: The number of parameter updates
        :param x_start: The interval start of the range to train
        :param x_end: The interval end of the range to train
        :param n_train_datapoints: Number of datapoints in the interval
        :return: The L2-error for the best parameters
        """

        mean_weights = []
        mean_weight_gradients = []
        losses = []
        last_loss = self.plot_eval([x_start, x_end], func, 0, 20, losses, mean_weight_gradients, mean_weights, figure_path=figure_path, show_plot=False)
        for iteration in tqdm(range(1, n_train_datapoints + 1)):
            if self.lr is not None:
                this_lr = self.lr
            else:
                this_lr = 1/iteration
            if min_lr is not None:
                this_lr = max(this_lr, min_lr)

            x = np.random.uniform(low=x_start, high=x_end, size=(1, 1))
            y = func(x)[0]
            # <START Your code here>
            self.compute_gradients(x, y)
            self.update_parameters(lr=this_lr)
            # <END Your code here>

            if iteration % plot_every_n_data == 0:
                # print(f"This lr: {this_lr}")
                eval_loss = self.plot_eval([x_start, x_end], func, iteration, 20, losses, mean_weight_gradients, mean_weights, figure_path,
                                           show_plot=False)
                last_loss = eval_loss
                if eval_loss <= loss_threshold:
                    break

        vid_prefix = f"func {func.__name__} -- {self.__name__}"
        self.plots_to_vid(figure_path, video_prefix=vid_prefix, del_images=True)


##############################################
# P6.1 - A continuous neuron with GD fitting #
##############################################
class ContinuousNeuron(FunctionApproximator):
    def __init__(self, n_inputs, lr=None):
        super().__init__()
        self.lr = lr
        self.w = np.ones(n_inputs)
        self.theta = 0
        self.loss_function = l2Loss
        # Randomly initialize weights and threshold
        self.set_random_params()
        self.__name__ = f"ContinuousNeuron_{n_inputs}"
        # The gradients of the weights
        self.dw = np.zeros(n_inputs)
        # The gradient of the threshold theta
        self.dtheta = 0

        # Some data for debugging
        self.smallest_loss = np.inf

    # Implement a forward-pass function for a continuous neuron
    def forward(self, x):
        weighted_sum = np.inner(self.w, x) - self.theta
        y = sigmoid(weighted_sum)
        return y

    def get_mean_weight_gradient(self):
        return np.mean(self.dw)

    def get_mean_weight(self):
        return np.mean(self.w)

    def compute_gradients(self, x, y):
        """
        :param x: The input data
        :param y: The ground truth output.
        :return: Nothing
        Computes the gradients and sets self.dw and self.b
        """
        # TODO: Overwrite self.w and self.dtheta with the actual gradients given as the derivative of the loss function.
        self.dw = np.zeros_like(self.w)
        self.dtheta = 0
        # <START Your code here>
        # Forward pass to get prediction
        y_pred = self.forward(x)
        
        # Compute the error
        error = y_pred - y

        # Compute the gradients using the chain rule
        self.dw = error * sigmoid_derivative(np.dot(self.w, x) - self.theta) * x
        self.dtheta = -error * sigmoid_derivative(np.dot(self.w, x) - self.theta)
         # <END Your code here>

    def update_parameters(self, lr, zero_grad=True):
        """
        Updates self.w and self.b according to the previously computed gradients self.dw and self.dtheta.
        :param lr: The learning rate
        :param zero_grad: Whether to zero the gradients after the update.
        :return:  nothing
        """
        # <START Your code here>
        # Update the weights
        self.w -= lr * self.dw
        
        # Update the threshold
        self.theta -= lr * self.dtheta
        # <END Your code here>

    # Implement a function to randomize the weights and the threshold theta
    def set_random_params(self, magnitude=1.0):
        self.w = np.random.random(self.w.shape)
        self.w *= 2 * magnitude
        self.w -= 1 * magnitude

        self.theta = np.random.random(1)[0]
        self.theta *= 2 * magnitude
        self.theta -= 1 * magnitude


    def get_params(self):
        return {"w": self.w, "theta": self.theta}

    def set_params(self, params):
        self.w = params["w"]
        self.theta = params["theta"]

    # Implement a function to guess the neuron's weight and threshold parameters given a function
    def fit_random_guess(self, func, n_updates, x_start, x_end, n_train_datapoints):
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
        for n_update in tqdm(range(n_updates)):
            self.set_random_params(magnitude=50)
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
        print(f"theta: {self.theta}")


###############################################
# P6.2 - A continuous network with GD fitting #
###############################################
class ContinuousNetwork(FunctionApproximator):
    def __init__(self, n_inputs, n_hidden, lr=None):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.w = np.ones(n_hidden)
        self.theta = 0.0
        # Add hidden layer
        self.hidden = []
        for _ in range(n_hidden):
            self.hidden.append(ContinuousNeuron(n_inputs))
        self.__name__ = f"ContinuousNetwork_{n_inputs}_{n_hidden}"
        self.set_random_params()
        self.lr = lr
        self.loss_function = l2Loss
        self.dw = np.zeros(n_hidden)
        self.dtheta = 0.0
        return

    def forward(self, x):
        """
        :param x: The input values
        :return: The output
        """
        hidden_out = np.array([h.forward(x) for h in self.hidden])
        weighted_sum = np.inner(self.w, hidden_out.T) - self.theta
        # y = sigmoid(weighted_sum)
        y = weighted_sum
        return y

    def get_mean_weight_gradient(self):
        dws = list(self.dw)
        for h in self.hidden:
            dws += list(h.dw)
        mean_dw = np.mean(dws)
        return mean_dw

    def get_mean_weight(self):
        ws = list(self.w)
        for h in self.hidden:
            ws += list(h.w)
        mean_w = np.mean(ws)
        return mean_w

    def set_random_params(self, magnitude=1.0):

        self.w = np.random.random(self.w.shape)
        self.w *= 2 * magnitude
        self.w -= 1 * magnitude
        self.theta = np.random.random(1)[0]
        self.theta *= 2 * magnitude
        self.theta -= 1 * magnitude

        for h in range(self.n_hidden):
            self.hidden[h].set_random_params(magnitude=magnitude)


    def get_params(self):
        return {"w": self.w, "theta": self.theta, "hidden": [h.get_params for h in self.hidden]}

    def set_params(self, params):
        self.w = params["w"]
        self.theta = params["theta"]
        for h_idx, h in enumerate(self.hidden):
            self.hidden[h_idx].set_params(params["hidden"][h_idx])

    def fit_random_guess(self, func, n_updates, x_start, x_end, n_train_datapoints):
        """
        :param func: The function to approximate
        :param n_updates: The number of parameter updates
        :param x_start: The interval start of the range to train
        :param x_end: The interval end of the range to train
        :param n_train_datapoints: Number of datapoints in the interval
        :return: The L2-error for the best parameters
        """
        best_params_err = 0 # TODO: Overwrite the 0

        x_step = (x_end - x_start) / n_train_datapoints
        best_param_err = np.inf
        best_params = self.get_params()
        for n_guess in tqdm(range(n_updates)):
            self.set_random_params(magnitude=50)
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

        return best_params_err

    def update_parameters(self, lr, zero_grad=True):
        """
        Updates self.w and self.b according to the previously computed gradients self.dw and self.dtheta.
        This also involves the hidden units.
        :param lr: The learning rate
        :param zero_grad: Whether to zero the gradients after the update.
        :return:  nothing
        """
        # <START Your code here>

        # <END Your code here>

    def compute_gradients(self, x, y):
        """
        :param x: The input data
        :param y: The ground truth output.
        :return: Nothing
        Computes the gradients and sets self.dw and self.dtheta, and the gradients of the hidden units.
        Consider Sec. 3.7 in the script for details.
        """
        # TODO: Overwrite self.w and self.dtheta with the actual gradients given as the derivative of the loss function.
        self.dw = np.zeros_like(self.w)
        self.dtheta = 0


        # <START Your code here>
        # Consider the script, page 60
        # Step 1: Evaluate all o_j, and all sigma'(net_j)

        # Step 2: Compute \delta_h for all neurons, starting with output layer.

        # Step 3: Adjust weights. Here: set self.dw and self.hidden[].dw

        # <END Your code here>

    def display_parameters(self):
        print(f"w: {self.w}")
        print(f"theta: {self.theta}")
        for h in self.hidden:
            h.display_parameters()