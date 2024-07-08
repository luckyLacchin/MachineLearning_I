import matplotlib
matplotlib.use('Agg')
import math
import sys
import os
import time

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
# from approx_err_util import BetterPytorchContiNN, EvenBetterPytorchContiNN
import torch.optim as optim
from plot_functions import plot_3d_surface_hidden_interval_approx
from gd_conti_util import ContinuousNeuron, ContinuousNetwork


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


#####################
# Testing functions #
#####################

def function_0(x):
    y = sigmoid(x)
    return y

def function_1(x):
    y = np.exp(np.divide(-np.power(x, 2), 2))
    return y


def function_2(x):
    y = np.sin(x)
    return y


def function_3(x):
    y = np.cos(x)
    return y


def function_4(x):
    y = sigmoid(x) + np.sin(x)
    return y


def function_5(x):
    y = np.power(x / 6, 2) - np.power(x / 8, 4)
    return y


def function_6(x):
    y = np.sin(x) * np.sqrt(abs(x))
    return y


def perform_approx(func, n_hidden, x_start, x_end, lr, n_epochs, n_datapoints, sampling='homogeneous', optimizer='sgd',
                   best_err=None, batch_size=0):
    approx = BetterPytorchContiNN(1, n_hidden, lr=lr, opti_str=optimizer, sampling=sampling, batch_size=batch_size)
    print(f"using approximator  {approx.__name__}")
    fname = f'{str(func.__name__)}_{str(x_start)}-{str(x_start)}_{str(approx.__name__)}.png'
    err = approx.fit(func, n_epochs, x_start, x_end, n_datapoints, figure_path=F_PATH, figure_filename=fname,
                     best_err=best_err)
    return err


####################
# Global variables #
####################

FUNCTIONS = [function_0, function_1, function_2, function_3, function_4, function_5, function_6] # Remove some functions if computing all takes too long.
N_HIDDEN_UNITS = [1, 2, 5, 25, 50, 255] # Remove some values if computing all takes too long.
N_TRAIN_DATAPOINTS = 5000 # Set to a lower value if this takes too long.

# FUNCTIONS = [function_0, function_1, function_2] # Remove some functions if computing all takes too long.
# N_HIDDEN_UNITS = [25]
# N_TRAIN_DATAPOINTS = 2000 # Set to a lower value if this takes too long.
LOSS_THRESHOLD = 0.005
F_PATH = 'gd_conti_NN_data'
PLOT_EVERY_N_DATA = N_TRAIN_DATAPOINTS // 50

if __name__ == '__main__':
    os.makedirs(F_PATH, exist_ok=True)
    for func in FUNCTIONS:
        ############################################
        # P6.1: Approximation with a single Neuron #
        #############################################

        approx = ContinuousNeuron(1, lr=None)
        print(f"Optimizing {approx} for {func}:")
        x_start = -2*math.pi
        x_end = 2*math.pi
        approx.fit_osgd(func, x_start, x_end, n_train_datapoints=N_TRAIN_DATAPOINTS, figure_path=F_PATH,
                            loss_threshold=LOSS_THRESHOLD, plot_every_n_data=PLOT_EVERY_N_DATA, min_lr=0.05)

        ######################################
        # P6.2: Approximation with a Network #
        ######################################

        for n_hidden in N_HIDDEN_UNITS:
            approx = ContinuousNetwork(1, n_hidden, lr=None)
            print(f"Optimizing {approx.__name__} for {func.__name__}:")
            x_start = -2 * math.pi
            x_end = 2 * math.pi
            approx.fit_osgd(func, x_start, x_end, n_train_datapoints=N_TRAIN_DATAPOINTS, figure_path=F_PATH,
                            loss_threshold=LOSS_THRESHOLD, plot_every_n_data=PLOT_EVERY_N_DATA, min_lr=0.02)
