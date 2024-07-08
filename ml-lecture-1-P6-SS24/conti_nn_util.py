import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from matplotlib import pyplot
from math import cos, sin, atan


class MagicPytorchContiNN(nn.Module):

    def __init__(self, n_inputs, n_hidden, w=None, threshold=0, optimizer=optim.SGD):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.__name__ = f"MagicPytorchContiNN_{optimizer.__name__}_{n_inputs}_{n_hidden}"
        self.hidden_layer = nn.Linear(n_inputs, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.optimizer = optimizer(self.parameters(), lr=0.01)
        self.optimizer.zero_grad()
        self.loss_function = nn.MSELoss()
        return

    def forward(self, x):
        """
        :param x: The input values
        :return: The output
        """
        x = th.asarray(x, dtype=th.float32)
        hidden_activations = F.sigmoid(self.hidden_layer(x))
        output = self.output(hidden_activations)
        return output

    def display_parameters(self):
        params = list(self.parameters())
        print(len(params))
        for p in params:
            print(p)

    def fit(self, func, n_updates, x_start, x_end, n_train_datapoints):
        x_step = (x_end - x_start) / n_train_datapoints
        inputs = th.arange(x_start, x_end, x_step, dtype=th.float32)
        inputs = inputs.reshape(inputs.size()[0],1)
        ys_orig = th.asarray([func(x) for x in inputs], dtype=th.float32)

        for n_guess in tqdm(range(n_updates)):
            ys_approximator = self(inputs)
            loss = self.loss_function(ys_approximator, ys_orig)
            loss.backward()
            self.optimizer.step()

