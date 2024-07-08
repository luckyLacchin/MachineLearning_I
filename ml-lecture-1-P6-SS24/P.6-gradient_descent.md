# Programming Exercise 6: Gradient Descent
Gradient descent (GD) is a general method for optimization. In this exercise, you will use online stochastic gradient descent (OSGD) to optimize the weights of a continuous neural network. You will first implement OSGD for a single neuron with a 1-dimensional input. Then you will consider OSGD for a neural network with a single hidden layer.


## P6.1 - Gradient descent for a single neuron (9P)
Consider the class `ContinuousNeuron` in [gd_conti_util.py](gd_conti_util.py). 
First, implement the GD method for a single neuron in the `compute_gradient` function (5P). Then implement `update_parameters` (2P) and insert these functions correctly in `fit_osgd` (2P). 
Test your code by running the respective lines in [gd_conti_nn.py](gd_conti_nn.py) . 

Hint: use the chain rule to derive the gradient formula for a perceptron. The derivative of a logistic sigmoid can be found in the code. 


## P6.2 - Gradient descent for a neural network (7P)
Consider the class `ContinuousNetwork` in [gd_conti_util.py](gd_conti_util.py). 
Implement the GD method for a neural network with one hidden layer in the `compute_gradient` function (5P). 
Also implement `update_parameters` (2P). 
Test your code by running the respective lines in [gd_conti_nn.py](gd_conti_nn.py).

Backpropagation, for us it's not too important tbh!!!


