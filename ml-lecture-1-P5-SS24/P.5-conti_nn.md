
## P5.1 - Implement the following testing functions to approximate as Python functions (2.5P)
1. <img src="https://latex.codecogs.com/svg.latex?f_1 = \sigma(x)">
   (This is trivial, the function is already in the code.)
2. <img src="https://latex.codecogs.com/svg.latex?f_2 = sin(x)">
3. <img src="https://latex.codecogs.com/svg.latex?f_3 = \sigma(x) %2B sin(x)">
4. <img src="https://latex.codecogs.com/svg.latex?f_4 = \left(\frac{x}{6}\right)^2 - \left(\frac{x}{8}\right)^4">
5. <img src="https://latex.codecogs.com/svg.latex?f_5 =  sin(x) \cdot \sqrt(x)">


## P5.2 - Implement a continuous neuron with a forward function and a function to randomly set its weights. Then test its approximation abilities (10P).
Complete the code of the `forward` function (2P) and `set_random_weights` function (2P) of the class [ContinuousNeuron](continuous_nn). 
Also complete the code for the function `fit` of the `ContinuousNeuron` class, where you implement a loop to randomly set `n_updates=500` times the weight and threshold of the neuron as follows (6P): For each of the random guesses compute the output of the neuron for 100 homogeneously or randomly chosen values in the interval <img src="https://latex.codecogs.com/svg.latex?x \in [-4 \pi, 4\pi]">. Compute for each of the 100 values also the output of the original function and determine the L2 norm between the original function and the neuron over the 100 steps. Store the neuron parameters with the smallest average L2-error (hint: use `get_parameters` and `set_parameters` for this purpose). 
Once you have implemented the neuron the provided code will automatically evaluate and plot its approximation performance for different numbers of hidden units.


## P5.3 - Implement a continuous network with a single hidden layer (10P). 
 
Do what you did for the `ContinuousNeuron` class now for the [ContinuousNetwork](continuous_nn) class: Complete the code for the functions `forward` (2P), `set_random_weights` (2P) and `fit` (6P). Then test its approximation abilities. The provided code will automatically evaluate and plot it for you. 


## P5.4 - Can humans do better? (6P)
You have evaluated the approximation capabilities of continual neurons and networks for the random guessing approach and the gradient descent approach implemented in Pytorch. However, as you may assume, the results are not optimal. For this exercise, take the function <img src="https://latex.codecogs.com/svg.latex?sin(x)"> with <img src="https://latex.codecogs.com/svg.latex?x \in [0, 2 \cdot \pi]"> as an example and see how good you can approximate it when searching parameters by hand. First, find parameters for a single neuron (2P), so that the approximation performance is better than the random guess and the gradient descent approach. Then do the same for a "network" with a single hidden unit (4P). Is your manual approach better than the random/sgd-based methods?
