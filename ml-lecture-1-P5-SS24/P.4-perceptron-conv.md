# Programming Exercise P4. -- Perceptron learning and generalization (15P) 
This exercise is concerned with the perceptron learning algorithm for a single perceptron and the overall concept of supervised learning and generalization. Therefore, you will complete the code in [perceptron.py](perceptron.py). Please make sure that you use the Python files and functions provided in this package and not your own Python implementation from the previous exercise. In particular, make sure that you do not use your own file `mcp.py`, but the one provided here.

## P4.1: Complete the Perceptron class and implement the Rosenblatt algorithm (8P).
We are now looking into perceptrons. Unlike MCP neurons they are not Boolean and, in this version, do not have an activation function and no threshold value. Complete the code in [perceptron.py](perceptron.py) to implement Rosenblatt's perceptron convergence algorithm.  Start by completing the code for the  `forward` function of the class (1P).
Then complete the code for the `learn` function of the Perceptron class (4P) and program a loop to iterate over the `learn` function (2P). Count how many corrections were necessary with the `corrections_ctr` variable (1P). The plotting functions will visualize the learning process for you.   

## P4.2: Evaluate the generalization performance of the Rosenblatt algorithm (7P).
"Generalization" means that a machine learning algorithm should be able to correctly handle also data samples that it has not been trained on. 
The generalization performance can be evaluated by measuring how many data samples that have not been used for training are correctly classified.
To evaluate the generalization performance, split the dataset D generated previously into two sets, such that 15% of the points are in one set, and 85% in the other set (2P). We call the smaller part the training set and the larger part the testing set. Use the Rosenblatt algorithm to train the weights on the training set only (1P). Then test the weights that were determined during the training with the testing sets and calculate the accuracy of the classification of the testing set (4P) (to implement with the function `get_accuracy(...)`). The accuracy is the percentage of data points in the testing set that are correctly classified.  


## P4.3: Extend the Rosenblatt algorithm to consider datasets that are not linearly separable (5P - optional extra task). 
In reality, data are rarely linearly separable. However, it is possible to extend the algorithm, so that it can deal with such data. Your task is to come up yourself with an idea to achieve this. Here are some tips: 

* It might be possible to use the margin $`\gamma`$ for this in some way, but this is not what we have in mind for this task. 
* Do some selfp-responsible research about learning rates, and in particular, adaptive learning rates (using Google, Wikipedia, and Machine Learning websites).
* Think of how you can extend the `learn(...)` function appropriately.