# Programming Exercise P3. -- Two-layer MCP networks and linear separability of the perceptron (26P)
This exercise is concerned with two topics:

First, you will build on the previous exercise P2 and extend it to a network of McCulloch-Pitts neurons with a single layer of hidden units. Then you will empirically determine the fraction of Boolean functions it can represent. 
Second, you will implement the perceptron learning algorithm for a single perceptron. 

For this exercise, you will complete the code in [mcp_network.py](mcp_network.py) and [perceptron.py](perceptron.py). Please make sure that you use the Python files and functions provided in this package and not your own Python implementation from the previous exercise. In particular, make sure that you do not use your own file `mcp.py`, but the one provided here. It may also work with your solution, but we cannot guarantee it.   

## P3.1: Implement a McCulloch-Pitts network and test its universal representation abilities (11P - split)
In the lecture you learned that a 2-layered network of MCPNeurons can potentially represent all Boolean functions if it contains enough hidden neurons. In this exercise, you will empirically research the relation between the number of hidden units and the number of Boolean functions that a MCP-network can approximate. 
1. Use the class MCPNeuron from the last exercise to implement the constructor (`__init__`) and the `forward` function for the class `MCPNetwork` in [mcp_network.py](mcp_network.py) (3P).
2. Similar to P2.3, fill the function  [set_random_params](mcp_network.py) In the class `MCPNetwork`, add a function to randomly set the weights and thresholds of the whole network (2P). 
3. As in P2.4, complete the function `is_bf` to test whether the MCPNetwork represents a Boolean function (1P).
4. Use the function `generate_boolean_functions` fom P2.2 to generate the set of boolean functions with at least 3 input dimensions (if your computer allows try 4 input dimensions). Then program a loop to test how many Boolean functions for the given dimension can be represented. Therefore, for each Boolean function, set 5000 times random parameters for the network and check if it represents the Boolean function (3P).
5. With the nested loop over inputs and hidden units fill the dictionary `succ_rates` with the appropriate values and view the 3D plot. For your presentation, you should discuss the 3D plot to get the points (2P)

## P3.2: Complete the Perceptron class and implement the Rosenblatt algorithm (8P).
We are now looking into perceptrons. Unlike MCP neurons they are not Boolean and, in this version, do not have an activation function and no threshold value. Complete the code in [perceptron.py](perceptron.py) to implement Rosenblatt's perceptron convergence algorithm.  Start by completing the code for the  `forward` function of the class (1P).
Then complete the code for the `learn` function of the Perceptron class (4P) and program a loop to iterate over the `learn` function (2P). Count how many corrections were necessary with the `corrections_ctr` variable (1P). The plotting functions will visualize the learning process for you.   

## P3.3: Evaluate the generalization performance of the Rosenblatt algorithm (7P).
"Generalization" means that a machine learning algorithm should be able to correctly handle also data samples that it has not been trained on. 
The generalization performance can be evaluated by measuring how many data samples that have not been used for training are correctly classified.
To evaluate the generalization performance, split the dataset D generated previously into two sets, such that 20% of the points are in one set, and 80% in the other set (2P). We call the smaller part the training set and the larger part the testing set. Use the Rosenblatt algorithm to train the weights on the training set only (1P). Then test the weights that were determined during the training with the testing sets (3P). Then calculate the accuracy of the classification of the testing set (1P). The accuracy is the percentage of data points in the testing set that are correctly classified.  


