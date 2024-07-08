# Programming Exercise P2. -- McCullochPitts Neurons (24P)
This exercise is concerned with the MCP neurons, and you will complete the code in [mcp.py](mcp.py).

## P2.1: Implement the forward pass of a single MCP unit (3P)
Given is a stub for a Python class that represents a single MCP unit with a parameterizable number of inputs and a parameterizable threshold value. Implement the forward pass of the class, i.e., implement $`y=<w,x> - \vartheta`$ and return y.  

Use your debugger to test the forward pass for 1 input, 2 inputs and 3 inputs. 
  

## P2.2: Implement a function that randomly samples Boolean functions given an input dimension (8P).
Recall that a function is just a set of pairs of inputs and output. Here, we are interested in functions that map combinations of boolean inputs to boolean outputs. Hence, follow these steps:
First, generate the set of possible inputs given a specific dimension. For example, for dim=2, the set of possible inputs is [-1, -1], [-1, 1], [1, -1], [1, 1]. 
Second, compute function to sample a possible ouput for a set of possible inputs. Consider that for the above set of possible inputs, the set of possible outputs is [-1, -1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, -1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, 1], [1, -1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1], [1, 1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]. 
Each element in the set of possible outputs represents one boolean function. For example, [1, 1, -1, 1] represents a boolean function f_14 with f_14(-1,-1) = 1, f_14(-1,1) = 1, f_14(1,-1) = -1 and f_14(1,1) =1. 

Use your debugger to test the function by sampling 200 random functions for dim=2 and dim=3

Hint 1: It helps to represent the index of the possible output as a binary number. For example, the index 14 in binary is 1110. Now, you only need to convert 1110 into a list of integers [1,1,1,-1]. You can generate a string of a binary number `n` using `bin_num_string=bin(n)[3:]`. 

Hint 2: I recommend implementing a helper function to convert an integer (the dimension) into a list of permutations of the interger's binary representation with -1s and 1s, as in Hint 1. E.g. for dim=2, the helper function should return [-1, -1], [-1, 1], [1, -1], [1, 1]. You can use the helper function twice, to first compute the set of possible inputs and then to compute the set of possible outputs for a given input as described above. 

## P2.3: Extend the implementation of the MCP class with a function to randomize the weights and the threshold with values in the interval [-1,1) (2P)
Complete the code for the function [set_random_params](mcp.py). 
Use your debugger to test the function. 

## P2.4: Extend the implementation of the MCP class with a function to check whether the MCP represents a particular Boolean function. (4P)
Complete the code for the function [is_bf](mcp.py) that checks whether a MCP unit represents a Boolean function. [is_bf](mcp.py) accepts as input a Boolean function of P2.2 and returns true if for all possible inputs the correct output is given. 

Test the code for the following Boolean functions:

`bf_1 = ({(-1, -1): -1, (-1, 1): -1, (1,-1): -1, (1, 1): 1})`

`bf_2 = ({(-1, -1): 1, (-1, 1): 1, (1, -1): -1, (1, 1): 1})`

Therefore, initialize two MCPs and manually find weights and thresholds such that one MCP represents `bf_1` and the other represents `bf_2`. The code for this test is already provided in  [mcp.py](mcp.py), you just need to change the values for the weights and the thresholds. 

## P2.5: Collect data to measure the number of guesses required for a MCP to represent a particular Boolean function, in relation to the number of input units. (7P)
Implement a loop over the number of inputs between 1 and 4 (including 4, if your hardware permits, otherwise compute only for 3 inputs). During each iteration, sample 200 Boolean function for the number of inputs, using the function you implemented in P.2.2. Then initialize a MCP neuron and try 5000 times to guess randomly the weights and threshold, using the function you have implemented in P.2.3. For each random guess, check whether the MCP is equivalent to the Boolean function, as implemented in P.2.4.  

For each number of inputs, record the number of Boolean functions that can be represented with the MCP neuron. Display the results in a graph, such that on the x-axis there is the number of inputs and on the y-axis there is the fraction of Boolean functions that can be approximated by guessing the random weights. Compare your results with the following numbers from the table you saw in the lecture: 4 inputs: 6%; 3 inputs: 50%; 2 inputs: 88%, 1 input: 100%. The comparison should be done by visualizing both the estimated and the above lecture data in a single plot.   
