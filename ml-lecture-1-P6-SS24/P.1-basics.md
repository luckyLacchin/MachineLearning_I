# Programming Exercise P1. -- The angle between cats and dogs: bridging the dimensionality imagination gap (19P).

A typical issue with mathematical operations in high-dimensional spaces is the inability of the human mind to imagine them. Humans are well able to imagine two and three dimensions, at most maybe four when considering also time. However, in machine learning we often deal with high-dimensional spaces. So how can we obtain an intuition of mathematical operations in high-dimensional spaces?

In this exercise, you will train your intuition of vector operations in n-dimensional spaces. You will first apply operations in two dimensions and then transfer them to high-dimensional data. The exercises also includes a rehearsal of basic mathematical vector operations and basic Python programming. 

You will need to use basic Python operations, operations from the `math` package and operations using the `numpy` package. You should follow the de-facto convention and import numpy with the alias ``np``, i.e.,``import numpy as np``.
For most functions, there exists a condition whether to use numpy. If this condition exists, complete the function with both versions, with and without numpy. This allows you to compare both implementations and check the correctness of your code. 

If all your implementations are correct, you will be able to run the script `2d2nd.py`


## P1.1: Implement an n-dimensional random vector generator (2P)
implement a function that generates a vector with random values in the interval [0,1], i.e., complete the code for the function [generate_random_vector](helper_functions.py)


## P1.2: Write a function to compute the magnitude of a vector (2P). 
Implement a function that computes the magnitude of a vector, i.e., complete the code for the function [mag](helper_functions.py).

For the first version, use  just Python built-in operations and the `math` package. For the second version, use ``np.linalg.norm``.
Compare if both versions compute the same result for 10 pairs of n-dimensional vectors generated with the `generate_random_vector` function implemented earlier. 

## P1.3: Inner product of two vectors. (3P)
Implement a function to compute the inner product of two n-dimensional vectors, i.e., complete the code for the function [inner_product](helper_functions.py).

## P1.4: Vectors as images (3P)
Display 2D vectors as images with two pixels, as described in [2d2nd.py](2d2nd.py).

## Ex. 1.5: Angle between two vectors (7P).
Implement a function to compute the angle between two n-dimensional vectors, i.e, complete the code for the [compute_angle](helper_functions.py) function. Use the definition of the inner product for this purpose, and use the `inner_product` function you implemented earlier.
Furthermore, use the `compute_angle` function to compute the angle between high-dimensional random vectors, as described in [2d2nd.py](2d2nd.py)
In addition, display the random vectors as images. 

## Ex. 1.6: The angle between cats and dogs (2P). 
Compute the angle between the image of a cat and the image of a dog, as described in [2d2nd.py](2d2nd.py)


