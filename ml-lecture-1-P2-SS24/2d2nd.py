import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
import math
from plot_functions import visualize_scalar_product_2D, get_angle
import random
from helper_functions import generate_random_vector, load_image, plot_inner_product, flatten_matrix, show_image, compute_angle, mag


if __name__ == "__main__":
    # P1.1.: Generate two random 2D vectors and print them to the console.
    # This only works if the function generate_random_vector in helper_functions.py is implemented correctly.
    x1_2d_vec = generate_random_vector(dim=2, use_numpy=True)
    print(x1_2d_vec)
    x2_2d_vec = generate_random_vector(dim=2, use_numpy=True)
    print(x2_2d_vec)

    # P1.2.: Compute the magnitudes of these vectors and print the values to the console.
    # This only works if the function mag in helper_functions.py is implemented correctly.
    mag1 = mag(x1_2d_vec)
    print(mag1)
    mag2 = mag(x2_2d_vec)
    print(mag2)

    # P1.3.: Compute and plot the inner product of these vectors using the plot_inner_product function. The function
    # only works if the inner_product function in helper_functions.py is implemented correctly.
    plot_inner_product(x1_2d_vec, x2_2d_vec)

    # P1.4: Train your intuition. Think of a 2D vector as an image with two pixels, and think of an image of
    # size <width> X <height> as a vector of that size. An image is a matrix <width> X <height>, so we transform
    # the 2D vector into a matrix with <width>=2 and <height>=1 and display it as an image.
    # To actually see it displayed, scale the vector values up to 255. Also, scale the image width up to 200 pixels
    # and display it with the show_image function.
    # <START Your code here>
    x1_2d_img = np.reshape(x1_2d_vec, (1, 2)) * 255
    x2_2d_img = np.reshape(x2_2d_vec, (1, 2)) * 255
    show_image(x1_2d_img, scale_to_width=200)
    show_image(x2_2d_img, scale_to_width=200)
    # <END Your code here>

    # P1.5.: Compute the angle between two vectors. Note that you can understand this also as the angle between the two images.
    # First compute the angle between the 2D-vectors generated in P1.3.
    ang = compute_angle(x1_2d_vec, x2_2d_vec)
    print(f"The angle is {ang}")

    # Use the compute_angle function to determine the angle between two random 20000 dimensional vectors.
    # Display the vectors as images with <width>=200 and <height>=100
    ang = 45  # TODO: Overwrite this with the correct angle
    # <START Your code here>
    x1_nd_vec = generate_random_vector(dim=20000, use_numpy=True)
    x2_nd_vec = generate_random_vector(dim=20000, use_numpy=True)
    ang = compute_angle(x1_nd_vec, x2_nd_vec)
    # <END Your code here>
    print(f"The angle is {ang}")

    # Display the 20000-dimensional vectors as images with <width>=200 and <height>=100. Use the function show_image for this.
    # <START Your code here>
    x1_nd_img = np.reshape(x1_nd_vec, (100, 200))
    x2_nd_img = np.reshape(x2_nd_vec, (100, 200))
    show_image(x1_nd_img * 255)
    show_image(x2_nd_img * 255)
    # <END Your code here>

    # P1.6.: Use the compute_angle function for images to determine the angle between `cat.jpg` and `dog.jpg`.
    # Note that an image is a <width> X <height> matrix, but the inner product is only defined for vectors.
    # Hence, you should "flatten" the image matrices first so they become vectors with a single dimension of size <width> * <height>.
    x1 = load_image(filename="cat.jpg", scale_to_size=(100, 100))
    x2 = load_image(filename="dog.jpg", scale_to_size=(100, 100))
    show_image(x1)
    show_image(x2)
    ang = 45  # TODO: Overwrite this with the correct angle
    # <START Your code here>
    x1 = flatten_matrix(x1)
    x2 = flatten_matrix(x2)
    ang = compute_angle(x1, x2)
    # <END Your code here>
    print(f"The angle between the cat and the dog is {ang} deg.")
