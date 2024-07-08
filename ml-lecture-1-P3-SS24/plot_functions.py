import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import math
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_angle(line_xy):
    """
    Computes the angle between a straight line and the x-axis.
    :param line_xy: The line start and end point
    :return: The angle in degrees.
    """
    offset_vector = [line_xy[1][0] - line_xy[0][0], line_xy[1][1] - line_xy[0][1]]
    slope = offset_vector[1] / offset_vector[0]
    radAng = np.arctan(slope)
    if offset_vector[0] < 0:
        radAng += math.pi
    degAng = np.rad2deg(radAng)
    if math.isnan(degAng):
        print("Warning, invalid angle of line! Setting the angle to 0.")
        degAng = 0
    return degAng

# Helper function to generate the Arc object for matplotlib between two lines.
def get_angle_plot(line1, line2, offset=0.1, color=None, origin=(0, 0), len_x_axis = 1, len_y_axis = 1):

    # Get angles between lines and x-axis
    angle1 = get_angle(line1.get_xydata())
    angle2 = get_angle(line2.get_xydata())

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color() # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, theta1, theta2, color=color, label = str(angle)+u"\u00b0")


# Helper function to generate the angle text for an arc between two lines.
def get_angle_text(angle_plot):
    angle = angle_plot.get_label()[:-1] # Excluding the degree symbol
    # angle = "%0.2f"%float(angle) # Display angle upto 2 decimal places
    angle = "%0.2f" % float(angle) + u"\u00b0"  # Display angle upto 2 decimal places

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()

    # Get the midpoint of the arc extremes
    x_width = (vertices[0][0] + vertices[-1][0]) / 2.0
    y_width = (vertices[0][1] + vertices[-1][1]) / 2.0

    separation_radius = max(x_width/2.0, y_width/2.0)

    return [x_width + separation_radius, y_width + separation_radius, angle]


# Helper function for initializing the 2D linear separation plot
def visualize_scalar_product_2D(x1, x2, inner_product, vec_rotate_function):
    """

    :param x1: The first 2d vector of the inner product
    :param x2: The second 2d vector of the inner product
    :param inner_product: The inner product value of the two vectors
    :param vec_rotate_function: A function for rotating a vector counter-clockwise
    :return:
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    print(x1.shape)
    assert x1.shape == (2,) and x2.shape == (2,), "Error, cannot visualize scalar product because at least one vector is not 2-dimensional."
    # Create canvas
    fig, ax = plt.subplots(figsize=(10, 10))
    x_limits = (min([x1[0], x2[0], 0]) - 1, max([x1[0], x2[0], 0]) + 1)
    y_limits = (min([x1[1], x2[1], 0]) - 1, max([x1[1], x2[1], 0]) + 1)
    coord_x = Line2D([x_limits[0], x_limits[1]], [0, 0], linestyle='--', color='black')
    coord_y = Line2D([0, 0], [y_limits[0], y_limits[1]], linestyle='--', color='black')
    ax.add_line(coord_x)
    ax.add_line(coord_y)
    vec_1 = Line2D([0, x1[0]], [0, x1[1]], linestyle='-', color='red', linewidth=2)
    vec_2 = Line2D([0, x2[0]], [0, x2[1]], linestyle='-', color='blue', linewidth=2)
    ax.add_line(vec_1)
    ax.text(x1[0], x1[1], "x1")
    ax.add_line(vec_2)
    ax.text(x2[0], x2[1], "x2")
    angle_plot = get_angle_plot(vec_1, vec_2, len_x_axis=x_limits[1] - x_limits[0], len_y_axis=y_limits[1]-y_limits[0])
    angle_text = get_angle_text(angle_plot)
    ax.add_patch(angle_plot)  # To display the angle arc
    ax.text(*angle_text)

    angle1 = get_angle(vec_1.get_xydata())
    angle2 = get_angle(vec_2.get_xydata())

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)
    mid_angle = theta1 + (theta2 - theta1) / 2

    dot_prod_x = [inner_product, 0]
    dot_prod_line = vec_rotate_function(dot_prod_x, mid_angle)
    mid_line = Line2D([0, dot_prod_line[0]], [0, dot_prod_line[1]], linestyle='-', color='black', linewidth=2)
    ax.add_line(mid_line)
    x_limits_canvas = (min(x_limits[0], dot_prod_line[0]) - 1, max(x_limits[1], dot_prod_line[0]) + 1)
    y_limits_canvas = (min(y_limits[0], dot_prod_line[1]) - 1, max(y_limits[1], dot_prod_line[1]) + 1)
    ax.set_xlim(*x_limits_canvas)
    ax.set_ylim(*y_limits_canvas)

    plt.text(x_limits[1]/4, -0.5, "The black line is not a vector, it just illustrates ")
    plt.text(x_limits[1] / 4, -0.7, "the inner product (the line's length) as the amount ")
    plt.text(x_limits[1] / 4, -0.9, "to which the vectors x1 and x2 point to the same direction.")

    plt.show()
    pass


# Helper function to get coordinates of normal hyperplane segment of a vector
def hyperplane_from_normal_2D(w, l):
    """
    Generates a 2D hyperplane (an orthogonal straight line segment) from a 2D normal vector.
    :param w: The 2D normal vector
    :param l: The length of the line segment
    :return: A tuple where the first element is the starting point and the second element is the ending point of the line.
    """
    ortho_normal_l = np.array([-w[1], w[0]]) / (np.linalg.norm(w)) * l
    ortho_normal_r = np.array([w[1], -w[0]]) / (np.linalg.norm(w)) * l
    return [ortho_normal_l[0], ortho_normal_r[0]], [ortho_normal_l[1], ortho_normal_r[1]]


# Helper function for initializing the 2D linear separation plot
def init_2D_linear_separation_plot(C, NotC, w, R):
    """
    Intializes an animated plot with two classes of points in 2D.
    :param C: The set of points of one class
    :param NotC: the set of points in the other class
    :param w: the normal vector of the separating line segment (2D hyperplane)
    :param R: the radius of the set of points, i.e., the distance from the origin to the farthest point.
    :return: a tuple containing the figure, axis, orthonormal and w-vector matplotlib objects.
    """
    # Turn on interactive plotting
    plt.ion()

    # Create canvas
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add data points
    Ct = np.transpose(C)
    plt.scatter(Ct[0], Ct[1], c='blue')
    NotCt = np.transpose(NotC)
    plt.scatter(NotCt[0], NotCt[1], c='red')

    # Draw radius and circle
    rad_circ = plt.Circle((0, 0), R, fill=False)
    ax.add_patch(rad_circ)

    # Draw vector w*
    w_arrow, = plt.plot([0, 0], [w[0], w[1]], 'k', label="w")

    # Draw line orthogonal to w* (separating hyperplane)
    ortho_normal = hyperplane_from_normal_2D(w, R)
    w_ortho, = plt.plot(ortho_normal[0], ortho_normal[1], 'k--')
    plt.legend()
    plt.show()
    figure_data = fig, ax, w_ortho, w_arrow
    update_2D_linear_separation_plot(figure_data, w, R)
    return figure_data


# Helper function for updating the 2D linear separation plot
def update_2D_linear_separation_plot(figure_data, w, R):
    """
    Updates the figure with a new normal vector of the separating line segment.
    :param figure_data: a tuple containing the figure, axis, orthogonal and w-vector matplotlib objects
    :param w: the orthogonal vector to the separating line segment
    """
    fig, ax, w_ortho, w_arrow = figure_data
    w_arrow.set_xdata([0, w[0]])
    w_arrow.set_ydata([0, w[1]])

    w_ortho_points = hyperplane_from_normal_2D(w, R)
    w_ortho.set_xdata(w_ortho_points[0])
    w_ortho.set_ydata(w_ortho_points[1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

def plot_3d_surface(data, filename='', x_label='', y_label='', z_label='', rotate_z=(None,None)):
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    Y = sorted(data.keys())
    X = sorted(data[Y[0]].keys())
    Z = [[data[n_inputs][n_hidden] for n_hidden in X] for n_inputs in Y]
    X, Y = np.meshgrid(X, Y)

    Z = np.array(Z)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    if x_label != '': ax.set_xlabel(x_label)
    if y_label != '': ax.set_ylabel(y_label)
    if z_label != '': ax.set_zlabel(z_label)
    # Customize the z axis.
    ax.set_zlim(0.0, np.nanmax(Z))
    # ax.zaxis._set_scale('log')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(rotate_z[1], rotate_z[0])
    if filename == '':
        plt.show()
    else:
        plt.savefig(filename)


def plot_3d_surface_hidden_interval_approx(data, filename='', x_label='', y_label='', z_label='', rotate_z=(None,None)):
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    Y = sorted(data.keys())
    X = sorted(data[Y[0]].keys())
    Z = [[data[interval_len][n_hidden] for n_hidden in X] for interval_len in Y]
    X, Y = np.meshgrid(X, Y)

    Z = np.array(Z)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    if x_label != '': ax.set_xlabel(x_label)
    if y_label != '': ax.set_ylabel(y_label)
    if z_label != '': ax.set_zlabel(z_label)
    # Customize the z axis.
    ax.set_zlim(0.0, np.nanmax(Z))
    # ax.zaxis._set_scale('log')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(rotate_z[1], rotate_z[0])
    if filename == '':
        plt.show()
    else:
        plt.savefig(filename)