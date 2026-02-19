"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time


def mandelbrot_point(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter


def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            mandelbrot_set[i, j] = mandelbrot_point(c, max_iter)
    return mandelbrot_set

def view_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax):
    """
    Made with AI
    """
    plt.imshow(mandelbrot_set, extent=(xmin, xmax, ymin, ymax), cmap='inferno')
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

    #save the image
    plt.imsave('mandelbrot.png', mandelbrot_set, cmap='inferno')

def time_function(func, *args):
    start_time = time()
    result = func(*args)
    end_time = time()
    print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
    return result

if __name__ == "__main__":
    # Parameters for the Mandelbrot set
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    #width, height = 800, 600
    width, height = 1024, 1024
    max_iter = 100


    mandelbrot_set = time_function(compute_mandelbrot,xmin, xmax, ymin, ymax, width, height, max_iter)



    # Visualize the Mandelbrot set
    view_mandelbrot(mandelbrot_set, xmin, xmax, ymin, ymax)