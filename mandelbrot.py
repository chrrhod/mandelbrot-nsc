"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np

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
