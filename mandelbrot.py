"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
def mandelbrot_point(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

