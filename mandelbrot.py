"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time , statistics


def benchmark(func, *args, n_runs=3):
    """ Time func, return median of n_runs. """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    
    median_t = statistics.median(times)
    print(f"Method: {func.__name__}, Median: {median_t:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    
    return median_t, result
    

@profile # Add this decorator
def mandelbrot_naive( xmin , xmax , ymin , ymax , width , height , max_iter =100):
    x = np.linspace ( xmin , xmax , width )
    y = np.linspace ( ymin , ymax , height )
    result = np.zeros (( height , width ) , dtype = int )
    for i in range ( height ):
        for j in range ( width ):
            c = x [j] + 1j * y[ i]
            z = 0
            for n in range ( max_iter ):
                z = z *z + c
                if abs (z) > 2:
                    result [i , j] = n
                    break
            else:
                result [i , j ] = max_iter
    return result

@profile # Add this decorator
def mandelbrot_numpy ( xmin , xmax , ymin , ymax , width , height , max_iter =100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=complex)
    result = np.full(C.shape, max_iter, dtype=int)
    mask = np.ones(C.shape, dtype=bool)
    for n in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = (np.abs(Z) > 2) & mask
        result[escaped] = n
        mask[escaped] = False
        if not mask.any():
            break
    return result





if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    run_naive = True
    run_numpy = True

    times = {}
    results = {}

    if run_naive:
        naive_t, naive_M = benchmark(mandelbrot_naive, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Naive"] = naive_t
        results["Naive"] = naive_M

    if run_numpy:
        numpy_t, numpy_M = benchmark(mandelbrot_numpy, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Numpy"] = numpy_t
        results["Numpy"] = numpy_M

    print("\nPerformance gains:")
    for method, t in times.items():
        if method == "Naive":
            continue
        gain = times["Naive"] / t
        print(f"{method}: {gain:.2f}x faster than Naive")
    



    #check if the results are the same
    if run_naive and run_numpy:
        if np.allclose (results["Naive"], results["Numpy"]):
            print (" Results match !")
        else:
            print (" Results differ !")
    
    # Check where they differ :
    diff = np .abs ( results["Naive"] - results["Numpy"] )
    print (f" Max difference : { diff.max ()}")
    print (f" Different pixels : {( diff > 0). sum ()}")


