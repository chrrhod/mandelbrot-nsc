"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time , statistics
from numba import njit

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
    

#@profile
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

#@profile
def mandelbrot_numpy( xmin , xmax , ymin , ymax , width , height , max_iter =100):
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

@njit
def mandelbrot_naive_numba( xmin , xmax , ymin , ymax , width , height , max_iter =100):
    """ Fully JIT - compiled Mandelbrot --- structure identical to naive ."""
    x = np . linspace ( xmin , xmax , width )
    y = np . linspace ( ymin , ymax , height )
    result = np . zeros (( height , width ) , dtype = np . int32 )
    for i in range ( height ): # compiled loop
        for j in range ( width ): # compiled loop
            c = x [j] + 1j * y[ i]
            z = 0j # complex literal : type inference works !
            n = 0
            while n < max_iter and (z. real * z. real + z. imag * z. imag ) <= 4.0:
                z = z *z + c
                n += 1
            result [i , j ] = n
    return result

@njit
def mandelbrot_point_numba(c, max_iter=100):
    """ JIT - compiled Mandelbrot for a single point ."""
    z = 0j
    n = 0
    while n < max_iter and (z. real * z. real + z. imag * z. imag ) <= 4.0:
        z = z *z + c
        n += 1
    return n

def mandelbrot_hybrid_numba( xmin , xmax , ymin , ymax , width , height , max_iter =100):
    """ Hybrid approach : JIT - compile the point function only ."""
    x = np . linspace ( xmin , xmax , width )
    y = np . linspace ( ymin , ymax , height )
    result = np . zeros (( height , width ) , dtype = np . int32 )
    for i in range ( height ):
        for j in range ( width ):
            c = x [j] + 1j * y[ i]
            result [i , j ] = mandelbrot_point_numba(c, max_iter)
    return result



if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    run_naive = True
    run_numpy = True
    run_naive_numba = True
    run_hybrid_numba = True
    run_numpy_numba = True

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
    
    if run_naive_numba:
        _ = mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter)  # Warm-up
        naive_numba_t, naive_numba_M = benchmark(mandelbrot_naive_numba, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Naive Numba"] = naive_numba_t
        results["Naive Numba"] = naive_numba_M
    
    if run_hybrid_numba:
        _ = mandelbrot_hybrid_numba(xmin, xmax, ymin, ymax, width, height, max_iter)  # Warm-up
        hybrid_numba_t, hybrid_numba_M = benchmark(mandelbrot_hybrid_numba, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Hybrid Numba"] = hybrid_numba_t
        results["Hybrid Numba"] = hybrid_numba_M



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


