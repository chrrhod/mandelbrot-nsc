"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numba import njit, prange

def benchmark(func, *args, n_runs=15):
    """Time func, return median of n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    
    median_t = statistics.median(times)
    print(f"Method: {func.__name__}, Median: {median_t:.4f}s (min={min(times):.4f}, max={max(times):.4f})")
    
    return median_t, result
    

#@profile
def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter =100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype = int)
    for i in range(height):
        for j in range(width):
            c = x [j] + 1j * y[i]
            z = 0
            for n in range (max_iter):
                z = z *z + c
                if abs (z) > 2:
                    result[i, j] = n
                    break
            else:
                result[i, j ] = max_iter
    return result

#@profile
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter =100):
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
def mandelbrot_pixel(c_real, c_imag, max_iter=100):
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)


def visualize(result):
    plt.imshow(result, extent=(-2, 1, -1.5, 1.5), cmap='inferno')
    plt.colorbar()
    plt.title('Mandelbrot Set')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()



if __name__ == "__main__":
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    run_naive = False
    run_numpy = False
    run_serial_numba = True
    
    

    times = {}
    results = {}
    dtypes = [np.float32, np.float64]
    if run_naive:
        naive_t, naive_M = benchmark(mandelbrot_naive, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Naive"] = naive_t
        results["Naive"] = naive_M

    if run_numpy:
        numpy_t, numpy_M = benchmark(mandelbrot_numpy, xmin, xmax, ymin, ymax, width, height, max_iter)
        times["Numpy"] = numpy_t
        results["Numpy"] = numpy_M
    
    if run_serial_numba:
        serial_result = mandelbrot_serial(width, xmin, xmax, ymin, ymax, max_iter)
        visualize(serial_result)
        numba_t, numba_M = benchmark(mandelbrot_serial, width, xmin, xmax, ymin, ymax, max_iter)
        times["Numba"] = numba_t
        results["Numba"] = numba_M
    



    print("\nPerformance gains:")
    for method, t in times.items():
        if method == "Naive":
            continue
        gain = times["Naive"] / t
        print(f"{method}: {gain:.2f}x faster than Naive")
    



    #check if the results are the same
    if run_naive and run_numpy:
        if np.allclose (results["Naive"], results["Numpy"]):
            print ("Results match!")
        else:
            print ("Results differ!")
    
    # Check where they differ :
    diff = np.abs (results["Naive"] - results["Numpy"])
    print (f"Max difference : { diff.max ()}")
    print (f"Different pixels : {(diff > 0). sum ()}")


