"""
Mandelbrot Set Generator
Author : Christian Rhod
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time, statistics
from numba import njit, prange
from multiprocessing import Pool

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


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter=100):
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
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


def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max,
    max_iter=100, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None: # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny) # warm-up: load JIT cache in workers
        parts = p.map(_worker, chunks)
    return np.vstack(parts)

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
    n_workers = 8

    run_naive = True
    run_numpy = False
    run_serial_numba = True
    run_parallel_numba = True
    
    

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
    
    mandelbrot_chunk(0, width, width, xmin, xmax, ymin, ymax, max_iter)
    if run_serial_numba:
        local_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            serial_M = mandelbrot_serial(width, xmin, xmax, ymin, ymax, max_iter=100)
            local_times.append(time.perf_counter() - t0)
        t_serial = statistics.median(local_times)
        print(f"Serial: {t_serial:.3f}s")
        times["Numba_serial"] = t_serial
        results["Numba_serial"] = serial_M
    
    if run_parallel_numba:
        print(f"\nParallel with {n_workers} workers:")
        tiny = [(0, 8, 8, xmin, xmax, ymin, ymax, max_iter)]
        for mult in [1, 2, 4, 8, 16]:
            n_chunks = mult * n_workers
            with Pool(processes=n_workers) as pool:
                pool.map(_worker, tiny) # warm-up: load JIT cache in workers
                local_times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    parallel_M = mandelbrot_parallel(width, xmin, xmax, ymin, ymax, max_iter, n_workers, n_chunks=n_chunks, pool=pool)
                local_times.append(time.perf_counter() - t0)
            t_par = statistics.median(local_times)
            lif = n_workers * t_par / t_serial - 1
            print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
            times[f"ParallelW{n_workers}C{n_chunks} Numba"] = t_par
            results[f"ParallelW{n_workers}C{n_chunks} Numba"] = parallel_M
    



    print("\nPerformance gains:")
    for method, t in times.items():
        if method == "Naive":
            continue
        if "Naive" not in times:
            print(f"{method}: {times[method]:.3f} No Naive baseline to compare against")
            continue
        else:
            gain = times["Naive"] / t
            print(f"{method}: {gain:.2f}x faster than Naive")
    



    """ #check if the results are the same
    if run_naive and run_numpy:
        if np.allclose (results["Naive"], results["Numpy"]):
            print ("Results match!")
        else:
            print ("Results differ!")
    
    # Check where they differ :
    diff = np.abs (results["Naive"] - results["Numpy"])
    print (f"Max difference : { diff.max ()}")
    print (f"Different pixels : {(diff > 0). sum ()}") """


