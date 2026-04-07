from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
import numpy as np
import time
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from mandelbrot import mandelbrot_serial, benchmark

@njit(cache=True)
def mandelbrot_chunk_opt(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    rows = row_end - row_start
    out  = np.zeros((rows, N), dtype=np.int32)
    dx   = (x_max - x_min) / N
    dy   = (y_max - y_min) / N

    # Store c values and z state for every pixel in the chunk
    zr   = np.zeros((rows, N), dtype=np.float64)
    zi   = np.zeros((rows, N), dtype=np.float64)
    cr   = np.empty((rows, N), dtype=np.float64)
    ci   = np.empty((rows, N), dtype=np.float64)
    done = np.zeros((rows, N), dtype=np.bool_)

    for r in range(rows):
        for col in range(N):
            cr[r, col] = x_min + col * dx
            ci[r, col] = y_min + (r + row_start) * dy

    for step in range(1, max_iter + 1):
        all_done = True
        for r in range(rows):
            for col in range(N):
                if not done[r, col]:
                    zr2 = zr[r, col] * zr[r, col]
                    zi2 = zi[r, col] * zi[r, col]
                    if zr2 + zi2 > 4.0:
                        done[r, col] = True
                        out[r, col]  = step - 1
                    else:
                        zi[r, col] = 2.0 * zr[r, col] * zi[r, col] + ci[r, col]
                        zr[r, col] = zr2 - zi2 + cr[r, col]
                        all_done = False   # chunk still has live pixels
        if all_done:
            break   # ← early exit: every pixel in chunk has diverged

    # Pixels that never diverged stay 0 (inside the set)
    return out


# ── Dask driver ───────────────────────────────────────────────────────────────
def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk_opt)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────
def median_time(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def sweep(client, N, x_min, x_max, y_min, y_max, max_iter, chunk_values):
    # single-chunk baseline (n_chunks=1 → serial reference)
    t1 = median_time(
        lambda: mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                                max_iter, n_chunks=1))

    print(f"\n{'─'*62}")
    print(f"  {'n_chunks':>8} │ {'time (s)':>8} │ {'vs 1x':>6} │ "
          f"{'speedup':>7} │ {'LIF':>7}")
    print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}")

    n_workers = sum(client.nthreads().values())
    results = []

    for nc in chunk_values:
        t = median_time(
            lambda nc=nc: mandelbrot_dask(
                N, x_min, x_max, y_min, y_max, max_iter, n_chunks=nc))
        speedup = t1 / t
        lif = speedup / n_workers - 1
        print(f"  {nc:>8} │ {t:>8.3f} │ {t/t1:>6.2f} │ {speedup:>7.2f}x │ {lif:>7.3f}")
        results.append(dict(n_chunks=nc, t=t, speedup=speedup, lif=lif))

    return results


def plot_sweep(results, out_path="dask_chunk_sweep.png"):
    xs = [r["n_chunks"] for r in results]
    ts = [r["t"]        for r in results]
    ls = [r["lif"]      for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dask LocalCluster – chunk sweep (early-exit kernel)",
                 fontsize=14, fontweight="bold")

    axes[0].plot(xs, ts, marker="o", color="steelblue")
    axes[0].set_title("Wall time vs n_chunks")
    axes[0].set_ylabel("Wall time (s)")

    axes[1].plot(xs, ls, marker="s", color="darkorange")
    axes[1].set_title("LIF vs n_chunks")
    axes[1].set_ylabel("LIF  (speedup/n_workers − 1)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_chunks (log₂ scale)")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N        = 8192
    MAX_ITER = 100
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    # Covers the three-way trade-off: underparallelism → sweet spot → Dask overhead
    CHUNK_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # ── One cluster for all measurements ─────────────────────────────────────
    client = Client("tcp://10.92.1.232:8786") #10.92.1.232:8786

    #cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    #client = Client(cluster)

    print(f"Dashboard: {client.dashboard_link}")

    #numberOfWorkers = len(client.nthreads())

    # ── Warm up Numba JIT on every worker ────────────────────────────────────
    client.run(lambda: mandelbrot_chunk_opt(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))
    print(f"Numba JIT warmed up on all {len(client.nthreads())} workers.\n")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results = sweep(client, N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER, CHUNK_VALUES)

    client.close()
    #cluster.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    best = min(results, key=lambda r: r["t"])
    print(f"\n  n_chunks optimal : {best['n_chunks']}")
    print(f"  t_min            : {best['t']:.3f} s")
    print(f"  LIF at optimum   : {best['lif']:.3f}")

    #Run the benchmark on the serial version to get the reference time
    
    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER) # warm-up
    serial_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, MAX_ITER)
        serial_times.append(time.perf_counter() - t0)
    serial_time = statistics.median(serial_times)

    print(f"\n  Serial reference time: {serial_time:.3f} s")




    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_sweep(results, out_path="dask_chunk_sweep.png")