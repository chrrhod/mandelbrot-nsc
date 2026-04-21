import numpy as np
import pytest
from multiprocessing import Pool

from mandelbrot import (
    _worker,
    mandelbrot_chunk,
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_parallel,
    mandelbrot_pixel,
    mandelbrot_serial,
)


@pytest.mark.parametrize(
    "c_real,c_imag,max_iter,expected",
    [
        (0.0, 0.0, 50, 50),
        (3.0, 0.0, 50, 1),
        (1.0, 0.0, 50, 3),
    ],
)
def test_mandelbrot_pixel_python_analytic(c_real, c_imag, max_iter, expected):
    assert mandelbrot_pixel.py_func(c_real, c_imag, max_iter) == expected


@pytest.mark.parametrize(
    "c_real,c_imag,max_iter",
    [
        (0.0, 0.0, 50),
        (3.0, 0.0, 50),
        (1.0, 0.0, 50),
        (-0.75, 0.1, 80),
    ],
)
def test_mandelbrot_pixel_numba_matches_python(c_real, c_imag, max_iter):
    expected = mandelbrot_pixel.py_func(c_real, c_imag, max_iter)
    got = mandelbrot_pixel(c_real, c_imag, max_iter)
    assert got == expected


def test_numpy_matches_naive_small_grid():
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width = height = 32
    max_iter = 50

    naive = mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter)
    numpy_impl = mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter)

    assert np.array_equal(numpy_impl, naive)


def test_worker_matches_chunk_output():
    args = (4, 12, 16, -2.0, 1.0, -1.5, 1.5, 40)
    expected = mandelbrot_chunk(*args)
    got = _worker(args)
    assert np.array_equal(got, expected)


def test_parallel_matches_serial_small_grid():
    N = 32
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 50

    expected = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    with Pool(processes=2) as pool:
        got = mandelbrot_parallel(
            N,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iter=max_iter,
            n_workers=2,
            n_chunks=4,
            pool=pool,
        )

    assert np.array_equal(got, expected)


def test_dask_submit_gather_matches_expected_chunk():
    pytest.importorskip("dask")
    pytest.importorskip("dask.distributed")
    from dask.distributed import Client, LocalCluster
    from mandelbrot_dask import mandelbrot_chunk_opt

    args = (0, 8, 8, -2.0, 1.0, -1.5, 1.5, 40)
    expected = mandelbrot_chunk(*args)

    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            future = client.submit(mandelbrot_chunk_opt, *args)
            got = client.gather(future)

    # mandelbrot_chunk_opt uses 0 for non-escaped pixels; mandelbrot_chunk uses max_iter.
    got = np.where(got == 0, args[-1], got)
    assert np.array_equal(got, expected)