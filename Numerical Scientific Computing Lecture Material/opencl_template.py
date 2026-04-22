#!/usr/bin/env python3
"""
opencl_template.py — Starting point for writing a PyOpenCL kernel.

Replace the vector-add kernel below with your own kernel.
The six steps are the same for every OpenCL program.
"""

import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

VEC_SIZE = 50_000

# --- Step 1: create context and command queue ---
ctx   = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
print(f"Device: {ctx.devices[0].name}")

# --- Step 2: prepare host arrays ---
a_host      = np.random.rand(VEC_SIZE).astype(np.float32)
b_host      = np.random.rand(VEC_SIZE).astype(np.float32)
result_host = np.empty_like(a_host)

# --- Step 3: allocate device buffers and copy input data ---
mf       = cl.mem_flags
a_dev    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=a_host)
b_dev    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b_host)
res_dev  = cl.Buffer(ctx, mf.WRITE_ONLY, a_host.nbytes)

# --- Step 4: compile the kernel ---
# To load from a separate file instead: KERNEL_SRC = open("kernel.cl").read()
KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""
prog = cl.Program(ctx, KERNEL_SRC).build()


N, MAX_ITER = 1024, 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)


prog.mandelbrot(queue, (64, 64), None, image_dev,
                np.float32(X_MIN), np.float32(X_MAX),
                np.float32(Y_MIN), np.float32(Y_MAX),
                np.int32(64), np.int32(MAX_ITER))
queue.finish()


# --- Step 5: launch the kernel ---
t0 = time.perf_counter()
prog.mandelbrot(
    queue, (N, N), None,      # global size (N, N); let OpenCL pick local
    image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(N), np.int32(MAX_ITER),
)
queue.finish()
elapsed = time.perf_counter() - t0

# --- Step 6: copy result back to host ---
cl.enqueue_copy(queue, image, image_dev)
queue.finish()

print(f"GPU {N}x{N}: {elapsed*1e3:.1f} ms")
plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
plt.savefig("mandelbrot_gpu.png", dpi=150, bbox_inches='tight')

# Verify and report
#print(f"Elapsed:  {elapsed*1000:.3f} ms")
#print(f"Correct:  {np.allclose(result_host, a_host + b_host)}")
