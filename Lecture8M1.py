import numpy as np
import matplotlib.pyplot as plt
from mandelbrot import mandelbrot_serial

N, MAX_ITER, TAU = 512, 1000, 0.01
x_min = -0.7530
x_max = -0.7490
y_min = 0.0990
y_max = 0.1030
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)

C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)
diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
active = np.ones((N, N), dtype=bool)

for k in range(MAX_ITER):
    if not active.any(): break
    z32[active] = z32[active]**2 + C32[active]
    z64[active] = z64[active]**2 + C64[active]
    
    diff = (np.abs(z32.real.astype(np.float64) - z64.real)
            + np.abs(z32.imag.astype(np.float64) - z64.imag))
    
    newly = active & (diff > TAU)
    diverge[newly] = k
    active[newly] = False


mandelbrot_image = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=1000)



number_of_max_iter = np.sum(diverge == MAX_ITER)
print(f'Number of points that did not diverge within {MAX_ITER} iterations: {number_of_max_iter}')



earliest_divergence = np.unravel_index(np.argmin(diverge), diverge.shape)
print(f'Earliest divergence at index: {earliest_divergence}, C = {C64[earliest_divergence]}')


fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

im0 = ax[0].imshow(diverge, cmap='plasma', origin='lower',
                   extent=[x_min, x_max, y_min, y_max])
fig.colorbar(im0, ax=ax[0], label='First divergence iteration')
ax[0].set_title(f'Trajectory divergence (tau={TAU})')

im1 = ax[1].imshow(mandelbrot_image, cmap='inferno', origin='lower',
                   extent=[x_min, x_max, y_min, y_max])
fig.colorbar(im1, ax=ax[1], label='Escape iteration')
ax[1].set_title('Mandelbrot escape map')

for a in ax:
    a.set_xlabel('Re(c)')
    a.set_ylabel('Im(c)')

plt.savefig('mp3_m1_output.png', dpi=150)
plt.show()