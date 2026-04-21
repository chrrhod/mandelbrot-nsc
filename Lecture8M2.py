import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N, MAX_ITER = 512, 1000
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace( 0.0990, 0.1030, N)
C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)

eps32 = float(np.finfo(np.float32).eps)
delta = np.maximum(eps32 * np.abs(C), 1e-10)

def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    for k in range(max_iter):
        z[~esc] = z[~esc]**2 + C[~esc]
        newly = ~esc & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt

n_base = escape_count(C, MAX_ITER).astype(float)
n_perturb = escape_count(C + delta, MAX_ITER).astype(float)
dn = np.abs(n_base - n_perturb)

kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
cmap_k = plt.cm.hot.copy()
cmap_k.set_bad('0.25')

vmax = np.nanpercentile(kappa, 99)


#print where kappa is largest
max_idx = np.unravel_index(np.nanargmax(kappa), kappa.shape)
print(f'Max kappa at index {max_idx}, C = {C[max_idx]}, kappa = {kappa[max_idx]:.2e}')


interior = (n_base == MAX_ITER)
print("Interior pixel count:", interior.sum())
if interior.any():
    vals = kappa[interior]
    print("kappa interior: min =", np.nanmin(vals),
          "max =", np.nanmax(vals),
          "mean =", np.nanmean(vals))
else:
    print("No interior pixels in this window.")


plt.imshow(kappa, cmap=cmap_k, origin='lower',
           extent=[-0.7530, -0.7490, 0.0990, 0.1030],
           norm=LogNorm(vmin=1, vmax=vmax))
plt.colorbar(label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
plt.title(r'Condition number approx $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\,n(c))$')
plt.savefig('mp3_m2_output.png', dpi=150)
plt.show()