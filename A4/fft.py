import numpy as np
import matplotlib.pyplot as plt

## a)

# Initialize the random seed and generate particle positions
np.random.seed(121)

n_mesh = 16
n_part = 1024
positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))

# Grid and density initialization
grid = np.arange(n_mesh) + 0.5
densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
cellvol = 1.0

# Cloud-In-Cell method for density assignment
for p in range(n_part):
    cellind = np.zeros(shape=(3, 2), dtype=int)
    dist = np.zeros(shape=(3, 2))

    for i in range(3):
        cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
                              (abs(positions[i, p] - grid - 16) < 1) | 
                              (abs(positions[i, p] - grid + 16) < 1))[0]
        dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])

    for (x, dx) in zip(cellind[0], dist[0]):    
        for (y, dy) in zip(cellind[1], dist[1]):
            for (z, dz) in zip(cellind[2], dist[2]):
                if dx > 15: dx = abs(dx - 16)
                if dy > 15: dy = abs(dy - 16)
                if dz > 15: dz = abs(dz - 16)

                densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol

# Convert densities to density contrast delta
mean_density = n_part / n_mesh**3
delta = (densities - mean_density) / mean_density

# Plot the 2D slices of delta
slices = [4.5, 9.5, 11.5, 14.5]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, z in enumerate(slices):
    zi = int(z)
    im = axes[i].imshow(delta[:, :, zi], origin='lower', cmap='viridis')
    axes[i].set_title(f'Density contrast at z={z}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("./plots/fig2a.png")
plt.close()

## b)
                
# FFT using Cooley-Tukey algorithm (recursively)
def fft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Inverse FFT using Cooley-Tukey algorithm (recursively)
def ifft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = ifft_recursive(x[0::2])
    odd = ifft_recursive(x[1::2])
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

def fft3d(data):
    n = data.shape[0]
    fft_data = np.zeros_like(data, dtype=complex)

    # Apply FFT to each dimension
    for i in range(n):
        for j in range(n):
            fft_data[i, j, :] = fft_recursive(data[i, j, :])
        for k in range(n):
            fft_data[i, :, k] = fft_recursive(fft_data[i, :, k])
    for j in range(n):
        for k in range(n):
            fft_data[:, j, k] = fft_recursive(fft_data[:, j, k])

    return fft_data

def ifft3d(data):
    n = data.shape[0]
    ifft_data = np.zeros_like(data, dtype=complex)

    # Apply inverse FFT to each dimension
    for i in range(n):
        for j in range(n):
            ifft_data[i, j, :] = ifft_recursive(data[i, j, :])
        for k in range(n):
            ifft_data[i, :, k] = ifft_recursive(ifft_data[i, :, k])
    for j in range(n):
        for k in range(n):
            ifft_data[:, j, k] = ifft_recursive(ifft_data[:, j, k])

    return ifft_data

def fftfreq(n):
    """Returns the Discrete Fourier Transform sample frequencies."""
    val = 1.0 / n
    results = np.zeros(n)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val

def fftshift(x):
    """
    Shift the zero-frequency component to the center of the spectrum.
    This is for visualization purposes, since it is easier to interpret low and high frequencies
    centered around zero.
    """
    n = len(x)
    p2 = x[n//2:]
    p1 = x[:n//2]
    return np.concatenate((p2, p1))

# Create the k-grid
def create_k_grid(n):
    k = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                kx = (i - n if i > n // 2 else i) * 2 * np.pi / n
                ky = (j - n if j > n // 2 else j) * 2 * np.pi / n
                kz = (l - n if l > n // 2 else l) * 2 * np.pi / n
                k[i, j, l] = kx**2 + ky**2 + kz**2
    k[0, 0, 0] = 1  # To avoid division by zero
    return k

# Perform FFT on delta
delta_k = fft3d(delta)

# Calculate potential in Fourier space
k = fftfreq(n_mesh) * 2 * np.pi
k_squared = create_k_grid(n_mesh)

phi_k = delta_k / k_squared

# Manually compute the real part of the inverse FFT
def ifft_real(data):
    n = data.shape[0]
    real_data = np.zeros_like(data, dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                sum_val = 0
                for kx in range(n):
                    for ky in range(n):
                        for kz in range(n):
                            angle = 2 * np.pi * ((kx * x / n) + (ky * y / n) + (kz * z / n))
                            complex_val = data[kx, ky, kz] * np.exp(1j * angle)
                            # Use symmetry property of FFT of real-valued signal
                            if kx <= n//2 and ky <= n//2 and kz <= n//2:
                                sum_val += complex_val
                            else:
                                sum_val += complex_val  # This is actually the complex conjugate
                real_data[x, y, z] = sum_val / (n**3)
    return real_data


# Perform inverse FFT to get phi in real space
phi = ifft_real(phi_k)

# Plot the 2D slices of phi
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, z in enumerate(slices):
    zi = int(z)
    im = axes[i].imshow(phi[:, :, zi], origin='lower', cmap='viridis')
    axes[i].set_title(f'Potential at z={z}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("./plots/fig2b.png")
plt.close()

# Plot the log of the absolute value of the Fourier-transformed potential
log_phi_k = np.log10(np.abs(phi_k))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, z in enumerate(slices):
    zi = int(z)
    im = axes[i].imshow(fftshift(log_phi_k[:, :, zi]), origin='lower', cmap='viridis')
    axes[i].set_title(f'log10(|$\~\Phi$|) at z={z}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("./plots/fig2b_pot.png")
plt.close()
