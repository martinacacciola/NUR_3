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
    # If the array is of length 1, return the input
    if N <= 1:
        return x
    # Call the DFT routine recursively on all even and odd elements
    # Store the output respectively on the first and second half of the array (corresponding to bit reversal)
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    # Combine results of the even and odd parts
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Inverse FFT using Cooley-Tukey algorithm (recursively)
def ifft_recursive(x):
    N = len(x)
    if N <= 1:
        return x
    even = ifft_recursive(x[0::2])
    odd = ifft_recursive(x[1::2])
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    # Combine results of the even and odd parts and divide by 2 to account for the normalization factor
    # This ensures that the energy of the signal is preserved during the transformation 
    # from the frequency domain back to the time domain
    return [(even[k] + T[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k]) / 2 for k in range(N // 2)]

def fft3d(data):
    n = data.shape[0]
    # Convert the input element to complex type
    fft_data = np.zeros_like(data, dtype=complex)

    # Apply FFT to each dimension
    # Series of 1D FFTs over rows, then over columns
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
    # Convert the input element to complex type
    ifft_data = np.zeros_like(data, dtype=complex)

    # Apply inverse FFT to each dimension
    # Series of 1D FFTs over rows, then over columns
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
    """
    Calculates the Discrete Fourier Transform sample frequencies of a signal
    Input:
    - n: Number of data points in the input signal

    Returns:
    - results: Array of sample frequencies.
    """

    # Step size between frequencies
    val = 1.0 / n
    # Array to store the frequencies
    results = np.zeros(n)
    # Number of positive frequencies (including zero frequency)
    N = (n - 1) // 2 + 1
    # Generate positive frequencies
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    # Generate negative frequencies
    p2 = np.arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    # Multiply frequencies by the step size to get the actual frequency values
    return results * val

def fftshift(x):
    """
    Shift the zero-frequency component to the center of the spectrum.
    """
    n = len(x)
    # Split the input array into two parts: right half (from index n//2 to end) and left half (from index 0 to n//2)
    p2 = x[n//2:]
    p1 = x[:n//2]
    # Concatenate the two parts to shift the zero-frequency component to the center
    return np.concatenate((p2, p1))

def create_k_grid(n):
    """
    Create a  3D grid of wave numbers (k) for Fourier space representation
    Input:
    - n: Size of the grid in each dimension.

    Returns:
    - k: 3D array representing the wave numbers.
    """

    # Initialize a 3D array to store wave numbers
    k = np.zeros((n, n, n))

    # Iterate over each point in the grid
    for i in range(n):
        for j in range(n):
            for l in range(n):
                # Calculate the wave number components for each dimension
                # Apply periodic boundary conditions if the index exceeds the Nyquist frequency
                # ensuring that the wave numbers wrap around appropriately within the grid
                kx = (i - n if i > n // 2 else i) * 2 * np.pi / n
                ky = (j - n if j > n // 2 else j) * 2 * np.pi / n
                kz = (l - n if l > n // 2 else l) * 2 * np.pi / n

                # Compute the magnitude of the wave number vector
                k[i, j, l] = kx**2 + ky**2 + kz**2

    # Set the value at the origin to avoid division by zero
    k[0, 0, 0] = 1  

    return k

# Perform FFT on delta
delta_k = fft3d(delta)

# Calculate potential in Fourier space
k = fftfreq(n_mesh) * 2 * np.pi
k_squared = create_k_grid(n_mesh)

phi_k = delta_k / k_squared

# Perform inverse FFT to get phi in real space
# Taking the real part is necessary in this case because of the mixing between real and imaginary components due to round-off error
# In this way, we get rid of the small imaginary part to allow the visualization of the potential
phi = np.real(ifft3d(phi_k))

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
