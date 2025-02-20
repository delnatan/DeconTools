"""
A collection of functions to reduce the image data into 1D profiles

"""

import numpy as np


def line_scan(image, start, end):
    """
    Performs an intensity line scan between two points in an image using
    bilinear interpolation.
    Number of samples is determined by the Euclidean distance between points.

    Parameters:
        image (ndarray): Input 2D image
        start (tuple): Starting point (y, x)
        end (tuple): End point (y, x)

    Returns:
        ndarray: Array of intensity values along the line
    """
    # Calculate the distance and use it as number of samples
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    num_samples = int(np.ceil(np.sqrt(dy * dy + dx * dx)))

    # Create sample points along the line
    t = np.linspace(0, 1, num_samples)
    y = start[0] + dy * t
    x = start[1] + dx * t

    # Get the four neighboring pixels for each point
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    # Ensure we don't go out of bounds
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)
    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)

    # Get the weights for interpolation
    wy = y - y0
    wx = x - x0

    # Get the four corner values
    Q00 = image[y0, x0]
    Q01 = image[y0, x1]
    Q10 = image[y1, x0]
    Q11 = image[y1, x1]

    # Perform bilinear interpolation
    intensities = (
        Q00 * (1 - wy) * (1 - wx)
        + Q01 * (1 - wy) * wx
        + Q10 * wy * (1 - wx)
        + Q11 * wy * wx
    )

    return intensities


def radial_power_spectrum_2d(image, spacing=1, half=True):
    """compute radial average of the power spectrum of input image

    Note:
    to properly display the frequency axis, you can use fraction labels
    to make the graph more intuitive

    def create_fraction_labels(frequencies):
        latex_frac ='$\\frac{{1}}{{{x:.1f}}}$'
        return [latex_frac.format(x=1/f) if f != 0 else '0' for f in frequencies]

    xticks = np.linspace(0, max_radial_frequencies, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(create_fraction_labels(xticks))
    ax.set_xlabel("Spatial frequency (1/length unit)")

    Returns:
    dictionary with 'power_spectrum', 'ky', 'kx', 'k', 'I(k)' and 'max_k'

    """
    Ny, Nx = image.shape

    # calculate frequency indices
    freq_idx_y = (np.fft.fftfreq(Ny) * Ny).astype(int)

    if half:
        ps = np.fft.rfft2(image)
        ps = np.abs(ps) ** 2
        freq_idx_x = (np.fft.rfftfreq(Nx) * Nx).astype(int)
    else:
        ps = np.fft.fft2(image)
        ps = np.abs(ps) ** 2
        freq_idx_x = (np.fft.fftfreq(Nx) * Nx).astype(int)

    nx, ny = ps.shape

    υ, ξ = np.meshgrid(freq_idx_y, freq_idx_x, indexing="ij")

    radial_indices = np.sqrt(υ**2 + ξ**2).astype(int)

    # compute the number of elements for each index
    counts = np.bincount(radial_indices.ravel())
    sums = np.bincount(radial_indices.ravel(), weights=ps.ravel())
    radial_profile = sums / counts

    Nprof = len(radial_profile)
    physical_frequencies = np.arange(Nprof) / (spacing * Nprof)

    # get the limiting dimension
    N = min(Ny, Nx)
    nyquist_index = int(N / 2) + 1 if N % 2 == 0 else int((N - 1) / 2)
    max_radial_freq = nyquist_index / (spacing * N)

    # for display flip the power spectrum
    if half:
        ret_ps = np.fft.fftshift(ps, axes=0)
        freq_x = np.fft.rfftfreq(Nx, d=spacing)
    else:
        ret_ps = np.fft.fftshift(ps)
        freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, d=spacing))

    freq_y = np.fft.fftshift(np.fft.fftfreq(Ny, d=spacing))

    return {
        "power_spectrum": ret_ps,
        "ky": freq_y,  # frequency along y-axis
        "kx": freq_x,  # frequency along x-axis
        "k": physical_frequencies,  # radial frequency
        "I(k)": radial_profile,  # radial average
        "max_k": max_radial_freq,  # max radial freq (from smallest dimension)
    }


def cylindrical_average_otf(psf: np.ndarray):
    """does cylindrical averaging about the k_z axis"""
    Nz, Ny, Nx = psf.shape

    # compute the intensity otf
    otfi = np.abs(np.fft.rfftn(psf)) ** 2

    is_even = lambda n: n % 2 == 0

    # build Fourier index for y and x axes which corresond to rfft convention
    # careful of upper-bound exclusive indexing in Python!
    if is_even(Ny):
        nyquist = int(Ny / 2)
        pos_yfreq_idx = np.arange(0, nyquist)
        neg_yfreq_idx = np.arange(-nyquist, 0)
    else:
        nyquist = int((Ny - 1) / 2)
        pos_yfreq_idx = np.arange(0, nyquist + 1)
        neg_yfreq_idx = np.arange(-nyquist, 0)

    yfreq_idx = np.concatenate((pos_yfreq_idx, neg_yfreq_idx))

    xfreq_idx = np.arange(0, Nx // 2 + 1)

    k_y, k_x = np.meshgrid(yfreq_idx, xfreq_idx, indexing="ij")

    # compute radial distance
    k_r = np.sqrt(k_x**2 + k_y**2)

    # use the smallest maximum index between yx axis
    max_indices = [abs(k_y).max(), abs(k_x).max()]

    # max_axis = 0 is 'y'; max_axis = 1 is 'x'
    max_axis, max_bin = min(enumerate(max_indices), key=lambda x: x[1])

    # choose coordinate for the radial axis (x is always the shortest)
    radial_coord = xfreq_idx.astype(float) / Nx
    axial_coord = np.fft.fftshift(np.fft.fftfreq(Nz))

    # bin by integer part of radial distance, clipped to max_bin
    # so that we don't go beyond the (smallest) nyquist frequency
    bins = np.minimum(np.floor(k_r).astype(int), max_bin)
    n_bins = max_bin + 1

    averaged = np.array(
        [
            np.bincount(
                bins.ravel(), weights=otfi[z].ravel(), minlength=n_bins
            )
            / np.bincount(bins.ravel(), minlength=n_bins)
            for z in range(Nz)
        ]
    )

    # center the axial origin in the middle
    averaged = np.fft.fftshift(averaged, axes=0)

    return radial_coord, axial_coord, averaged
