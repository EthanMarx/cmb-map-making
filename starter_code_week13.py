import time

import numpy as np
import matplotlib.pyplot as plt

# These imports speed up FFTs, which makes a big difference to the map-maker runtime
# compared to just using the numpy or scipy fft.
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import multiprocessing
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()


class NoisePointingModel:
    """Represents the pointing operator and noise matrix.

    Parameters
    ----------
    x : 1D array of data length
        x pixel index (will be rounded to an integer). Values must be between 0 and nx - 1.
    y : 1D array of data length
        y pixel index (will be rounded to an integer). Values must be between 0 and ny - 1.
    nx : int
        Map size in x direction
    ny : int
        Map size in y direction
    noise_spec: 1D array
        Noise power spectrum. Length should be the same as `fft.rfft(data)`. Entries
        should be an estimate of < abs(rfft(data))**2 > / len(data).

    """

    def __init__(self, x, y, nx, ny, noise_spec):
        self._x = np.round(x).astype(int)
        self._y = np.round(y).astype(int)
        self._nx = nx
        self._ny = ny
        self._flat_inds = self._y + ny * self._x
        self._noise_spec = noise_spec.copy()
        # Replace the 0-frequency (mean mode) with twice the fundamental (frequency 1).
        self._noise_spec[0] = noise_spec[1] * 2

    def apply_noise_weights(self, data):
        """Noise weight a time-order-data array.

        Performs the operation $N^{-1} d$.

        """

        # Note that I don't need unitary normalizations for the FFTs since the normalization
        # factors cancel between the forward and inverse FFT. However, the noise power
        # spectrum must be normalized to be that of the unitary case.
        fdata = fft.rfft(data)
        fdata /= self._noise_spec
        #fdata[0] = 0
        return fft.irfft(fdata)

    def grid_data(self, data, out):
        """Accumulate time-order data into a map.

        Performs the operation $P^{T} d$.

        For performance reasons, output must be preallocated.
        It should be an array with shape (nx, ny).

        80% of the the runtime of the function `noise_ing_to_map_domain`
        is calling this function but I can't think of a simple way to
        speed it up.

        """

        #out = np.zeros((self._nx, self._ny), dtype=float)
        np.add.at(out, (self._x, self._y), data)
        return out

    def map_noise_inv(self):
        """Calculate the map noise inverse matrix.

        Performs the operation $P^T N^{-1} P$.

        Returns
        -------
        CN : 4D array with shape (nx, ny, nx, ny)

        """

        nx = self._nx
        ny = self._ny
        out = np.zeros((nx, ny, nx, ny), dtype=float)
        colP = pyfftw.empty_aligned(len(self._x), dtype=float)
        for ii in range(nx):
            print("x-index", ii)
            for jj in range(ny):
                #t0 = time.time()
                colP[:] = np.logical_and(self._x == ii, self._y == jj)
                #t1 = time.time() - t0
                colP[:] = self.apply_noise_weights(colP)
                #t2 = time.time() - t0
                self.grid_data(colP, out[ii, jj])
                #t3 = time.time() - t0
                #print(t1, t2, t3)
        return out


def naieve_PS_estimator(map, pix_size):
    """A simple angular power spectrum estimator.

    Implements the power spectrum estimator from the week 10 P-set, but in 2D. This
    is sub-optimal because it does not know about the correlations in the noise. It is
    also wrong on large scales since it assume a map with periodic boundary conditions.

    Parameters
    ----------
    map : 2D array
    pix_size : float
        Pixel size in radians.

    Returns
    -------
    Cl : 1d array len n_l
        Angular power spectrum estimate
    n_modes : 1d array len n_l
        Number of modes in each bin
    l_bin_edges : 1d array len n_l + 1
        Edges of the multipole (l) bins.

    """

    nx, ny = map.shape

    l_max = np.sqrt(2) * 2 * np.pi / (2 * pix_size)
    l_min = 2 * np.pi / (max(nx, ny) * pix_size)
    #l_bin_edges = np.logspace(np.log10(l_min), np.log10(l_max), 50, endpoint=True)
    l_bin_edges = np.linspace(l_min, l_max, 50, endpoint=True)

    al = fft.fft2(map) * pix_size**2
    Cl_2D = abs(al)**2
    lx = fft.fftfreq(map.shape[0], pix_size) * 2 * np.pi
    ly = fft.fftfreq(map.shape[1], pix_size) * 2 * np.pi
    l = np.sqrt(lx[:, None]**2 + ly**2)
    Cl = np.zeros(len(l_bin_edges) - 1)
    n_modes = np.zeros(len(l_bin_edges) - 1)
    for ii in range(len(l_bin_edges) - 1):
        ledge_l = l_bin_edges[ii]
        ledge_h = l_bin_edges[ii + 1]
        m = np.logical_and(l < ledge_h, l >= ledge_l)
        Cl[ii] += np.sum(Cl_2D[m])
        n_modes[ii] += np.sum(m)
    return Cl / n_modes / map.size / pix_size**2, n_modes, l_bin_edges


def colab_fast_linear_algebra():
    """How to accelerate linear algebra on Google colab using a GPU and tensorflow.

    For this to work you need to enable the GPU in your notebook. In the notebook menus:
    Edit -> Notebook settings and choose GPU.

    """

    import tensorflow as tf

    print(tf.config.experimental.list_physical_devices('GPU'))
    tf.debugging.set_log_device_placement(True)

    N = 64 * 256
    A = np.eye(N)
    B = np.eye(N)

    with tf.device('/device:GPU:0'):
        t0 = time.time()
        tf.linalg.matmul(A, B)
        print(time.time() - t0)
        # Roughly 1s.

    with tf.device('/device:GPU:0'):
        t0 = time.time()
        tf.linalg.inv(A, B)
        print(time.time() - t0)
        # About 1 minute.


def fast_camb_settings():
    """These settings seem to speed up camb to about 1s per run, and are accurate enough for our
    purposes.

    """

    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.7, ombh2=0.022, omch2=0.122, omk=0, tau=0.057);
    pars.InitPower.set_params(As=np.exp(3.0) / 1e10, ns=0.965, r=0)
    pars.WantTensors = False
    pars.WantTransfer = False
    pars.DoLensing = False
    pars.DoLateRadTruncation = True
    pars.set_for_lmax(1500, lens_potential_accuracy=0);
    return pars
