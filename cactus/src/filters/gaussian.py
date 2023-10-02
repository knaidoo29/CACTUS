import numpy as np

from ...ext import shift


def smooth3D(f, Rn, boxsize, ngrid):
    """Smoothing a 3D field with gaussian kernel in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    Rn : float
        Gaussian kernel smoothing radius.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.

    Returns
    -------
    fsmooth : 3darray
        Gaussian smoothed field.
    """
    # Create fourier mode grid
    ngrid = len(f)
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.fft3D(f, boxsize)
    # Smooth in Fourier space
    fk = fk*shift.cart.convolve_gaussian(kmag, Rn)
    # Backward FFT
    fsmooth = shift.cart.ifft3D(fk, boxsize)
    return fsmooth


def mpi_smooth3D(f, Rn, boxsize, ngrid, MPI):
    """Smoothing a 3D field with gaussian kernel in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    Rn : float
        Gaussian kernel smoothing radius.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    fsmooth : 3darray
        Gaussian smoothed field.
    """
    # Create fourier mode grid
    kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, MPI)
    # Smooth in Fourier space
    fk = fk*shift.cart.convolve_gaussian(kmag, Rn)
    # Backward FFT
    fsmooth = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, MPI)
    return fsmooth
