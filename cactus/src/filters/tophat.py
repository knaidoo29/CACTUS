import numpy as np

from ...ext import shift


def tophat3D(f, R, boxsize):
    """Top-hat filter in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    rmin : float
        Minimum value.
    boxsize : float
        Size of the box.

    Returns
    -------
    ftophat : 3darray
        Tophat filtered field.
    """
    # Create fourier mode grid
    ngrid = len(f)
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.fft3D(f, boxsize)
    WTH = np.ones(np.shape(fk))
    cond = np.where(kmag != 0.)
    WTH[cond] = 3*(np.sin(kmag[cond]*R) - (kmag[cond]*R)*np.cos(kmag[cond]*R))
    WTH[cond] /= (kmag[cond]*R)**3.
    fk *= WTH
    # Backward FFT
    ftophat = shift.cart.ifft3D(fk, boxsize)
    return ftophat


def mpi_tophat3D(f, R, boxsize, MPI):
    """Top-hat filter in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    rmin : float
        Minimum value.
    boxsize : float
        Size of the box.
    MPI : obj
        MPIutils MPI object.

    Returns
    -------
    ftophat : 3darray
        Tophat filtered field.
    """
    # Create fourier mode grid
    ngrid = len(f)
    kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, MPI)
    WTH = np.ones(np.shape(fk))
    cond = np.where(kmag != 0.)
    WTH[cond] = 3*(np.sin(kmag[cond]*R) - (kmag[cond]*R)*np.cos(kmag[cond]*R))
    WTH[cond] /= (kmag[cond]*R)**3.
    fk *= WTH
    # Backward FFT
    ftophat = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, MPI)
    return ftophat
