import numpy as np

from ...ext import shift, magpie


def smooth3D(f, Rn, boxsize, ngrid, boundary='periodic'):
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
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).

    Returns
    -------
    fsmooth : 3darray
        Gaussian smoothed field.
    """
    # Create fourier mode grid
    ngrid = len(f)
    if boundary == 'periodic':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    elif boundary == 'neumann':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D_dct(boxsize, ngrid)
    elif boundary == 'dirichlet':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D_dst(boxsize, ngrid)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    if boundary == 'periodic':
        fk = shift.cart.fft3D(f, boxsize)
    elif boundary == 'neumann':
        fk = shift.cart.dct3D(f, boxsize)
    elif boundary == 'dirichlet':
        fk = shift.cart.dst3D(f, boxsize)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    # Smooth in Fourier space
    fk = fk*shift.cart.convolve_gaussian(kmag, Rn)
    # Backward FFT
    if boundary == 'periodic':
        fsmooth = shift.cart.ifft3D(fk, boxsize)
    elif boundary == 'neumann':
        fsmooth = shift.cart.idct3D(fk, boxsize)
    elif boundary == 'dirichlet':
        fsmooth = shift.cart.idst3D(fk, boxsize)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    return fsmooth


def mpi_smooth3D(f, Rn, boxsize, ngrid, MPI, boundary='periodic'):
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
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).

    Returns
    -------
    fsmooth : 3darray
        Gaussian smoothed field.
    """
    # Create fourier mode grid
    if boundary == 'periodic':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    if boundary == 'periodic':
        fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        fk = shift.cart.mpi_dct3D(f, boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        fk = shift.cart.mpi_dst3D(f, boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    # Smooth in Fourier space
    fk = fk*shift.cart.convolve_gaussian(kmag, Rn)
    # Backward FFT
    if boundary == 'periodic':
        fsmooth = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        fsmooth = shift.cart.mpi_idct3D(fk, boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        fsmooth = shift.cart.mpi_idst3D(fk, boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    return fsmooth
