import numpy as np

from ...ext import shift


def tophat3D(f, R, boxsize, ngrid, boundary='periodic'):
    """Top-hat filter in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    rmin : float
        Minimum value.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).

    Returns
    -------
    ftophat : 3darray
        Tophat filtered field.
    """
    # Create fourier mode grid
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
    WTH = np.ones(np.shape(fk))
    cond = np.where(kmag != 0.)
    WTH[cond] = 3*(np.sin(kmag[cond]*R) - (kmag[cond]*R)*np.cos(kmag[cond]*R))
    WTH[cond] /= (kmag[cond]*R)**3.
    fk *= WTH
    # Backward FFT
    if boundary == 'periodic':
        ftophat = shift.cart.ifft3D(fk, boxsize)
    elif boundary == 'neumann':
        ftophat = shift.cart.idct3D(fk, boxsize)
    elif boundary == 'dirichlet':
        ftophat = shift.cart.idst3D(fk, boxsize)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    return ftophat


def mpi_tophat3D(f, R, boxsize, ngrid, MPI, boundary='periodic'):
    """Top-hat filter in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    rmin : float
        Minimum value.
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
    ftophat : 3darray
        Tophat filtered field.
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
    WTH = np.ones(np.shape(fk))
    cond = np.where(kmag != 0.)
    WTH[cond] = 3*(np.sin(kmag[cond]*R) - (kmag[cond]*R)*np.cos(kmag[cond]*R))
    WTH[cond] /= (kmag[cond]*R)**3.
    fk *= WTH
    # Backward FFT
    if boundary == 'periodic':
        ftophat = shift.cart.mpi_ifft3D(fk, boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        ftophat = shift.cart.mpi_idct3D(fk, boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        ftophat = shift.cart.mpi_idst3D(fk, boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    return ftophat
