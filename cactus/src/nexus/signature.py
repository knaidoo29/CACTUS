import numpy as np

from ...ext import shift, fiesta
from .. import maths, filters
from .. import fortran_src as fsrc


def get_theta(x):
    """Heaviside function with center at 0.

    Parameters
    ----------
    x : float or array
        Array.
    """
    return np.heaviside(x, 0.)


def get_Theta(x):
    """Positive linear function.

    Parameters
    ----------
    x : float or array
        Array.
    """
    return x*get_theta(x)


def get_nexus_signature(eig1, eig2, eig3):
    """Nexus+ strenght S.

    Parameters
    ----------
    eig1, eig2, eig3 : float or arrays
        Eigenvalues with sorted eigenvalues, i.e. eig1 < eig2 < eig3.

    Returns
    -------
    Sc, Sf, Sw : array
        Signature for clusters (Sc), filament (Sf), walls (Sw).
    """
    Ic = abs(eig3/eig1)
    If = abs(eig2/eig1)*get_Theta(1-abs(eig3/eig1))
    Iw = get_Theta(1-abs(eig2/eig1))*get_Theta(1-abs(eig3/eig1))
    Sc = Ic*abs(eig3)*get_theta(-eig1)*get_theta(-eig2)*get_theta(-eig3)
    Sf = If*abs(eig2)*get_theta(-eig1)*get_theta(-eig2)
    Sw = Iw*abs(eig1)*get_theta(-eig1)
    return Sc, Sf, Sw


def _Hessian3D_Fourier(f, Rn, boxsize, ngrid):
    """Compute Hessian matrix for the field f, computed in Fourier space.

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
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    # Create Fourier mode grid
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.fft3D(f, boxsize)
    # Computer Hessian xx
    Hkxx = (Rn**2.) * shift.cart.dfdk2(kx3d, fk)
    Hxx = shift.cart.ifft3D(Hkxx, boxsize)
    # Computer Hessian xy
    Hkxy = (Rn**2.) * shift.cart.dfdk2(kx3d, fk, k2=ky3d)
    Hxy = shift.cart.ifft3D(Hkxy, boxsize)
    # Computer Hessian xz
    Hkxz = (Rn**2.) * shift.cart.dfdk2(kx3d, fk, k2=kz3d)
    Hxz = shift.cart.ifft3D(Hkxz, boxsize)
    # Computer Hessian yy
    Hkyy = (Rn**2.) * shift.cart.dfdk2(ky3d, fk)
    Hyy = shift.cart.ifft3D(Hkyy, boxsize)
    # Computer Hessian yz
    Hkyz = (Rn**2.) * shift.cart.dfdk2(ky3d, fk, k2=kz3d)
    Hyz = shift.cart.ifft3D(Hkyz, boxsize)
    # Computer Hessian zz
    Hkzz = (Rn**2.) * shift.cart.dfdk2(kz3d, fk)
    Hzz = shift.cart.ifft3D(Hkzz, boxsize)
    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz


def _Hessian3D_Real(f, Rn, boxsize, ngrid, periodic=True):
    """Compute Hessian matrix for the field f.

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
    periodic : bool, optional
        Periodic boundarys.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    _, xgrid = shift.cart.grid1D(boxsize, ngrid)
    _, ygrid = shift.cart.grid1D(boxsize, ngrid)
    _, zgrid = shift.cart.grid1D(boxsize, ngrid)
    Hx = fiesta.maths.dfdx(xgrid, f, periodic=periodic)
    Hy = fiesta.maths.dfdy(ygrid, f, periodic=periodic)
    Hz = fiesta.maths.dfdz(zgrid, f, periodic=periodic)
    Hxx = (Rn**2.)*fiesta.maths.dfdx(xgrid, Hx, periodic=periodic)
    Hxy = (Rn**2.)*fiesta.maths.dfdy(ygrid, Hx, periodic=periodic)
    Hxz = (Rn**2.)*fiesta.maths.dfdz(zgrid, Hx, periodic=periodic)
    del Hx
    Hyy = (Rn**2.)*fiesta.maths.dfdy(ygrid, Hy, periodic=periodic)
    Hyz = (Rn**2.)*fiesta.maths.dfdz(zgrid, Hy, periodic=periodic)
    del Hy
    Hzz = (Rn**2.)*fiesta.maths.dfdz(zgrid, Hz, periodic=periodic)
    del Hz
    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz


def Hessian3D(f, Rn, boxsize, ngrid, periodic=True, usereal=True):
    """Compute Hessian matrix for the field f.

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
    periodic : bool, optional
        Periodic boundarys.
    usereal : bool, optional
        Compute in real space.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    if usereal:
        return _Hessian3D_Real(f, Rn, boxsize, ngrid, periodic=periodic)
    else:
        if periodic == False:
            print(' Non-periodic conditions can only be computed in real space.')
            return _Hessian3D_Real(f, Rn, boxsize, ngrid, periodic=periodic)
        else:
            return _Hessian3D_Fourier(f, Rn, boxsize, ngrid)


def _mpi_Hessian3D_Fourier(f, Rn, boxsize, ngrid, MPI):
    """Compute Hessian matrix for the field f in Fourier space.

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
    MPI : object
        MPIutils object.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    # Create fourier mode grid
    kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # Forward FFT
    fk = shift.cart.mpi_fft3D(f, boxsize, ngrid, MPI)
    # Computer Hessian xx
    Hkxx = (Rn**2.) * shift.cart.dfdk2(kx3d, fk)
    Hxx = shift.cart.mpi_ifft3D(Hkxx, boxsize, ngrid, MPI)
    # Computer Hessian xy
    Hkxy = (Rn**2.) * shift.cart.dfdk2(kx3d, fk, k2=ky3d)
    Hxy = shift.cart.mpi_ifft3D(Hkxy, boxsize, ngrid, MPI)
    # Computer Hessian xz
    Hkxz = (Rn**2.) * shift.cart.dfdk2(kx3d, fk, k2=kz3d)
    Hxz = shift.cart.mpi_ifft3D(Hkxz, boxsize, ngrid, MPI)
    # Computer Hessian yy
    Hkyy = (Rn**2.) * shift.cart.dfdk2(ky3d, fk)
    Hyy = shift.cart.mpi_ifft3D(Hkyy, boxsize, ngrid, MPI)
    # Computer Hessian yz
    Hkyz = (Rn**2.) * shift.cart.dfdk2(ky3d, fk, k2=kz3d)
    Hyz = shift.cart.mpi_ifft3D(Hkyz, boxsize, ngrid, MPI)
    # Computer Hessian zz
    Hkzz = (Rn**2.) * shift.cart.dfdk2(kz3d, fk)
    Hzz = shift.cart.mpi_ifft3D(Hkzz, boxsize, ngrid, MPI)
    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz


def _mpi_Hessian3D_Real(f, Rn, boxsize, ngrid, MPI, periodic=True):
    """Compute Hessian matrix for the field f in real space.

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
    MPI : object
        MPIutils object.
    periodic : bool, optional
        Periodic boundarys.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    _, xgrid = shift.cart.mpi_grid1D(boxsize, ngrid, MPI)
    _, ygrid = shift.cart.grid1D(boxsize, ngrid)
    _, zgrid = shift.cart.grid1D(boxsize, ngrid)
    Hx = fiesta.maths.mpi_dfdx(xgrid, f, MPI, periodic=periodic)
    Hy = fiesta.maths.mpi_dfdy(ygrid, f, MPI, periodic=periodic)
    Hz = fiesta.maths.mpi_dfdz(zgrid, f, MPI, periodic=periodic)
    Hxx = (Rn**2.)*fiesta.maths.mpi_dfdx(xgrid, Hx, MPI, periodic=periodic)
    Hxy = (Rn**2.)*fiesta.maths.mpi_dfdy(ygrid, Hx, MPI, periodic=periodic)
    Hxz = (Rn**2.)*fiesta.maths.mpi_dfdz(zgrid, Hx, MPI, periodic=periodic)
    del Hx
    Hyy = (Rn**2.)*fiesta.maths.mpi_dfdy(ygrid, Hy, MPI, periodic=periodic)
    Hyz = (Rn**2.)*fiesta.maths.mpi_dfdz(zgrid, Hy, MPI, periodic=periodic)
    del Hy
    Hzz = (Rn**2.)*fiesta.maths.mpi_dfdz(zgrid, Hz, MPI, periodic=periodic)
    del Hz
    return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz


def mpi_Hessian3D(f, Rn, boxsize, ngrid, MPI, periodic=True, usereal=True):
    """Compute Hessian matrix for the field f in real space.

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
    MPI : object
        MPIutils object.
    periodic : bool, optional
        Periodic boundarys.

    Returns
    -------
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz : ndarrays
        Hessian matrix elements.
    """
    if usereal:
        return _mpi_Hessian3D_Real(f, Rn, boxsize, ngrid, MPI, periodic=periodic)
    else:
        if periodic == False:
            print(' Non-periodic conditions can only be computed in real space.')
            return _mpi_Hessian3D_Real(f, Rn, boxsize, ngrid, MPI, periodic=periodic)
        else:
            return _mpi_Hessian3D_Fourier(f, Rn, boxsize, ngrid, MPI)


def get_nexus_sig(dens, boxsize, ngrid, logsmooth=True, R0=0.5, Nmax=7,
    boundary='periodic', verbose=True, verbose_prefix=''):
    """Classifies simulation based on multiscale Hessian of a log-Gaussian
    smoothed density field.

    Parameters
    ----------
    dens : 3darray
        Density of the 3d field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    logsmooth : bool
        Whether to smooth in logspace or normal.
    R0 : float, optional
        Minimum smoothing scale.
    Nmax : int, optional
        Number of smoothing scales, going by sqrt(2)^N * R0.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).
    verbose : bool, optional
        Determines whether to print updates about NEXUS+ calculation.
    verbose_prefix : str, optional
        Prefix to verbose print statements.

    Returns
    -------
    Sc, Sf, Sw : 3darray
        Nexus signature for cluster, filament and wall environment.
    """
    if boundary == 'periodic':
        periodic = True
    elif boundary == 'neumann' or boundary == 'dirichlet':
        periodic = False
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    # figure out shape
    dshape = np.shape(dens)
    # smoothing scales
    N = np.arange(Nmax)
    Rns = R0*np.sqrt(2.)**N
    # run signature calculation over smoothing scales
    for (n, Rn) in enumerate(Rns):
        # Smooth field
        if logsmooth:
            if verbose:
                print()
                print(verbose_prefix+"Smooth log(dens) at Rn = %.4f" % Rn)
            dsmooth = filters.logsmooth3D(dens, Rn, boxsize, ngrid,
                boundary=boundary, setzeroto=None, zero2min=True)
        else:
            if verbose:
                print()
                print(verbose_prefix+"Smooth dens at Rn = %.4f" % Rn)
            dsmooth = filters.smooth3D(dens, Rn, boxsize, ngrid, boundary=boundary)
        # Compute Hessian
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = Hessian3D(dsmooth, Rn, boxsize, ngrid,
            periodic=periodic)
        # Flatten Hessian
        Hxx = Hxx.flatten()
        Hxy = Hxy.flatten()
        Hxz = Hxz.flatten()
        Hyy = Hyy.flatten()
        Hyz = Hyz.flatten()
        Hzz = Hzz.flatten()
        if verbose:
            print(verbose_prefix+"Compute Nexus signature at Rn = %.4f" % Rn)
        # Computes eigenvalues
        eig1, eig2, eig3 = fsrc.sym_eig3by3_array(m00=Hxx, m01=Hxy, m02=Hxz,
            m11=Hyy, m12=Hyz, m22=Hzz, mlen=len(Hxx))
        _Sc, _Sf, _Sw = get_nexus_signature(eig1, eig2, eig3)
        # Find maximum across already evaluated smoothing scales
        if Rn == Rns[0]:
            Sc = np.copy(_Sc)
            Sf = np.copy(_Sf)
            Sw = np.copy(_Sw)
        else:
            cond = np.where(_Sc > Sc)[0]
            Sc[cond] = _Sc[cond]
            cond = np.where(_Sf > Sf)[0]
            Sf[cond] = _Sf[cond]
            cond = np.where(_Sw > Sw)[0]
            Sw[cond] = _Sw[cond]
    if verbose:
        print()
        print(verbose_prefix+"Output Nexus signature")
    Sc = Sc.reshape(dshape)
    Sf = Sf.reshape(dshape)
    Sw = Sw.reshape(dshape)
    return Sc, Sf, Sw



def mpi_get_nexus_sig(dens, boxsize, ngrid, MPI, logsmooth=True, R0=0.5, Nmax=7,
    boundary='periodic', verbose=True, verbose_prefix=''):
    """Classifies simulation based on multiscale Hessian of a log-Gaussian
    smoothed density field.

    Parameters
    ----------
    dens : 3darray
        Density of the 3d field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    MPI : obj
        MPIutils MPI object.
    logsmooth : bool
        Whether to smooth in logspace or normal.
    R0 : float, optional
        Minimum smoothing scale.
    Nmax : int, optional
        Number of smoothing scales, going by sqrt(2)^N * R0.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).
    verbose : bool, optional
        Determines whether to print updates about NEXUS+ calculation.
    verbose_prefix : str, optional
        Prefix to verbose print statements.

    Returns
    -------
    Sc, Sf, Sw : 3darray
        Nexus signature for cluster, filament and wall environment.
    """
    if boundary == 'periodic':
        periodic = True
    elif boundary == 'neumann' or boundary == 'dirichlet':
        periodic = False
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    # figure out shape
    dshape = np.shape(dens)
    # smoothing scales
    N = np.arange(Nmax)
    Rns = R0*np.sqrt(2.)**N
    # run signature calculation over smoothing scales
    for (n, Rn) in enumerate(Rns):
        # Smooth field
        if logsmooth:
            if verbose:
                MPI.mpi_print_zero()
                MPI.mpi_print_zero(verbose_prefix+"Smooth log(dens) at Rn = %.4f" % Rn)
            dsmooth = filters.mpi_logsmooth3D(dens, Rn, boxsize, ngrid, MPI,
                setzeroto=None, zero2min=True, boundary=boundary)
        else:
            if verbose:
                MPI.mpi_print_zero()
                MPI.mpi_print_zero(verbose_prefix+"Smooth dens at Rn = %.4f" % Rn)
            dsmooth = filters.mpi_smooth3D(dens, Rn, boxsize, ngrid, MPI,
                boundary=boundary)
        # Compute Hessian
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = mpi_Hessian3D(dsmooth, Rn, boxsize, ngrid,
            MPI, periodic=periodic)
        # Flatten Hessian
        Hxx = Hxx.flatten()
        Hxy = Hxy.flatten()
        Hxz = Hxz.flatten()
        Hyy = Hyy.flatten()
        Hyz = Hyz.flatten()
        Hzz = Hzz.flatten()
        if verbose:
            MPI.mpi_print_zero(verbose_prefix+"Compute Nexus signature at Rn = %.4f" % Rn)
        # Computes eigenvalues
        eig1, eig2, eig3 = fsrc.sym_eig3by3_array(m00=Hxx, m01=Hxy, m02=Hxz,
            m11=Hyy, m12=Hyz, m22=Hzz, mlen=len(Hxx))
        _Sc, _Sf, _Sw = get_nexus_signature(eig1, eig2, eig3)
        # Find maximum across already evaluated smoothing scales
        if Rn == Rns[0]:
            Sc = np.copy(_Sc)
            Sf = np.copy(_Sf)
            Sw = np.copy(_Sw)
        else:
            cond = np.where(_Sc > Sc)[0]
            Sc[cond] = _Sc[cond]
            cond = np.where(_Sf > Sf)[0]
            Sf[cond] = _Sf[cond]
            cond = np.where(_Sw > Sw)[0]
            Sw[cond] = _Sw[cond]
    if verbose:
        MPI.mpi_print_zero()
        MPI.mpi_print_zero(verbose_prefix+"Output Nexus signature")
    Sc = Sc.reshape(dshape)
    Sf = Sf.reshape(dshape)
    Sw = Sw.reshape(dshape)
    return Sc, Sf, Sw
