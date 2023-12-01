import numpy as np
from ...ext import shift, fiesta

from .. import density, maths


def run_tweb(dens, boxsize, ngrid, threshold, Rsmooth=None, boundary='periodic',
    usereal=True, verbose=True, prefix=''):
    """Returns the T-Web cosmic web classification from the input density
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    threshold : float
        Threshold for T-Web eigenvalue classifications
    Rsmooth : float, optional
        Size of the Gaussian smoothing applied to the density field.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).
    usereal : bool, optional
        Compute in real space.
    verbose : bool, optional
        Determines whether to print updates about T-Web calculation.
    prefix : str, optional
        Optional prefix before any print statements.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification where: 0 = void, 1 = wall, 2 = filament and
        3 = cluster.
    """
    dshape = np.shape(dens)
    # get grids
    if verbose:
        print(prefix + 'Convert density to density contrast.')
    delta = density.norm_dens(dens) - 1.
    # get grids
    if verbose:
        print(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid)
    if boundary == 'periodic':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    elif boundary == 'neumann':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D_dct(boxsize, ngrid)
    elif boundary == 'dirichlet':
        kx3d, ky3d, kz3d = shift.cart.kgrid3D_dst(boxsize, ngrid)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # smooth fields
    if verbose:
        print(prefix + 'Forward FFT')
    if boundary == 'periodic':
        deltak = shift.cart.fft3D(delta, boxsize)
    elif boundary == 'neumann':
        deltak = shift.cart.dct3D(delta, boxsize)
    elif boundary == 'dirichlet':
        deltak = shift.cart.dst3D(delta, boxsize)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    if Rsmooth is not None:
        if verbose:
            print(prefix + 'Smoothing density field.')
        deltak *= shift.cart.convolve_gaussian(kmag, Rsmooth)
    # compute potential field.
    if verbose:
        print(prefix + 'Computing potential field in fourier space.')
    cond = np.where(kmag != 0)
    if boundary == 'periodic':
        phik = np.zeros(np.shape(deltak)) + 1j*np.zeros(np.shape(deltak))
    elif boundary == 'neumann' or boundary == 'dirichlet':
        phik = np.zeros(np.shape(deltak))
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    phik[cond] = -deltak[cond]/kmag[cond]**2.
    if usereal:
        if boundary == 'periodic':
            phi = shift.cart.ifft3D(phik, boxsize)
        elif boundary == 'neumann':
            phi = shift.cart.idct3D(phik, boxsize)
        elif boundary == 'dirichlet':
            phi = shift.cart.idst3D(phik, boxsize)
        else:
            assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
        # differentiate in Real space
        if verbose:
            print(prefix + 'Differentiating gravitational potential.')
        _, xgrid = shift.cart.grid1D(boxsize, ngrid)
        _, ygrid = shift.cart.grid1D(boxsize, ngrid)
        _, zgrid = shift.cart.grid1D(boxsize, ngrid)
        phi_x = fiesta.maths.dfdx(xgrid, phi, periodic=periodic)
        phi_y = fiesta.maths.dfdy(ygrid, phi, periodic=periodic)
        phi_z = fiesta.maths.dfdz(zgrid, phi, periodic=periodic)
        phi_xx = fiesta.maths.dfdx(xgrid, phi_x, periodic=periodic)
        phi_xy = fiesta.maths.dfdy(ygrid, phi_x, periodic=periodic)
        phi_xz = fiesta.maths.dfdz(zgrid, phi_x, periodic=periodic)
        del phi_x
        phi_yx = fiesta.maths.dfdx(xgrid, phi_y, periodic=periodic)
        phi_yy = fiesta.maths.dfdy(ygrid, phi_y, periodic=periodic)
        del phi_y
        phi_zz = fiesta.maths.dfdz(zgrid, phi_z, periodic=periodic)
        del phi_z
    else:
        if boundary == 'neumann' or boundary == 'dirichlet':
            if verbose:
                print(prefix + '[Note: Differentials will be computed with FFTs. Meaning ')
                print(prefix + ' boundaries are assumed to be periodic and %s' % boundary)
                print(prefix + " are not satisfied. Switch 'usereal=True' to compute in real space.]")
        # differentiate in Fourier space
        if verbose:
            print(prefix + 'Differentiating gravitational potential and running backward FFT.')
        phi_xxk = shift.cart.dfdk2(kx3d, phik, k2=None)
        phi_xx = shift.cart.ifft3D(phi_xxk, boxsize)
        del phi_xxk
        phi_xyk = shift.cart.dfdk2(kx3d, phik, k2=ky3d)
        phi_xy = shift.cart.ifft3D(phi_xyk, boxsize)
        del phi_xyk
        phi_xzk = shift.cart.dfdk2(kx3d, phik, k2=kz3d)
        phi_xz = shift.cart.ifft3D(phi_xzk, boxsize)
        del phi_xzk
        phi_yyk = shift.cart.dfdk2(ky3d, phik, k2=None)
        phi_yy = shift.cart.ifft3D(phi_yyk, boxsize)
        del phi_yyk
        phi_yzk = shift.cart.dfdk2(ky3d, phik, k2=kz3d)
        phi_yz = shift.cart.ifft3D(phi_yzk, boxsize)
        del phi_yzk
        phi_zzk = shift.cart.dfdk2(kz3d, phik, k2=None)
        phi_zz = shift.cart.ifft3D(phi_zzk, boxsize)
        del phi_zzk
    phi_xx = phi_xx.flatten()
    phi_xy = phi_xy.flatten()
    phi_xz = phi_xz.flatten()
    phi_yy = phi_yy.flatten()
    phi_yz = phi_yz.flatten()
    phi_zz = phi_zz.flatten()
    if verbose:
        print(prefix + 'Calculating eigenvalues.')
    eig1, eig2, eig3 = maths.get_eig_3by3(phi_xx, phi_xy, phi_xz, phi_yy, phi_yz,
        phi_zz)
    if verbose:
        print(prefix + 'Determining cosmic web environments.')
    cweb = np.zeros(len(eig1))
    cond = np.where((eig3 >= threshold) & (eig2 < threshold) & (eig1 < threshold))[0]
    cweb[cond] = 1.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 < threshold))[0]
    cweb[cond] = 2.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 >= threshold))[0]
    cweb[cond] = 3.
    cweb = cweb.reshape(dshape)
    return cweb


def mpi_run_tweb(dens, boxsize, ngrid, threshold, MPI, Rsmooth=None,
    boundary='periodic', usereal=True, verbose=True, prefix=''):
    """Returns the T-Web cosmic web classification from the input density
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box from which the density field has been given.
    ngrid : int
        Grid size along each axis.
    threshold : float
        Threshold for T-Web eigenvalue classifications.
    MPI : object
        MPIutils object.
    Rsmooth : float, optional
        Size of the Gaussian smoothing applied to the density field.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).
    usereal : bool, optional
        Compute in real space.
    verbose : bool, optional
        Determines whether to print updates about T-Web calculation.
    prefix : str, optional
        Optional prefix before any print statements.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification where: 0 = void, 1 = wall, 2 = filament and
        3 = cluster.
    """
    dshape = np.shape(dens)
    # get grids
    if verbose:
        MPI.mpi_print_zero(prefix + 'Convert density to density contrast.')
    delta = density.mpi_norm_dens(dens, MPI) - 1.
    # get grids
    if verbose:
        MPI.mpi_print_zero(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.mpi_grid3D(boxsize, ngrid, MPI)
    if boundary == 'periodic':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # smooth fields
    if verbose:
        MPI.mpi_print_zero(prefix + 'Forward FFT')
    if boundary == 'periodic':
        deltak = shift.cart.mpi_fft3D(delta, boxsize, ngrid, MPI)
    elif boundary == 'neumann':
        deltak = shift.cart.mpi_dct3D(delta, boxsize, ngrid, MPI)
    elif boundary == 'dirichlet':
        deltak = shift.cart.mpi_dst3D(delta, boxsize, ngrid, MPI)
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    if Rsmooth is not None:
        if verbose:
            MPI.mpi_print_zero(prefix + 'Smoothing density field.')
        deltak *= shift.cart.convolve_gaussian(kmag, Rsmooth)
    # compute potential field.
    if verbose:
        MPI.mpi_print_zero(prefix + 'Computing potential field in fourier space.')
    cond = np.where(kmag != 0)
    if boundary == 'periodic':
        phik = np.zeros(np.shape(deltak)) + 1j*np.zeros(np.shape(deltak))
    elif boundary == 'neumann' or boundary == 'dirichlet':
        phik = np.zeros(np.shape(deltak))
    else:
        assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
    phik[cond] = -deltak[cond]/kmag[cond]**2.
    if usereal:
        # differentiate in Real space
        if boundary == 'periodic':
            phi = shift.cart.mpi_ifft3D(phik, boxsize, ngrid, MPI)
        elif boundary == 'neumann':
            phi = shift.cart.mpi_idct3D(phik, boxsize, ngrid, MPI)
        elif boundary == 'dirichlet':
            phi = shift.cart.mpi_idst3D(phik, boxsize, ngrid, MPI)
        else:
            assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
        _, xgrid = shift.cart.mpi_grid1D(boxsive, ngrid, MPI)
        _, ygrid = shift.cart.grid1D(boxsize, ngrid)
        _, zgrid = shift.cart.grid1D(boxsize, ngrid)
        phi_x = fiesta.maths.mpi_dfdx(xgrid, phi, MPI, periodic=periodic)
        phi_y = fiesta.maths.mpi_dfdy(ygrid, phi, MPI, periodic=periodic)
        phi_z = fiesta.maths.mpi_dfdz(zgrid, phi, MPI, periodic=periodic)
        phi_xx = fiesta.maths.mpi_dfdx(xgrid, phi_x, MPI, periodic=periodic)
        phi_xy = fiesta.maths.mpi_dfdy(ygrid, phi_x, MPI, periodic=periodic)
        phi_xz = fiesta.maths.mpi_dfdz(zgrid, phi_x, MPI, periodic=periodic)
        del phi_x
        phi_yx = fiesta.maths.mpi_dfdx(xgrid, phi_y, MPI, periodic=periodic)
        phi_yy = fiesta.maths.mpi_dfdy(ygrid, phi_y, MPI, periodic=periodic)
        del phi_y
        phi_zz = fiesta.maths.mpi_dfdz(zgrid, phi_z, MPI, periodic=periodic)
        del phi_z
    else:
        if boundary == 'neumann' or boundary == 'dirichlet':
            if verbose:
                MPI.mpi_print_zero(prefix + '[Note: Differentials will be computed with FFTs. Meaning ')
                MPI.mpi_print_zero(prefix + ' boundaries are assumed to be periodic and %s' % boundary)
                MPI.mpi_print_zero(prefix + " are not satisfied. Switch 'usereal=True' to compute in real space.]")
        # differentiate in Fourier space
        if verbose:
            MPI.mpi_print_zero(prefix + 'Differentiating gravitational potential and running backward FFT.')
        phi_xxk = shift.cart.dfdk2(kx3d, phik, k2=None)
        phi_xx = shift.cart.mpi_ifft3D(phi_xxk, boxsize, ngrid, MPI)
        del phi_xxk
        phi_xyk = shift.cart.dfdk2(kx3d, phik, k2=ky3d)
        phi_xy = shift.cart.mpi_ifft3D(phi_xyk, boxsize, ngrid, MPI)
        del phi_xyk
        phi_xzk = shift.cart.dfdk2(kx3d, phik, k2=kz3d)
        phi_xz = shift.cart.mpi_ifft3D(phi_xzk, boxsize, ngrid, MPI)
        del phi_xzk
        phi_yyk = shift.cart.dfdk2(ky3d, phik, k2=None)
        phi_yy = shift.cart.mpi_ifft3D(phi_yyk, boxsize, ngrid, MPI)
        del phi_yyk
        phi_yzk = shift.cart.dfdk2(ky3d, phik, k2=kz3d)
        phi_yz = shift.cart.mpi_ifft3D(phi_yzk, boxsize, ngrid, MPI)
        del phi_yzk
        phi_zzk = shift.cart.dfdk2(kz3d, phik, k2=None)
        phi_zz = shift.cart.mpi_ifft3D(phi_zzk, boxsize, ngrid, MPI)
        del phi_zzk
    phi_xx = phi_xx.flatten()
    phi_xy = phi_xy.flatten()
    phi_xz = phi_xz.flatten()
    phi_yy = phi_yy.flatten()
    phi_yz = phi_yz.flatten()
    phi_zz = phi_zz.flatten()
    if verbose:
        MPI.mpi_print_zero(prefix + 'Calculating eigenvalues.')
    eig1, eig2, eig3 = maths.get_eig_3by3(phi_xx, phi_xy, phi_xz, phi_yy, phi_yz,
        phi_zz)
    if verbose:
        MPI.mpi_print_zero(prefix + 'Determining cosmic web environments.')
    cweb = np.zeros(len(eig1))
    cond = np.where((eig3 >= threshold) & (eig2 < threshold) & (eig1 < threshold))[0]
    cweb[cond] = 1.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 < threshold))[0]
    cweb[cond] = 2.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 >= threshold))[0]
    cweb[cond] = 3.
    cweb = cweb.reshape(dshape)
    return cweb
