import numpy as np
from ...ext import shift

from .. import filters, maths


def run_vweb(vxf, vyf, vzf, boxsize, ngrid, threshold, Rsmooth=None,
    boundary='periodic', verbose=True, prefix=''):
    """Returns the V-Web cosmic web classification from the input velocity
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    vxf, vyf, vzf : 3darray
        Vector field in the x, y, z axis.
    boxsize : float
        Size of the box from which the vector field has been given.
    ngrid : int
        Grid size along each axis.
    threshold : float
        Threshold for V-Web eigenvalue classifications.
    Rsmooth : float
        Size of the Gaussian smoothing applied to the velocity field.
    boundary : str, optional
        Boundary conditions, either 'periodic' (FFT), 'neumann' (DCT) or
        'dirichlet' (DST).
    verbose : bool, optional
        Determines whether to print updates about V-Web calculation.
    prefix : str, optional
        Optional prefix before any print statements.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification where: 0 = void, 1 = wall, 2 = filament and
        3 = cluster.
    """
    vshape = np.shape(vxf)
    # get grids
    if verbose:
        print(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid)
    # smooth fields
    if smooth is not None:
        if verbose:
            print(prefix + 'Smoothing velocity fields in fourier space.')
        vxf = filters.smooth3D(vxf, Rsmooth, boxsize, ngrid, boundary=boundary)
        vyf = filters.smooth3D(vyf, Rsmooth, boxsize, ngrid, boundary=boundary)
        vzf = filters.smooth3D(vzf, Rsmooth, boxsize, ngrid, boundary=boundary)
    if usereal:
        # differentiate in Real space
        if verbose:
            print(prefix + 'Differentiating velocity.')
        _, xgrid = shift.cart.grid1D(boxsize, ngrid)
        _, ygrid = shift.cart.grid1D(boxsize, ngrid)
        _, zgrid = shift.cart.grid1D(boxsize, ngrid)
        vxf_x = fiesta.maths.dfdx(xgrid, vxf, periodic=periodic)
        vyf_y = fiesta.maths.dfdy(ygrid, vyf, periodic=periodic)
        vzf_z = fiesta.maths.dfdz(zgrid, vzf, periodic=periodic)
        vxf_xx = fiesta.maths.dfdx(xgrid, vxf_x, periodic=periodic)
        vxf_xy = fiesta.maths.dfdy(ygrid, vxf_x, periodic=periodic)
        vxf_xz = fiesta.maths.dfdz(zgrid, vxf_x, periodic=periodic)
        del vxf_x
        vyf_yx = fiesta.maths.dfdx(xgrid, vyf_y, periodic=periodic)
        vyf_yy = fiesta.maths.dfdy(ygrid, vyf_y, periodic=periodic)
        vyf_yz = fiesta.maths.dfdz(zgrid, vyf_y, periodic=periodic)
        del vyf_y
        vzf_zx = fiesta.maths.dfdx(xgrid, vzf_z, periodic=periodic)
        vzf_zy = fiesta.maths.dfdy(ygrid, vzf_z, periodic=periodic)
        vzf_zz = fiesta.maths.dfdz(zgrid, vzf_z, periodic=periodic)
        del vzf_z
    else:
        # differentiate in Fourier space
        if verbose:
            print(prefix + 'Differentiating velocity and running backward FFT.')
        if boundary == 'periodic':
            kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
        elif boundary == 'neumann':
            kx3d, ky3d, kz3d = shift.cart.kgrid3D_dct(boxsize, ngrid)
        elif boundary == 'dirichlet':
            kx3d, ky3d, kz3d = shift.cart.kgrid3D_dst(boxsize, ngrid)
        else:
            assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
        kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
        if verbose:
            print(prefix + 'Forward FFT.')
        vxk = shift.cart.fft3D(vxf, boxsize)
        vyk = shift.cart.fft3D(vyf, boxsize)
        vzk = shift.cart.fft3D(vzf, boxsize)
        vxxk = shift.cart.dfdk(kx3d, vxk)
        vxx = shift.cart.ifft3D(vxxk, boxsize)
        del vxxk
        vxyk = shift.cart.dfdk(ky3d, vxk)
        vxy = shift.cart.ifft3D(vxyk, boxsize)
        del vxyk
        vxzk = shift.cart.dfdk(kz3d, vxk)
        vxz = shift.cart.ifft3D(vxzk, boxsize)
        del vxzk
        vyxk = shift.cart.dfdk(kx3d, vyk)
        vyx = shift.cart.ifft3D(vyxk, boxsize)
        del vyxk
        vyyk = shift.cart.dfdk(ky3d, vyk)
        vyy = shift.cart.ifft3D(vyyk, boxsize)
        del vyyk
        vyzk = shift.cart.dfdk(kz3d, vyk)
        vyz = shift.cart.ifft3D(vyzk, boxsize)
        del vyzk
        vzxk = shift.cart.dfdk(kx3d, vzk)
        vzx = shift.cart.ifft3D(vzxk, boxsize)
        del vzxk
        vzyk = shift.cart.dfdk(ky3d, vzk)
        vzy = shift.cart.ifft3D(vzyk, boxsize)
        del vzyk
        vzzk = shift.cart.dfdk(kz3d, vzk)
        vzz = shift.cart.ifft3D(vzzk, boxsize)
        del vzzk
        # Calculate reduced velocity tensor matrix
        if verbose:
            print(prefix + 'Constructing reduced shear tensor.')
    Sigma_xx = vxx + vxx
    Sigma_xy = vxy + vyx
    Sigma_xz = vxz + vzx
    Sigma_yy = vyy + vyy
    Sigma_yz = vyz + vzy
    Sigma_zz = vzz + vzz
    # multiply by H0, note since the vector field is given on a grid in
    # comoving coordinates to make this unitless we must multiply by 100h,
    # hence why H0 = 100.
    H0 = 100.
    Sigma_xx *= -1./(2*H0)
    Sigma_xy *= -1./(2*H0)
    Sigma_xz *= -1./(2*H0)
    Sigma_yy *= -1./(2*H0)
    Sigma_yz *= -1./(2*H0)
    Sigma_zz *= -1./(2*H0)
    Sigma_xx = Sigma_xx.flatten()
    Sigma_xy = Sigma_xy.flatten()
    Sigma_xz = Sigma_xz.flatten()
    Sigma_yy = Sigma_yy.flatten()
    Sigma_yz = Sigma_yz.flatten()
    Sigma_zz = Sigma_zz.flatten()
    if verbose:
        print(prefix + 'Calculating eigenvalues.')
    eig1, eig2, eig3 = maths.get_eig_3by3(Sigma_xx, Sigma_xy, Sigma_xz, Sigma_yy,
        Sigma_yz, Sigma_zz)
    if verbose:
        print(prefix + 'Determining cosmic web environments.')
    cweb = np.zeros(len(eig1))
    cond = np.where((eig3 >= threshold) & (eig2 < threshold) & (eig1 < threshold))[0]
    cweb[cond] = 1.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 < threshold))[0]
    cweb[cond] = 2.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 >= threshold))[0]
    cweb[cond] = 3.
    cweb = cweb.reshape(vshape)
    return cweb


def mpi_run_vweb(vxf, vyf, vzf, boxsize, ngrid, threshold, MPI, smooth=None,
    verbose=True, prefix=''):
    """Returns the V-Web cosmic web classification from the input velocity
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    vxf, vyf, vzf : 3darray
        Vector field in the x, y, z axis.
    boxsize : float
        Size of the box from which the vector field has been given.
    ngrid : int
        Grid size along each axis.
    threshold : float
        Threshold for V-Web eigenvalue classifications.
    MPI : object
        MPIutils object.
    smooth : float
        Size of the Gaussian smoothing applied to the velocity field.
    verbose : bool, optional
        Determines whether to print updates about V-Web calculation.
    prefix : str, optional
        Optional prefix before any print statements.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification where: 0 = void, 1 = wall, 2 = filament and
        3 = cluster.
    """
    vshape = np.shape(vxf)
    # get grids
    if verbose:
        MPI.mpi_print_zero(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.mpi_grid3D(boxsize, ngrid, MPI)
    # smooth fields
    if smooth is not None:
        if verbose:
            MPI.mpi_print_zero(prefix + 'Smoothing velocity fields in fourier space.')
        vxf = filters.mpi_smooth3D(vxf, Rsmooth, boxsize, ngrid, MPI, boundary=boundary)
        vyf = filters.mpi_smooth3D(vyf, Rsmooth, boxsize, ngrid, MPI, boundary=boundary)
        vzf = filters.mpi_smooth3D(vzf, Rsmooth, boxsize, ngrid, MPI, boundary=boundary)
    if usereal:
        # differentiate in Real space
        if verbose:
            MPI.mpi_print_zero(prefix + 'Differentiating velocity.')
        _, xgrid = shift.cart.mpi_grid1D(boxsize, ngrid, MPI)
        _, ygrid = shift.cart.grid1D(boxsize, ngrid)
        _, zgrid = shift.cart.grid1D(boxsize, ngrid)
        vxf_x = fiesta.maths.mpi_dfdx(xgrid, vxf, MPI, periodic=periodic)
        vyf_y = fiesta.maths.mpi_dfdy(ygrid, vyf, MPI, periodic=periodic)
        vzf_z = fiesta.maths.mpi_dfdz(zgrid, vzf, MPI, periodic=periodic)
        vxf_xx = fiesta.maths.mpi_dfdx(xgrid, vxf_x, MPI, periodic=periodic)
        vxf_xy = fiesta.maths.mpi_dfdy(ygrid, vxf_x, MPI, periodic=periodic)
        vxf_xz = fiesta.maths.mpi_dfdz(zgrid, vxf_x, MPI, periodic=periodic)
        del vxf_x
        vyf_yx = fiesta.maths.mpi_dfdx(xgrid, vyf_y, MPI, periodic=periodic)
        vyf_yy = fiesta.maths.mpi_dfdy(ygrid, vyf_y, MPI, periodic=periodic)
        vyf_yz = fiesta.maths.mpi_dfdz(zgrid, vyf_y, MPI, periodic=periodic)
        del vyf_y
        vzf_zx = fiesta.maths.mpi_dfdx(xgrid, vzf_z, MPI, periodic=periodic)
        vzf_zy = fiesta.maths.mpi_dfdy(ygrid, vzf_z, MPI, periodic=periodic)
        vzf_zz = fiesta.maths.mpi_dfdz(zgrid, vzf_z, MPI, periodic=periodic)
        del vzf_z
    else:
        # differentiate in Fourier space
        if verbose:
            MPI.mpi_print_zero(prefix + 'Differentiating velocity and running backward FFT.')
        if boundary == 'periodic':
            kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
        elif boundary == 'neumann':
            kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dct(boxsize, ngrid, MPI)
        elif boundary == 'dirichlet':
            kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D_dst(boxsize, ngrid, MPI)
        else:
            assert False, "Boundary %s is not supported, must be periodic, neumann or dirichlet." % boundary
        kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
        if verbose:
            MPI.mpi_print_zero(prefix + 'Forward FFT.')
        vxk = shift.cart.mpi_fft3D(vxf, boxsize, ngrid, MPI)
        vyk = shift.cart.mpi_fft3D(vyf, boxsize, ngrid, MPI)
        vzk = shift.cart.mpi_fft3D(vzf, boxsize, ngrid, MPI)
        vxxk = shift.cart.dfdk(kx3d, vxk)
        vxx = shift.cart.mpi_ifft3D(vxxk, boxsize, ngrid, MPI)
        del vxxk
        vxyk = shift.cart.dfdk(ky3d, vxk)
        vxy = shift.cart.mpi_ifft3D(vxyk, boxsize, ngrid, MPI)
        del vxyk
        vxzk = shift.cart.dfdk(kz3d, vxk)
        vxz = shift.cart.mpi_ifft3D(vxzk, boxsize, ngrid, MPI)
        del vxzk
        vyxk = shift.cart.dfdk(kx3d, vyk)
        vyx = shift.cart.mpi_ifft3D(vyxk, boxsize, ngrid, MPI)
        del vyxk
        vyyk = shift.cart.dfdk(ky3d, vyk)
        vyy = shift.cart.mpi_ifft3D(vyyk, boxsize, ngrid, MPI)
        del vyyk
        vyzk = shift.cart.dfdk(kz3d, vyk)
        vyz = shift.cart.mpi_ifft3D(vyzk, boxsize, ngrid, MPI)
        del vyzk
        vzxk = shift.cart.dfdk(kx3d, vzk)
        vzx = shift.cart.mpi_ifft3D(vzxk, boxsize, ngrid, MPI)
        del vzxk
        vzyk = shift.cart.dfdk(ky3d, vzk)
        vzy = shift.cart.mpi_ifft3D(vzyk, boxsize, ngrid, MPI)
        del vzyk
        vzzk = shift.cart.dfdk(kz3d, vzk)
        vzz = shift.cart.mpi_ifft3D(vzzk, boxsize, ngrid, MPI)
        del vzzk
        # Calculate reduced velocity tensor matrix
        if verbose:
            MPI.mpi_print_zero(prefix + 'Constructing reduced shear tensor.')
    Sigma_xx = vxx + vxx
    Sigma_xy = vxy + vyx
    Sigma_xz = vxz + vzx
    Sigma_yy = vyy + vyy
    Sigma_yz = vyz + vzy
    Sigma_zz = vzz + vzz
    # multiply by H0, note since the vector field is given on a grid in
    # comoving coordinates to make this unitless we must multiply by 100h,
    # hence why H0 = 100.
    H0 = 100.
    Sigma_xx *= -1./(2*H0)
    Sigma_xy *= -1./(2*H0)
    Sigma_xz *= -1./(2*H0)
    Sigma_yy *= -1./(2*H0)
    Sigma_yz *= -1./(2*H0)
    Sigma_zz *= -1./(2*H0)
    Sigma_xx = Sigma_xx.flatten()
    Sigma_xy = Sigma_xy.flatten()
    Sigma_xz = Sigma_xz.flatten()
    Sigma_yy = Sigma_yy.flatten()
    Sigma_yz = Sigma_yz.flatten()
    Sigma_zz = Sigma_zz.flatten()
    if verbose:
        MPI.mpi_print_zero(prefix + 'Calculating eigenvalues.')
    eig1, eig2, eig3 = maths.get_eig_3by3(Sigma_xx, Sigma_xy, Sigma_xz, Sigma_yy,
        Sigma_yz, Sigma_zz)
    if verbose:
        MPI.mpi_print_zero(prefix + 'Determining cosmic web environments.')
    cweb = np.zeros(len(eig1))
    cond = np.where((eig3 >= threshold) & (eig2 < threshold) & (eig1 < threshold))[0]
    cweb[cond] = 1.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 < threshold))[0]
    cweb[cond] = 2.
    cond = np.where((eig3 >= threshold) & (eig2 >= threshold) & (eig1 >= threshold))[0]
    cweb[cond] = 3.
    cweb = cweb.reshape(vshape)
    return cweb
