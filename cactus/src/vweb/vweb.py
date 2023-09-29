import numpy as np
from ...ext import shift

from .. import maths


def run_vweb(vxf, vyf, vzf, boxsize, smooth, threshold, verbose=True, prefix=''):
    """Returns the V-Web cosmic web classification from the input velocity
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    vxf, vyf, vzf : 3darray
        Vector field in the x, y, z axis.
    boxsize : float
        Size of the box from which the vector field has been given.
    smooth : float
        Size of the Gaussian smoothing applied to the velocity field.
    threshold : float
        Threshold for V-Web eigenvalue classifications
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
    ngrid = vshape[0]
    # get grids
    if verbose:
        print(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid)
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # smooth fields
    if verbose:
        print(prefix + 'Forward FFT and Smoothing velocity fields in fourier space.')
    vxk = shift.cart.fft3D(vxf, boxsize)
    vxk *= shift.cart.convolve_gaussian(kmag, smooth)
    vyk = shift.cart.fft3D(vyf, boxsize)
    vyk *= shift.cart.convolve_gaussian(kmag, smooth)
    vzk = shift.cart.fft3D(vzf, boxsize)
    vzk *= shift.cart.convolve_gaussian(kmag, smooth)
    # differentiate in Fourier space
    if verbose:
        print(prefix + 'Differentiating velocity fields in fourier space and run backward FFT.')
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
    eigs = maths.get_eig_3by3(Sigma_xx, Sigma_xy, Sigma_xz, Sigma_yy, Sigma_yz,
        Sigma_zz)
    if verbose:
        print(prefix + 'Determining cosmic web environments.')
    cweb = np.zeros(len(eigs))
    cond = np.where((eigs[:,2]>=threshold) & (eigs[:,1]<threshold) & (eigs[:,0]<threshold))[0]
    cweb[cond] = 1.
    cond = np.where((eigs[:,2]>=threshold) & (eigs[:,1]>=threshold) & (eigs[:,0]<threshold))[0]
    cweb[cond] = 2.
    cond = np.where((eigs[:,2]>=threshold) & (eigs[:,1]>=threshold) & (eigs[:,0]>=threshold))[0]
    cweb[cond] = 3.
    cweb = cweb.reshape(vshape)
    return cweb
