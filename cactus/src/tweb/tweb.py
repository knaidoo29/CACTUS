import numpy as np
from ...ext import shift

from .. import density, maths


def run_tweb(dens, boxsize, smooth, threshold, verbose=True, prefix=''):
    """Returns the T-Web cosmic web classification from the input density
    field. Assuming periodic boundary conditions.

    Parameters
    ----------
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box from which the density field has been given.
    smooth : float
        Size of the Gaussian smoothing applied to the density field.
    threshold : float
        Threshold for T-Web eigenvalue classifications
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
    ngrid = dshape[0]
    # get grids
    if verbose:
        print(prefix + 'Convert density to density contrast.')
    delta = density.norm_dens(dens) - 1.
    # get grids
    if verbose:
        print(prefix + 'Constructing real and fourier grids.')
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid)
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # smooth fields
    if verbose:
        print(prefix + 'Forward FFT and Smoothing density field.')
    deltak = shift.cart.fft3D(delta, boxsize)
    deltak *= shift.cart.convolve_gaussian(kmag, smooth)
    # compute potential field.
    if verbose:
        print(prefix + 'Computing potential field in fourier space.')
    cond = np.where(kmag != 0)
    phik = np.zeros(np.shape(deltak)) + 1j*np.zeros(np.shape(deltak))
    phik[cond] = -deltak[cond]/kmag[cond]**2.
    # differentiate in Fourier space
    if verbose:
        print(prefix + 'Differentiating potential field in fourier space and run backward FFT.')
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
    eigs = maths.get_eig_3by3(phi_xx, phi_xy, phi_xz, phi_yy, phi_yz, phi_zz)
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
