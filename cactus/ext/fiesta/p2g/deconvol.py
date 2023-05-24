import numpy as np
from ... import shift

from .. import utils


def get_sinc(x):
    """Returns the sinc function for input x."""
    if np.isscalar(x) is True:
        if x == 0:
            sinc = 1.
        else:
            sinc = np.sin(x)/x
    else:
        sinc = np.ones(np.shape(x))
        condition = np.where(x != 0.)
        sinc[condition] = np.sin(x[condition])/x[condition]
    return sinc


def get_deconvol_p(method):
    """Returns the deconvolution power.

    Parameters
    ----------
    method : str
        grid assignment scheme, either NGP, CIC or TSC.

    Returns
    -------
    p : float
        Deconvolution power.
    """
    if method == 'NGP':
        p = 1.
    elif method == 'CIC':
        p = 2.
    elif method == 'TSC':
        p = 3.
    return p


def deconvolve_part2grid_2D(field, boxsize, method='TSC'):
    """Deconvolve the grid assignment scheme in Fourier space.

    Parameters
    ----------
    field : ndarray
        Grid assigned field.
    boxsize : float
        Box size.
    method : str
        grid assignment scheme, either NGP, CIC or TSC.
    """
    fieldk = shift.cart.forward_fft_2D(field, boxsize)
    ngrid = len(fieldk)
    kx2d, ky2d = shift.cart.get_fourier_grid_2D(boxsize, ngrid)
    kmag = utils.get_vector_magnitude_2D(kx2d, ky2d)
    sinc_x = get_sinc(kx2d*boxsize/(2.*ngrid))
    sinc_y = get_sinc(ky2d*boxsize/(2.*ngrid))
    p = get_deconvol_p(method)
    deconvol_factor = (sinc_x*sinc_y)**p
    fieldk = utils.complex_div(fieldk, deconvol_factor)
    field = shift.cart.backward_fft_2D(fieldk, boxsize)
    return field


def deconvolve_part2grid_3D(field, boxsize, method='TSC'):
    """Deconvolve the grid assignment scheme in Fourier space.

    Parameters
    ----------
    field : ndarray
        Grid assigned field.
    boxsize : float
        Box size.
    method : str
        grid assignment scheme, either NGP, CIC or TSC.
    """
    fieldk = shift.cart.forward_fft_3D(field, boxsize)
    ngrid = len(fieldk)
    kx3d, ky3d, kz3d = shift.cart.get_fourier_grid_3D(boxsize, ngrid)
    kmag = utils.get_vector_magnitude_3D(kx3d, ky3d, kz3d)
    sinc_x = get_sinc(kx3d*boxsize/(2.*ngrid))
    sinc_y = get_sinc(ky3d*boxsize/(2.*ngrid))
    sinc_z = get_sinc(ky3d*boxsize/(2.*ngrid))
    p = get_deconvol_p(method)
    deconvol_factor = (sinc_x*sinc_y*sinc_z)**p
    fieldk = utils.complex_div(fieldk, deconvol_factor)
    field = shift.cart.backward_fft_3D(fieldk, boxsize)
    return field
