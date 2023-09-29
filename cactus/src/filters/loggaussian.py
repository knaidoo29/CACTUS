import numpy as np

from . import gaussian
from ...ext import shift


def set_zero2val(f, val=None):
    """Removes zeros in a field.

    Parameters
    ----------
    f : ndarray
        Field.
    val : float, optional
        Value to set field values below zero. If None, this is set to
        minimum of field > 0.
    """
    # check if val is defined if not find minimum value > 0.
    if val is None:
        cond = np.where(f > 0.)
        val = np.min(f[cond])
    # replace zeros with val
    cond = np.where(f <= 0.)
    f[cond] = val
    return f


def logsmooth3D(f, Rn, boxsize, setzeroto=None, zero2min=True):
    """Smoothing a 3D field with a log gaussian kernel in Fourier space.

    Parameters
    ----------
    f : 3darray
        3D field.
    Rn : float
        Gaussian kernel smoothing radius.
    boxsize : float
        Size of the box.
    setzeroto : float, optional
        Set zeros to a given value.
    zero2min : bool, optional
        Sets zeros to minimum nonzero values in the field, if setzeroto=None.

    Returns
    -------
    fsmooth : 3darray
        Gaussian smoothed field.
    """
    # Correct for zeros.
    if setzeroto is None:
        if zero2min:
            f = set_zero2val(f)
    else:
        f = set_zero2val(f, val=setzeroto)
    # Find mean to correct logsmoothing to equal mean.
    fmean = np.mean(f)
    # Log field.
    logf = np.log10(f)
    # Apply smoothing in log space
    logfsmooth = gaussian.smooth3D(logf, Rn, boxsize)
    # un-log field
    fsmooth = 10.**logfsmooth
    # correct mean
    fsmoothmean = np.mean(fsmooth)
    fsmooth *= fmean/fsmoothmean
    return fsmooth
