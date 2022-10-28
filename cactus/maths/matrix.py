import numpy as np

from .. import src


def _get_eig_2by2(Mxx, Mxy, Myx, Myy):
    """Returns the eigenvalues for a single 2x2 matrix.

    Parameters
    ----------
    Mxx, Mxy, Myx, Myy : float
        Value of the 2x2 matrix.

    Returns
    -------
    eig : array
        Eigenvalues for single matrix.
    """
    m = np.array([Mxx,Mxy,Myx,Myy])
    eig = src.eig2by2(m=m)
    return eig


def get_eig_2by2(Mxx, Mxy, Myx, Myy):
    """Returns the eigenvalues for a 2x2 matrices.

    Parameters
    ----------
    Mxx, Mxy, Myx, Myy : float or array
        Value(s) of 2x2 matrices.

    Returns
    -------
    eig : array
        Eigenvalues for single or multiple matrices.
    """
    if np.isscalar(Mxx):
        eig = _get_eig_2by2(Mxx,Mxy,Myx,Myy)
    else:
        eig = np.array([_get_eig_2by2(Mxx[i],Mxy[i],Myx[i],Myy[i])
                        for i in range(0, len(Mxx))])
    return eig


def _get_eig_3by3(Mxx, Mxy, Mxz, Myy, Myz, Mzz):
    """Returns the eigenvalues for a single symmetric 3x3 matrix.

    Parameters
    ----------
    Mxx, Mxy, Mxz, Myy, Myz, Mzz : float
        Value of the symmetric 3x3 matrix.

    Returns
    -------
    eig : array
        Eigenvalues for single matrix.
    """
    m = np.array([Mxx,Mxy,Mxz,Mxy,Myy,Myz,Mxz,Myz,Mzz])
    eig = src.symeig3by3(m=m)
    return eig


def get_eig_3by3(Mxx, Mxy, Mxz, Myy, Myz, Mzz):
    """Returns the eigenvalues for a symmetric 3x3 matrices.

    Parameters
    ----------
    Mxx, Mxy, Mxz, Myy, Myz, Mzz : float or array
        Value(s) of the symmetric 3x3 matrices.

    Returns
    -------
    eig : array
        Eigenvalues for single or multiple matrices.
    """
    if np.isscalar(Mxx):
        eig = _get_eig_3by3(Mxx,Mxy,Mxz,Myy,Myz,Mzz)
    else:
        eig = np.array([_get_eig_3by3(Mxx[i],Mxy[i],Mxz[i],
                                      Myy[i],Myz[i],Mzz[i])
                        for i in range(0, len(Mxx))])
    return eig
