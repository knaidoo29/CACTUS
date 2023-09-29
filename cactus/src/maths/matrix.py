import numpy as np

from .. import fortran_src as fsrc


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
        eig1, eig2 = fsrc.eig2by2_single(m00=Mxx, m01=Mxy, m10=Myx, m11=Myy)
    else:
        eig1, eig2 = fsrc.eig2by2_array(m00=Mxx, m01=Mxy, m10=Myx, m11=Myy,
            mlen=len(Mxx))
    return eig1, eig2


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
        eig1, eig2, eig3 = fsrc.sym_eig3by3_single(m00=Mxx, m01=Mxy, m02=Mxz,
            m11=Myy, m12=Myz, m22=Mzz)
    else:
        eig1, eig2, eig3 = fsrc.sym_eig3by3_array(m00=Mxx, m01=Mxy, m02=Mxz,
            m11=Myy, m12=Myz, m22=Mzz, mlen=len(Mxx))
    return eig1, eig2, eig3
