import numpy as np

from .. import src


def bin_pix(pixID, pixlen, weights=None):
    """Bin weights by pixel index.

    Parameters
    ----------
    pixID : array
        Pixel indices.
    pixlen : int
        Size of the pixel grid.
    weights : array, optional
        Pixel index weights, if None is given this is assumed to be one.

    Returns
    -------
    pix : array
        Binned pixel values.
    """
    if weights is None:
        weights = np.ones(len(pixID))
    pix = src.bin_by_index(pix_id=pixID, id_weights=weights,
                           id_len=len(pixID), pix_len=pixlen)
    return pix
