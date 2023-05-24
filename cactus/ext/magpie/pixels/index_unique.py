import numpy as np


def get_unique_pixID(pixID):
    """Returns unique pixels and counts for those pixels.

    Parameters
    ----------
    pixID : array
        Pixel indices.

    Returns
    -------
    upixID : array
        Unique pixel indices.
    cpixID : array
        Counts per upix.
    """
    upixID, cpixID = np.unique(pixID, return_counts=True)
    return upixID, cpixID
