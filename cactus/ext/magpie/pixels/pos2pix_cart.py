import numpy as np

from .. import src
from .. import utils


def pos2pix_cart1d(x, length, ngrid, origin=0.):
    """X-coordinate to 1D pixel.

    Parameters
    ----------
    x : float or array
        X-coordinate.
    length : float
        Length of the 1D grid.
    ngrid : int
        The grid dimension.
    origin : float, optional
        Minimum value along the grid.

    Returns
    -------
    pixID : int or array
        The index of the pixel the points are located in.
    """
    dx = length/ngrid
    if utils.isscalar(x) is True:
        pixID = src.which_pix_id_scalar(x=x, xmin=origin, dx=dx)
    else:
        pixID = src.which_pix_id_array(x=x, xmin=origin, dx=dx, xlen=len(x))
    return pixID


def pos2pix_cart2d(x, y, lengths, ngrids, origin=[0., 0.]):
    """X and Y-coordinate to 2D pixel.

    Parameters
    ----------
    x, y : float or array
        X and Y-coordinate.
    lengths : float or list[float]
        Lengths of the 2D grids.
    ngrids : int or list[int]
        The grid dimension.
    origin : list[float], optional
        Minimum value along the grids.

    Returns
    -------
    pixID : int or array
        The index of the pixel the points are located in 2D.
    """
    if utils.isscalar(lengths):
        _lengths = [lengths, lengths]
    else:
        _lengths = lengths
    if utils.isscalar(ngrids):
        _ngrids = [ngrids, ngrids]
    else:
        _ngrids = ngrids
    if utils.isscalar(origin) is True:
        _origin = [origin, origin]
    else:
        _origin = origin
    xpixID = pos2pix_cart1d(x, _lengths[0], _ngrids[0], origin=_origin[0])
    ypixID = pos2pix_cart1d(y, _lengths[1], _ngrids[1], origin=_origin[1])
    if utils.isscalar(x) is True:
        pixID = src.pix_id_1dto2d_scalar(xpix_id=xpixID, ypix_id=ypixID,
                                         ygrid=_ngrids[1])
    else:
        pixID = src.pix_id_1dto2d_array(xpix_id=xpixID, ypix_id=ypixID,
                                        xlen=len(x), ygrid=_ngrids[1])
    return pixID


def pos2pix_cart3d(x, y, z, lengths, ngrids, origin=[0., 0., 0.]):
    """X, Y and Z-coordinate to 3D pixel.

    Parameters
    ----------
    x, y, z : float or array
        X, Y and Z-coordinate.
    lengths : float or list[float]
        Lengths of the 3D grids.
    ngrids : int or list[int]
        The grid dimension.
    origin : list[float], optional
        Minimum value along the grids.

    Returns
    -------
    pixID : int or array
        The index of the pixel the points are located in 3D.
    """
    if utils.isscalar(lengths):
        _lengths = [lengths, lengths, lengths]
    else:
        _lengths = lengths
    if utils.isscalar(ngrids):
        _ngrids = [ngrids, ngrids, ngrids]
    else:
        _ngrids = ngrids
    if utils.isscalar(origin) is True:
        _origin = [origin, origin, origin]
    else:
        _origin = origin
    xpixID = pos2pix_cart1d(x, _lengths[0], _ngrids[0], origin=_origin[0])
    ypixID = pos2pix_cart1d(y, _lengths[1], _ngrids[1], origin=_origin[1])
    zpixID = pos2pix_cart1d(z, _lengths[2], _ngrids[2], origin=_origin[2])
    if utils.isscalar(x) is True:
        pixID = src.pix_id_1dto3d_scalar(xpix_id=xpixID, ypix_id=ypixID,
                                         zpix_id=zpixID, ygrid=_ngrids[1],
                                         zgrid=_ngrids[2])
    else:
        pixID = src.pix_id_1dto3d_array(xpix_id=xpixID, ypix_id=ypixID,
                                        zpix_id=zpixID, xlen=len(x),
                                        ygrid=_ngrids[1], zgrid=_ngrids[2])
    return pixID
