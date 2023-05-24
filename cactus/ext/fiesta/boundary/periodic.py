import numpy as np

from .. import randoms


def buffer_periodic(data, axis, limits, buffer_length):
    """Returns periodic boundary conditions.

    Parameters
    ----------
    data : array
        Data for which we want periodic boundaries.
    axis : int
        The axis which we want periodic boundaries.
    limits : list
        Limits, i.e. min and max for the axis with periodic boundaries.
    buffer_length : float
        Length of buffer region for periodic conditions.

    Returns
    -------
    datap : array
        Data with periodic buffer particles.
    """
    cond1 = np.where(data[:,axis] >= limits[1]-buffer_length)[0]
    _d1 = data[cond1]
    _d1[:,axis] -= limits[1]-limits[0]
    cond2 = np.where(data[:,axis] <= limits[0]+buffer_length)[0]
    _d2 = data[cond2]
    _d2[:,axis] += limits[1]-limits[0]
    datap = np.vstack([data, _d1, _d2])
    return datap


def buffer_periodic_2D(data, boxsize, buffer_length, origin=0.):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Where columns 0 and 1 are x and y.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    origin : float, optional
        Origin point.

    Returns
    -------
    datap : array
        Periodic data values.
    """
    if np.isscalar(origin):
        xorigin, yorigin = origin, origin
    else:
        xorigin, yorigin = origin[0], origin[1]
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    datap = buffer_periodic(data, 0, [xorigin, xorigin+xboxsize], buffer_length)
    datap = buffer_periodic(datap, 1, [yorigin, yorigin+yboxsize], buffer_length)
    return datap


def buffer_periodic_3D(data, boxsize, buffer_length, origin=0.):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    data : array
        Where columns 0, 1 and 2 correspond to the x, y and z coordinates.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    origin : float, optional
        Origin point.

    Returns
    -------
    datap : array
        Periodic data values.
    """
    if np.isscalar(origin):
        xorigin, yorigin, zorigin = origin, origin, origin
    else:
        xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    datap = buffer_periodic(data, 0, [xorigin, xorigin+xboxsize], buffer_length)
    datap = buffer_periodic(datap, 1, [yorigin, yorigin+yboxsize], buffer_length)
    datap = buffer_periodic(datap, 2, [zorigin, zorigin+zboxsize], buffer_length)
    return datap
