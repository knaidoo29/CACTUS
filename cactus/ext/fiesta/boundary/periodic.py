import numpy as np


def buffer_periodic(data, axis, boxsize, buffer_length, origin=0.):
    """Returns periodic boundary conditions.

    Parameters
    ----------
    data : array
        Data for which we want periodic boundaries.
    axis : int
        The axis which we want periodic boundaries.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    origin : float, optional
        Origin point.

    Returns
    -------
    datap : array
        Data with periodic buffer particles.
    """
    cond1 = np.where(data[:,axis] >= origin+boxsize-buffer_length)[0]
    _d1 = data[cond1]
    _d1[:,axis] -= boxsize
    cond2 = np.where(data[:,axis] <= origin+buffer_length)[0]
    _d2 = data[cond2]
    _d2[:,axis] += boxsize
    datap = np.vstack([data, _d1, _d2])
    return datap


def subbox_buffer_periodic(data, axis, boxsize, buffer_length, subboxsize,
    origin=0., subbox_origin=0.):
    """Returns periodic boundary conditions in a subbox.

    Parameters
    ----------
    data : array
        Data for which we want periodic boundaries.
    axis : int
        The axis which we want periodic boundaries.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    subboxsize : float
        Size of subbox.
    origin : float, optional
        Origin point.
    subbox_origin : float, optional
        Origin point for the subbox.

    Returns
    -------
    datap : array
        Data with periodic buffer particles.
    """
    datap = buffer_periodic(data, axis, boxsize, buffer_length, origin=origin)
    cond = np.where((datap[:,axis] >= subbox_origin-buffer_length) &
        (datap[:,axis] <= subbox_origin+subboxsize+buffer_length))[0]
    datap = datap[cond]
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
    datap = buffer_periodic(data, 0, xboxsize, buffer_length, origin=xorigin)
    datap = buffer_periodic(datap, 1, yboxsize, buffer_length, origin=yorigin)
    return datap


def subbox_buffer_periodic_2D(data, boxsize, buffer_length, subboxsize,
    origin=0., subbox_origin=0.):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Data for which we want periodic boundaries.
    axis : int
        The axis which we want periodic boundaries.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    subboxsize : float
        Size of subbox.
    origin : float, optional
        Origin point.
    subbox_origin : float, optional
        Origin point for the subbox.

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
    if np.isscalar(subbox_origin):
        subbox_xorigin, subbox_yorigin = subbox_origin, subbox_origin
    else:
        subbox_xorigin, subbox_yorigin = subbox_origin[0], subbox_origin[1]
    if np.isscalar(subboxsize):
        xsubboxsize, ysubboxsize = subboxsize, subboxsize
    else:
        xsubboxsize, ysubboxsize = subboxsize[0], subboxsize[1]
    datap = subbox_buffer_periodic(data, 0, xboxsize, buffer_length,
        xsubboxsize, origin=xorigin, subbox_origin=subbox_xorigin)
    datap = subbox_buffer_periodic(datap, 1, yboxsize, buffer_length,
        ysubboxsize, origin=yorigin, subbox_origin=subbox_yorigin)
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
    datap = buffer_periodic(data, 0, xboxsize, buffer_length, origin=xorigin)
    datap = buffer_periodic(datap, 1, yboxsize, buffer_length, origin=yorigin)
    datap = buffer_periodic(datap, 2, zboxsize, buffer_length, origin=zorigin)
    return datap


def subbox_buffer_periodic_3D(data, boxsize, buffer_length, subboxsize,
    origin=0., subbox_origin=0.):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    data : array
        Data for which we want periodic boundaries.
    axis : int
        The axis which we want periodic boundaries.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    subboxsize : float
        Size of subbox.
    origin : float, optional
        Origin point.
    subbox_origin : float, optional
        Origin point for the subbox.

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
    if np.isscalar(subbox_origin):
        subbox_xorigin, subbox_yorigin, subbox_zorigin = \
            subbox_origin, subbox_origin, subbox_origin
    else:
        subbox_xorigin, subbox_yorigin, subbox_zorigin = \
            subbox_origin[0], subbox_origin[1], subbox_origin[2]
    if np.isscalar(subboxsize):
        xsubboxsize, ysubboxsize, zsubboxsize = \
            subboxsize, subboxsize, subboxsize
    else:
        xsubboxsize, ysubboxsize, zsubboxsize = \
            subboxsize[0], subboxsize[1], subboxsize[2]
    datap = subbox_buffer_periodic(data, 0, xboxsize, buffer_length,
        xsubboxsize, origin=xorigin, subbox_origin=subbox_xorigin)
    datap = subbox_buffer_periodic(datap, 1, yboxsize, buffer_length,
        ysubboxsize, origin=yorigin, subbox_origin=subbox_yorigin)
    datap = subbox_buffer_periodic(datap, 2, zboxsize, buffer_length,
        zsubboxsize, origin=zorigin, subbox_origin=subbox_zorigin)
    return datap
