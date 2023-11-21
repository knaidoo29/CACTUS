import numpy as np

from . import periodic
from .. import randoms


def mpi_buffer_periodic_2D(data, boxsize, buffer_length, MPI, origin=0.):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Where columns 0 and 1 are x and y.
    boxsize : float
        Boxsize.
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.
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
    datap = periodic.buffer_periodic(data, 1, yboxsize, buffer_length, origin=yorigin)
    cond = np.where(datap[:,0] <= xorigin+buffer_length)[0]
    data_send_down = MPI.send_down(datap[cond])
    cond = np.where(datap[:,0] >= xorigin+xboxsize-buffer_length)[0]
    data_send_up = MPI.send_up(datap[cond])
    if MPI.rank == 0:
        data_send_up[:,0] -= xboxsize
    elif MPI.rank == MPI.size - 1:
        data_send_down[:,0] += xboxsize
    datap = np.vstack([datap, data_send_down, data_send_up])
    return datap



def mpi_buffer_periodic_2D(data, boxsize, buffer_length, subboxsize, MPI,
    origin=0., subbox_origin=0.):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Where columns 0 and 1 are x and y.
    boxsize : float
        Boxsize.
    buffer_length : float
        Length of the buffer region.
    subboxsize : float
        Size of subbox.
    MPI : class object
        MPIutils MPI class object.
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

    datap = periodic.buffer_periodic(data, 1, yboxsize, buffer_length, origin=yorigin)
    cond = np.where(datap[:,0] <= xorigin+buffer_length)[0]
    data_send_down = MPI.send_down(datap[cond])
    cond = np.where(datap[:,0] >= xorigin+xboxsize-buffer_length)[0]
    data_send_up = MPI.send_up(datap[cond])
    if MPI.rank == 0:
        data_send_up[:,0] -= xboxsize
    elif MPI.rank == MPI.size - 1:
        data_send_down[:,0] += xboxsize
    datap = np.vstack([datap, data_send_down, data_send_up])
    return datap


def mpi_buffer_periodic_3D(data, boxsize, buffer_length, MPI, origin=0.):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    data : array
        Where columns 0, 1 and 2 correspond to the x, y and z coordinates.
    xboxsize : float
        Boxsize along x-axis.
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.
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
    datap = periodic.buffer_periodic(data, 1, yboxsize, buffer_length, origin=yorigin)
    datap = periodic.buffer_periodic(datap, 2, zboxsize, buffer_length, origin=zorigin)
    cond = np.where(datap[:,0] <= xorigin+buffer_length)[0]
    data_send_down = MPI.send_down(datap[cond])
    cond = np.where(datap[:,0] >= xorigin+xboxsize-buffer_length)[0]
    data_send_up = MPI.send_up(datap[cond])
    if MPI.rank == 0:
        data_send_up[:,0] -= xboxsize
    elif MPI.rank == MPI.size - 1:
        data_send_down[:,0] += xboxsize
    datap = np.vstack([datap, data_send_down, data_send_up])
    return datap
