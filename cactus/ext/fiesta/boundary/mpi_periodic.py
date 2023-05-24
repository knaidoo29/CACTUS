import numpy as np

from . import periodic
from .. import randoms


def mpi_buffer_periodic_2D(data, xboxsize, limits, buffer_length, MPI):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Where columns 0 and 1 are x and y.
    xboxsize : float
        Boxsize along x-axis.
    limits : list
        Ranges for each coordinate axis [xmin, xmax, ymin, ymax].
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.

    Returns
    -------
    datap : array
        Periodic data values.
    """
    datap = periodic.buffer_periodic(data, 1, [limits[2], limits[3]], buffer_length)
    cond = np.where(datap[:,0] <= limits[0]+buffer_length)[0]
    data_send_down = MPI.send_down(datap[cond])
    cond = np.where(datap[:,0] >= limits[1]-buffer_length)[0]
    data_send_up = MPI.send_up(datap[cond])
    if MPI.rank == 0:
        data_send_up[:,0] -= xboxsize
    elif MPI.rank == MPI.size - 1:
        data_send_down[:,0] += xboxsize
    datap = np.vstack([datap, data_send_down, data_send_up])
    return datap


def mpi_buffer_periodic_3D(data, xboxsize, limits, buffer_length, MPI):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    data : array
        Where columns 0, 1 and 2 correspond to the x, y and z coordinates.
    xboxsize : float
        Boxsize along x-axis.
    limits : list
        Ranges for each coordinate axis [xmin, xmax, ymin, ymax].
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.

    Returns
    -------
    datap : array
        Periodic data values.
    """
    datap = periodic.buffer_periodic(data, 1, [limits[2], limits[3]], buffer_length)
    datap = periodic.buffer_periodic(datap, 2, [limits[4], limits[5]], buffer_length)
    cond = np.where(datap[:,0] <= limits[0]+buffer_length)[0]
    data_send_down = MPI.send_down(datap[cond])
    cond = np.where(datap[:,0] >= limits[1]-buffer_length)[0]
    data_send_up = MPI.send_up(datap[cond])
    if MPI.rank == 0:
        data_send_up[:,0] -= xboxsize
    elif MPI.rank == MPI.size - 1:
        data_send_down[:,0] += xboxsize
    datap = np.vstack([datap, data_send_down, data_send_up])
    return datap
