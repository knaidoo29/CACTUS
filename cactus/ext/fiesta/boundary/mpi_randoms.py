import numpy as np

from .. import randoms


def mpi_buffer_random_2D(npart, boxsize, limits, buffer_length, MPI):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    npart : int
        Number of particles in the 2D box.
    boxsize : float
        Box size.
    limits : list
        Ranges for each coordinate axis [xmin, xmax, ymin, ymax].
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    # Determine number of particles needed in the padded region to maintain
    # the average density.
    part_dens = npart / (xboxsize*yboxsize)
    # box edge 1
    _xmin = limits[0] - buffer_length
    _xmax = limits[1] + buffer_length
    _ymin = limits[2] - buffer_length
    _ymax = limits[2]
    size1 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x1, y1 = randoms.random_box(size1, _xmin, _xmax, _ymin, _ymax)
    # box edge 2
    _xmin = limits[0] - buffer_length
    _xmax = limits[1] + buffer_length
    _ymin = limits[3]
    _ymax = limits[3] + buffer_length
    size2 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x2, y2 = randoms.random_box(size2, _xmin, _xmax, _ymin, _ymax)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    if MPI.rank == 0:
        _xmin = limits[0] - buffer_length
        _xmax = limits[0]
        _ymin = limits[2]
        _ymax = limits[3]
        size = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
        _x, _y = randoms.random_box(size, _xmin, _xmax, _ymin, _ymax)
        x = np.concatenate([x, _x])
        y = np.concatenate([y, _y])
    if MPI.rank == MPI.size-1:
        _xmin = limits[1]
        _xmax = limits[1] + buffer_length
        _ymin = limits[2]
        _ymax = limits[3]
        size = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
        _x, _y = randoms.random_box(size, _xmin, _xmax, _ymin, _ymax)
        x = np.concatenate([x, _x])
        y = np.concatenate([y, _y])
    return x, y


def mpi_buffer_random_3D(npart, boxsize, limits, buffer_length, MPI):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    npart : int
        Number of particles in the 3D box.
    boxsize : float
        Box size.
    limits : list
        Ranges for each coordinate axis [xmin, xmax, ymin, ymax, zmin, zmax].
    buffer_length : float
        Length of the buffer region.
    MPI : class object
        MPIutils MPI class object.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    z : array
        Random z-values.
    """
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    box_vol = xboxsize*yboxsize*zboxsize
    part_dens = npart / box_vol
    # Divide into three regions along z-axis
    # Region 1: zrange = [-buffer_length, 0.]
    # Region 2: zrange = [0., boxsize]
    # Region 3: zrange = [boxsize, boxsize+buffer_length]
    # Splitting region 2, into four regions.
    # Subregion 21: xrange = [-buffer_length, boxsize], yrange = [-buffer_length, 0]
    # Subregion 22: xrange = [boxsize, boxsize+buffer_length], yrange =[-buffer_length, boxsize]
    # Subregion 23: xrange = [0, boxsize+buffer_length], yrange=[boxsize, boxsize+buffer_length]
    # Subregion 24: xrange = [-buffer_length, 0], yrange = [0, boxsize+buffer_length]
    # Region 1
    xmin, xmax = limits[0]-buffer_length, limits[1]+buffer_length
    ymin, ymax = limits[2]-buffer_length, limits[3]+buffer_length
    zmin, zmax = limits[4]-buffer_length, limits[4]
    npart1 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x1, y1, z1 = randoms.random_cube(npart1, xmin, xmax, ymin, ymax, zmin, zmax)
    # Region 3
    xmin, xmax = limits[0]-buffer_length, limits[1]+buffer_length
    ymin, ymax = limits[2]-buffer_length, limits[3]+buffer_length
    zmin, zmax = limits[5], limits[5]+buffer_length
    npart3 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x3, y3, z3 = randoms.random_cube(npart3, xmin, xmax, ymin, ymax, zmin, zmax)
    # x = np.concatenate([x1, x3])
    # y = np.concatenate([y1, y3])
    # z = np.concatenate([z1, z3])
    # Subregion 21
    xmin, xmax = limits[0]-buffer_length, limits[1]+buffer_length
    ymin, ymax = limits[2]-buffer_length, limits[2]
    zmin, zmax = limits[4], limits[5]
    npart21 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x21, y21, z21 = randoms.random_cube(npart21, xmin, xmax, ymin, ymax, zmin, zmax)
    # Subregion 22
    xmin, xmax = limits[0]-buffer_length, limits[1]+buffer_length
    ymin, ymax = limits[3], limits[3]+buffer_length
    zmin, zmax = limits[4], limits[5]
    npart22 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x22, y22, z22 = randoms.random_cube(npart22, xmin, xmax, ymin, ymax, zmin, zmax)
    x = np.concatenate([x1, x21, x22, x3])
    y = np.concatenate([y1, y21, y22, y3])
    z = np.concatenate([z1, z21, z22, z3])
    if MPI.rank == 0:
        # Subregion 23
        xmin, xmax = limits[0]-buffer_length, limits[0]
        ymin, ymax = limits[2], limits[3]
        zmin, zmax = limits[4], limits[5]
        npart23 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
        x23, y23, z23 = randoms.random_cube(npart23, xmin, xmax, ymin, ymax, zmin, zmax)
        x = np.concatenate([x, x23])
        y = np.concatenate([y, y23])
        z = np.concatenate([z, z23])
    elif MPI.rank == MPI.size - 1:
        # Subregion 24
        xmin, xmax = limits[1], limits[1]+buffer_length
        ymin, ymax = limits[2], limits[3]
        zmin, zmax = limits[4], limits[5]
        npart24 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
        x24, y24, z24 = randoms.random_cube(npart24, xmin, xmax, ymin, ymax, zmin, zmax)
        x = np.concatenate([x, x24])
        y = np.concatenate([y, y24])
        z = np.concatenate([z, z24])
    return x, y, z


def mpi_buffer_random_utils(data, limits, buffer_length, MPI):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    data : array
        Where columns 0 and 1 are x and y.
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
    cond = np.where(data[:,0] <= limits[0]+buffer_length)[0]
    data_send_down = MPI.send_down(data[cond])
    cond = np.where(data[:,0] >= limits[1]-buffer_length)[0]
    data_send_up = MPI.send_up(data[cond])
    if MPI.rank == 0:
        datap = np.vstack([data, data_send_down])
    elif MPI.rank == MPI.size - 1:
        datap = np.vstack([data, data_send_up])
    else:
        datap = np.vstack([data, data_send_down, data_send_up])
    return datap
