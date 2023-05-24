import numpy as np

from .. import randoms


def buffer_random_2D(npart, boxsize, buffer_length, origin=0.):
    """Generates random buffer particles around a 2D box.

    Parameters
    ----------
    npart : int
        Number of particles in the 2D box.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    origin : float, optional
        Origin point.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    """
    if np.isscalar(origin):
        xorigin, yorigin = origin, origin
    else:
        xorigin, yorigin = origin[0], origin[1]
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    # Determine number of particles needed in the padded region to maintain
    # the average density.
    part_dens = npart / (xboxsize*yboxsize)
    # box edge 1
    _xmin = xorigin - buffer_length
    _xmax = xorigin + boxsize
    _ymin = yorigin - buffer_length
    _ymax = yorigin
    size1 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x1, y1 = randoms.random_box(size1, _xmin, _xmax, _ymin, _ymax)
    # box edge 2
    _xmin = xorigin + xboxsize
    _xmax = xorigin + xboxsize + buffer_length
    _ymin = yorigin - buffer_length
    _ymax = yorigin + yboxsize
    size2 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x2, y2 = randoms.random_box(size2, _xmin, _xmax, _ymin, _ymax)
    # box edge 3
    _xmin = xorigin
    _xmax = xorigin + xboxsize + buffer_length
    _ymin = yorigin + yboxsize
    _ymax = yorigin + yboxsize + buffer_length
    size3 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x3, y3 = randoms.random_box(size3, _xmin, _xmax, _ymin, _ymax)
    # box edge 4
    _xmin = xorigin - buffer_length
    _xmax = xorigin
    _ymin = yorigin
    _ymax = yorigin + yboxsize + buffer_length
    size4 = int(part_dens*(_xmax-_xmin)*(_ymax-_ymin))
    x4, y4 = randoms.random_box(size4, _xmin, _xmax, _ymin, _ymax)
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    return x, y


def buffer_random_3D(npart, boxsize, buffer_length, origin=0.):
    """Generates random buffer particles around a 3D box.

    Parameters
    ----------
    npart : int
        Number of particles in the 3D box.
    boxsize : float
        Box size.
    buffer_length : float
        Length of the buffer region.
    origin : float, optional
        Origin point.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    z : array
        Random z-values.
    """
    if np.isscalar(origin):
        xorigin, yorigin, zorigin = origin, origin, origin
    else:
        xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
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
    xmin, xmax = xorigin-buffer_length, xorigin+boxsize+buffer_length
    ymin, ymax = yorigin-buffer_length, yorigin+boxsize+buffer_length
    zmin, zmax = zorigin-buffer_length, zorigin
    npart1 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x1, y1, z1 = randoms.random_cube(npart1, xmin, xmax, ymin, ymax, zmin, zmax)
    # Region 3
    xmin, xmax = xorigin-buffer_length, xorigin+boxsize+buffer_length
    ymin, ymax = yorigin-buffer_length, yorigin+boxsize+buffer_length
    zmin, zmax = zorigin+boxsize, zorigin+boxsize+buffer_length
    npart3 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x3, y3, z3 = randoms.random_cube(npart3, xmin, xmax, ymin, ymax, zmin, zmax)
    # Subregion 21
    xmin, xmax = xorigin-buffer_length, xorigin+boxsize
    ymin, ymax = yorigin-buffer_length, yorigin
    zmin, zmax = zorigin, zorigin+boxsize
    npart21 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x21, y21, z21 = randoms.random_cube(npart21, xmin, xmax, ymin, ymax, zmin, zmax)
    # Subregion 22
    xmin, xmax = xorigin+boxsize, xorigin+boxsize+buffer_length
    ymin, ymax = yorigin-buffer_length, yorigin+boxsize
    zmin, zmax = zorigin, zorigin+boxsize
    npart22 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x22, y22, z22 = randoms.random_cube(npart22, xmin, xmax, ymin, ymax, zmin, zmax)
    # Subregion 23
    xmin, xmax = xorigin, xorigin+boxsize+buffer_length
    ymin, ymax = yorigin+boxsize, yorigin+boxsize+buffer_length
    zmin, zmax = zorigin, zorigin+boxsize
    npart23 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x23, y23, z23 = randoms.random_cube(npart23, xmin, xmax, ymin, ymax, zmin, zmax)
    # Subregion 24
    xmin, xmax = xorigin-buffer_length, xorigin
    ymin, ymax = yorigin, yorigin+boxsize+buffer_length
    zmin, zmax = zorigin, zorigin+boxsize
    npart24 = int(part_dens*(xmax-xmin)*(ymax-ymin)*(zmax-zmin))
    x24, y24, z24 = randoms.random_cube(npart24, xmin, xmax, ymin, ymax, zmin, zmax)
    x = np.concatenate([x1, x21, x22, x23, x24, x3])
    y = np.concatenate([y1, y21, y22, y23, y24, y3])
    z = np.concatenate([z1, z21, z22, z23, z24, z3])
    return x, y, z
