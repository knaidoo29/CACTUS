import numpy as np

from ... import shift

from . import dtfe2d
from . import dtfe3d


def mean_separation_2D(npart, boxsize):
    """Calculates the mean separation for particles inside a 2D box.

    Parameters
    ----------
    npart : int
        Number of particles.
    boxsize : float or list
        Dimensions of the box.

    Returns
    -------
    mean_sep : float
        Mean separation of particles.
    """
    if np.isscalar(boxsize):
        area = boxsize**2.
    else:
        area = boxsize[0]*boxsize[1]
    mean_sep = np.sqrt(area/npart)
    return mean_sep


def mean_separation_3D(npart, boxsize):
    """Calculates the mean separation for particles inside a 3D box.

    Parameters
    ----------
    npart : int
        Number of particles.
    boxsize : float or list
        Dimensions of the box.

    Returns
    -------
    mean_sep : float
        Mean separation of particles.
    """
    if np.isscalar(boxsize):
        vol = boxsize**3.
    else:
        vol = boxsize[0]*boxsize[1]*boxsize[2]
    mean_sep = (vol/npart)**(1./3.)
    return mean_sep


def dtfe4grid2D(x, y, ngrid, boxsize, f=None, mass=None, origin=0., buffer_type=None,
                buffer_length=0., buffer_val=0., buffer_mass=None, subsampling=4,
                outputgrid=False, calcdens=True):
    """Returns the Delaunay tesselation density or field on a grid.

    Parameters
    ----------
    x, y : array
        Coordinates of particles.
    ngrid : int or int list
        Grid dimensions.
    boxsize : float or list
        Dimensions of the grid.
    f : array, optional
        Field values, if None assumed output is density.
    mass : array, optional
        Mass of the particles.
    origin : float or list, optional
        Origin for grid.
    buffer_type : str, optional
        Buffer particle type, either:
            - 'random' for random buffer particles.
            - 'periodic' for periodic buffer particles.
            - None for no buffer particles.
    buffer_length : float, optional
        Buffer length.
    buffer_val : float, optional
        Value given to random buffer particles.
    buffer_mass : float, optional
        Must be provided if mass is provided and buffer particles are provided.
    subsampling : int, optional
        The pixel subsampling rate. Each pixel is evaluated subsampling^2 points
        on a grid within each pixel. This is to ensure each pixel is assigned a
        mean pixel value and not the value at the center.
    outputgrid : bool, optional
        Outputs coordinate grid.
    calcdens : bool, optional
        Calculates density.

    Returns
    -------
    f2d : ndarray
        Field values on a grid.
    x2d, y2d : ndarray, optional
        Pixel coordinate points.
    """
    # define boxsize on each axis
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    # define grid on each axis
    if np.isscalar(ngrid):
        nxgrid, nygrid = ngrid, ngrid
    else:
        nxgrid, nygrid = ngrid[0], ngrid[1]
    # define subsampling rate for each pixel across each axis
    if np.isscalar(subsampling):
        dnxgrid, dnygrid = subsampling, subsampling
    else:
        dnxgrid, dnygrid = subsampling[0], subsampling[1]
    # define grid coordinates
    x2d, y2d = shift.cart.grid2D(boxsize, ngrid, origin=origin)
    x2d, y2d = x2d.flatten(), y2d.flatten()
    # define pixel length across each axis
    dx = xboxsize/nxgrid
    dy = yboxsize/nygrid
    # define subsampling points for each pixel
    dx2d, dy2d = shift.cart.grid2D([dx, dy], [dnxgrid, dnygrid], origin=[-dx/2., -dy/2.])
    dx2d, dy2d = dx2d.flatten(), dy2d.flatten()
    # initialise Delaunay tesselation
    D2D = dtfe2d.Delaunay2D()
    # add points
    if mass is None:
        if f is None:
            D2D.set_points(x, y, np.ones(len(x)))
        else:
            D2D.set_points(x, y, f)
    else:
        if f is None:
            D2D.set_points(x, y, np.ones(len(x)), mass=mass)
        else:
            D2D.set_points(x, y, f, mass=mass)
    # set boundary buffer points, either periodic or random buffer points
    if buffer_type == 'periodic':
        D2D.set_periodic(boxsize, buffer_length)
    elif buffer_type == 'random':
        if mass is None:
            D2D.set_buffer(boxsize, buffer_length, buffer_val=buffer_val)
        else:
            if buffer_mass is not None:
                D2D.set_buffer(boxsize, buffer_length, buffer_val=buffer_val, buffer_mass=buffer_mass)
            else:
                assert False, "buffer_mass must be defined."
    # construct delaunay tesselation triangles
    D2D.construct()
    # calculate delaunay tesselation field, if f is None we compute the density
    if calcdens:
        D2D.get_dens()
        D2D.set_field(f=D2D.points_dens)
    else:
        D2D.set_field()
    # calculate the field at the grid points
    for i in range(0, len(dx2d)):
        _f2d = (1./len(dx2d))*D2D.estimate(x2d+dx2d[i], y2d+dy2d[i]).reshape(nxgrid, nygrid)
        if i == 0:
            f2d = _f2d
        else:
            f2d += _f2d
    # ouput the field on a grid
    if outputgrid:
        return x2d.reshape(nxgrid, nygrid), y2d.reshape(nxgrid, nygrid), f2d.reshape(nxgrid, nygrid)
    else:
        return f2d.reshape(nxgrid, nygrid)


def dtfe4grid3D(x, y, z, ngrid, boxsize, f=None, mass=None, origin=0., buffer_length=0.,
                buffer_val=0., buffer_mass=None, buffer_type=None, subsampling=4, useperiodic=False,
                outputgrid=False, calcdens=True):
    """Returns the Delaunay tesselation density or field on a grid.

    Parameters
    ----------
    x, y, z : array
        Coordinates of particles.
    ngrid : int or int list
        Grid dimensions.
    boxsize : float or list
        Dimensions of the grid.
    f : array, optional
        Field values, if None assumed output is density.
    mass : array, optional
        Mass of the particles.
    origin : float or list, optional
        Origin for grid.
    buffer_type : str, optional
        Buffer particle type, either:
            - 'random' for random buffer particles.
            - 'periodic' for periodic buffer particles.
            - None for no buffer particles.
    buffer_length : float, optional
        Buffer length.
    buffer_val : float, optional
        Value given to random buffer particles.
    buffer_mass : float, optional
        Must be provided if mass is provided and buffer particles are provided.
    subsampling : int, optional
        The pixel subsampling rate. Each pixel is evaluated subsampling^2 points
        on a grid within each pixel. This is to ensure each pixel is assigned a
        mean pixel value and not the value at the center.
    outputgrid : bool, optional
        Outputs coordinate grid.
    calcdens : bool, optional
        Calculates density.

    Returns
    -------
    f3d : ndarray
        Field values on a grid.
    x3d, y3d, z3d : ndarray, optional
        Pixel coordinate points.
    """
    # define boxsize on each axis
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    # define grid on each axis
    if np.isscalar(ngrid):
        nxgrid, nygrid, nzgrid = ngrid, ngrid, ngrid
    else:
        nxgrid, nygrid, nzgrid = ngrid[0], ngrid[1], ngrid[2]
    # define subsampling rate for each pixel across each axis
    if np.isscalar(subsampling):
        dnxgrid, dnygrid, dnzgrid = subsampling, subsampling, subsampling
    else:
        dnxgrid, dnygrid, dnzgrid = subsampling[0], subsampling[1], subsampling[2]
    # define grid coordinates
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid, origin=origin)
    x3d, y3d, z3d = x3d.flatten(), y3d.flatten(), z3d.flatten()
    # define pixel length across each axis
    dx = xboxsize/nxgrid
    dy = yboxsize/nygrid
    dz = zboxsize/nzgrid
    # define subsampling points for each pixel
    dx3d, dy3d, dz3d = shift.cart.grid3D([dx, dy, dz], [dnxgrid, dnygrid, dnzgrid], origin=[-dx/2., -dy/2., -dz/2.])
    dx3d, dy3d, dz3d = dx3d.flatten(), dy3d.flatten(), dz3d.flatten()
    # initialise Delaunay tesselation
    D3D = dtfe3d.Delaunay3D()
    # add points
    if mass is None:
        if f is None:
            D3D.set_points(x, y, z, np.ones(len(x)))
        else:
            D3D.set_points(x, y, z, f)
    else:
        if f is None:
            D3D.set_points(x, y, z, np.ones(len(x)), mass=mass)
        else:
            D3D.set_points(x, y, z, f, mass=mass)
    # set boundary buffer points, either periodic or random buffer points
    if buffer_type == 'periodic':
        D3D.set_periodic(boxsize, buffer_length)
    elif buffer_type == 'random':
        if mass is None:
            D3D.set_buffer(boxsize, buffer_length, buffer_val=buffer_val)
        else:
            if buffer_mass is not None:
                D3D.set_buffer(boxsize, buffer_length, buffer_val=buffer_val, buffer_mass=buffer_mass)
            else:
                assert False, "buffer_mass must be defined."
    # construct delaunay tesselation triangles
    D3D.construct()
    # calculate delaunay tesselation field, if f is None we compute the density
    if calcdens:
        D3D.get_dens()
        D3D.set_field(f=D3D.points_dens)
    else:
        D3D.set_field()
    # calculate the field at the grid points
    for i in range(0, len(dx3d)):
        _f3d = (1./len(dx3d))*D3D.estimate(x3d+dx3d[i], y3d+dy3d[i], z3d+dz3d[i])
        if i == 0:
            f3d = _f3d
        else:
            f3d += _f3d
    # output the field on a grid
    if outputgrid:
        return x3d.reshape(nxgrid, nygrid, nzgrid), y3d.reshape(nxgrid, nygrid, nzgrid), f3d.reshape(nxgrid, nygrid, nzgrid)
    else:
        return f3d.reshape(nxgrid, nygrid, nzgrid)
