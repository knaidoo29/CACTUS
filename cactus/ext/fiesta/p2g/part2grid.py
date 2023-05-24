import numpy as np
from .. import src


def part2grid2D(x, y, f, boxsize, ngrid, method='TSC', periodic=True, origin=0.):
    """Returns the density contrast for the nearest grid point grid assignment.

    Parameters
    ----------
    x : array
        X coordinates of the particle.
    y : array
        Y coordinates of the particle.
    f : array
        Value of each particle to be assigned to the grid.
    boxsize : float
        Box size.
    ngrid : int
        Grid divisions across one axis.
    method : str, optional
        Grid assignment scheme, either 'NGP', 'CIC' or 'TSC'.
    periodic : bool, optional
        Assign particles with periodic boundaries.
    origin : float, optional
        Origin.

    Returns
    -------
    fgrid : array
        Grid assigned values.
    """
    if np.isscalar(boxsize):
        xlength, ylength = boxsize, boxsize
    else:
        xlength, ylength = boxsize[0], boxsize[1]
    if np.isscalar(origin):
        xmin = origin
        ymin = origin
    else:
        xmin, ymin = origin[0], origin[1]
    if np.isscalar(ngrid):
        nxgrid, nygrid = ngrid, ngrid
    else:
        nxgrid, nygrid = ngrid[0], ngrid[1]
    if np.isscalar(periodic):
        periodx = periodic
        periody = periodic
    else:
        periodx, periody = periodic[0], periodic[1]
    if method == 'NGP':
        fgrid = src.part2grid_ngp_2d(x=x, y=y, f=f, xlength=xlength, ylength=ylength,
                                     xmin=xmin, ymin=ymin, npart=len(x), nxgrid=nxgrid, nygrid=nygrid)
    elif method == 'CIC':
        fgrid = src.part2grid_cic_2d(x=x, y=y, f=f, xlength=xlength, ylength=ylength,
                                     xmin=xmin, ymin=ymin, npart=len(x), nxgrid=nxgrid,
                                     nygrid=nygrid, periodx=periodx, periody=periody)
    elif method == 'TSC':
        fgrid = src.part2grid_tsc_2d(x=x, y=y, f=f, xlength=xlength, ylength=ylength,
                                     xmin=xmin, ymin=ymin, npart=len(x), nxgrid=nxgrid,
                                     nygrid=nygrid, periodx=periodx, periody=periody)
    return fgrid.reshape(nxgrid, nygrid)


def part2grid3D(x, y, z, f, boxsize, ngrid, method='TSC', periodic=True, origin=0.):
    """Returns the density contrast for the nearest grid point grid assignment.

    Parameters
    ----------
    x : array
        X coordinates of the particle.
    y : array
        Y coordinates of the particle.
    z : array
        Z coordinates of the particle.
    f : array
        Value of each particle to be assigned to the grid.
    boxsize : float or list
        Box size.
    ngrid : int or list
        Grid divisions across one axis.
    method : str, optional
        Grid assignment scheme, either 'NGP', 'CIC' or 'TSC'.
    periodic : bool, optional
        Assign particles with periodic boundaries.
    origin : float, optional
        Origin.

    Returns
    -------
    fgrid : array
        Grid assigned values.
    """
    if np.isscalar(boxsize):
        xlength, ylength, zlength = boxsize, boxsize, boxsize
    else:
        xlength, ylength, zlength = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(origin):
        xmin = origin
        ymin = origin
        zmin = origin
    else:
        xmin, ymin, zmin = origin[0], origin[1], origin[2]
    if np.isscalar(ngrid):
        nxgrid, nygrid, nzgrid = ngrid, ngrid, ngrid
    else:
        nxgrid, nygrid, nzgrid = ngrid[0], ngrid[1], ngrid[2]
    if np.isscalar(periodic):
        periodx = periodic
        periody = periodic
        periodz = periodic
    else:
        periodx, periody, periodz = periodic[0], periodic[1], periodic[2]
    if method == 'NGP':
        fgrid = src.part2grid_ngp_3d(x=x, y=y, z=z, f=f, xlength=xlength,
                                     ylength=ylength, zlength=zlength, xmin=0.,
                                     ymin=0., zmin=0., npart=len(x),
                                     nxgrid=nxgrid, nygrid=nygrid, nzgrid=nzgrid)
    elif method == 'CIC':
        fgrid = src.part2grid_cic_3d(x=x, y=y, z=z, f=f, xlength=xlength,
                                     ylength=ylength, zlength=zlength,
                                     xmin=xmin, ymin=ymin, zmin=zmin, npart=len(x),
                                     nxgrid=nxgrid, nygrid=nygrid, nzgrid=nzgrid,
                                     periodx=periodx, periody=periody, periodz=periodz)
    elif method == 'TSC':
        fgrid = src.part2grid_tsc_3d(x=x, y=y, z=z, f=f, xlength=xlength,
                                     ylength=ylength, zlength=zlength,
                                     xmin=xmin, ymin=ymin, zmin=zmin, npart=len(x),
                                     nxgrid=nxgrid, nygrid=nygrid, nzgrid=nzgrid,
                                     periodx=periodx, periody=periody, periodz=periodz)
    return fgrid.reshape(nxgrid, nygrid, nzgrid)
