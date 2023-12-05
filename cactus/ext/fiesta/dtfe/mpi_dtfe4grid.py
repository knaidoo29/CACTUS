import subprocess
import numpy as np

from ... import shift

from .. import coords
from .. import boundary

from . import dtfe4grid


def mpi_dtfe4grid2D(x, y, ngrid, boxsize, MPI, MPI_split, f=None, mass=None,
    buffer_type=None, buffer_length=0., buffer_val=0., buffer_mass=None,
    origin=0., subsampling=4, outputgrid=False, calcdens=True, verbose=False,
    verbose_prefix=""):
    """Returns the Delaunay tesselation density or field on a grid.

    Parameters
    ----------
    x, y : array
        Coordinates of particles.
    ngrid : int or int list
        Grid dimensions.
    boxsize : float or list
        Dimensions of the grid.
    MPI : class object
        MPIutils MPI class object.
    MPI_split : int or int list
        Determines how to split each axis for serial DTFE calculations.
    f : array, optional
        Field values, if None assumed output is density.
    mass : array, optional
        Particle mass values.
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
        Buffer particle mass, if mass for particles is assigned.
    subsampling : int, optional
        The pixel subsampling rate. Each pixel is evaluated subsampling^2 points
        on a grid within each pixel. This is to ensure each pixel is assigned a
        mean pixel value and not the value at the center.
    outputgrid : bool, optional
        Outputs coordinate grid.
    calcdens : bool, optional
        Calculate density.
    verbose : bool, optional
        If True prints out statements
    verbose_prefix : str, optional
        Prefix for print statement.

    Returns
    -------
    f2D : ndarray
        Field values on a grid.
    x2D, y2D : ndarray, optional
        Pixel coordinate points.
    """
    # check basic inputs
    if np.isscalar(ngrid):
        nxgrid, nygrid = ngrid, ngrid
    else:
        nxgrid, nygrid = ngrid[0], ngrid[1]
    if np.isscalar(boxsize):
        xboxsize, yboxsize = boxsize, boxsize
    else:
        xboxsize, yboxsize = boxsize[0], boxsize[1]
    if np.isscalar(origin):
        xorigin, yorigin = origin, origin
    else:
        xorigin, yorigin = origin[0], origin[1]
    # check buffer length
    if buffer_type is not None:
        buffer_length = buffer_length
    else:
        buffer_length = 0.
    # collapse data
    if x is not None:
        if mass is None:
            if f is None:
                data = coords.coord2points([x, y, np.ones(len(f))])
            else:
                data = coords.coord2points([x, y, f])
        else:
            if f is None:
                data = coords.coord2points([x, y, np.ones(len(f)), mass])
            else:
                data = coords.coord2points([x, y, f, mass])
    else:
        data = None
    # check size of data and calculate densities
    size = MPI.collect(len(data), outlist=True)
    if MPI.rank == 0:
        npart = np.sum(np.array(size))
        MPI.send(npart, tag=11)
    else:
        npart = MPI.recv(0, tag=11)
    MPI.wait()
    # sort coordinates and distribute by coordinate system
    MPI_SBX = coords.MPI_SortByX(MPI)
    MPI_SBX.settings(xboxsize, nxgrid, origin=xorigin)
    MPI_SBX.input(data)
    MPI_SBX.limits4grid()
    data = MPI_SBX.distribute()
    limits = [MPI_SBX.limits[0], MPI_SBX.limits[1], yorigin, yorigin+yboxsize]
    # add buffer particles.
    if buffer_type == 'random':
        _data = boundary.mpi_buffer_random_utils(data, limits, buffer_length, MPI)
        xr, yr = boundary.mpi_buffer_random_2D(npart, boxsize, limits, buffer_length, MPI)
        _x, _y = _data[:,0], _data[:,1]
        _x = np.concatenate([_x, xr])
        _y = np.concatenate([_y, yr])
        _f = np.concatenate([_data[:,2], np.ones(npart)*buffer_val])
        if len(_data[0]) == 4:
            _m = np.concatenate([_data[:,3], np.ones(npart)*buffer_mass])
    elif buffer_type == 'periodic':
        _data = boundary.mpi_buffer_periodic_2D(data, boxsize, buffer_length, MPI, origin=origin)
        _x, _y, _f = _data[:,0], _data[:,1], _data[:,2]
        if len(_data[0]) == 4:
            _m = _data[:,3]
    else:
        _x, _y, _f = data[:,0], data[:,1], data[:,2]
        if len(data[0]) == 4:
            _m = data[:,3]
    # create grids
    xedges, xgrid = shift.cart.mpi_grid1D(xboxsize, nxgrid, MPI, origin=xorigin)
    yedges, ygrid = shift.cart.grid1D(yboxsize, nygrid, origin=yorigin)
    x2D, y2D = shift.cart.mpi_grid2D(boxsize, ngrid, MPI, origin=origin)
    shape = np.shape(x2D)
    nxgrid, nygrid = shape[0], shape[1]
    f2D = np.zeros((nxgrid, nygrid))
    if np.isscalar(MPI_split):
        xs1, xs2 = 0, len(xgrid)
        ys1, ys2 = MPI.split(len(ygrid), size=MPI_split)
    else:
        xs1, xs2 = MPI.split(len(xgrid), size=MPI_split[0])
        ys1, ys2 = MPI.split(len(ygrid), size=MPI_split[1])
    xs1, ys1 = np.meshgrid(xs1, ys1, indexing='ij')
    xs1, ys1 = xs1.flatten(), ys1.flatten()
    xs2, ys2 = np.meshgrid(xs2, ys2, indexing='ij')
    xs2, ys2 = xs2.flatten(), ys2.flatten()
    # calculate dtfe
    for i in range(0, len(xs1)):
        _xmin, _xmax = xedges[xs1[i]], xedges[xs2[i]]
        _ymin, _ymax = yedges[ys1[i]], yedges[ys2[i]]
        _nxgrid, _nygrid = xs2[i]-xs1[i], ys2[i]-ys1[i]
        cond = np.where((_x >= _xmin-buffer_length) &
                        (_x < _xmax+buffer_length) &
                        (_y >= _ymin-buffer_length) &
                        (_y < _ymax+buffer_length))[0]
        _xx, _yy = _x[cond], _y[cond]
        if calcdens:
            _ff = None
            _mm = _m[cond]
        else:
            _ff = _f[cond]
            _mm = None
        _f2d = dtfe4grid.dtfe4grid2D(_xx, _yy, [_nxgrid, _nygrid], [_xmax-_xmin, _ymax-_ymin],
                                     f=_ff, mass=_mm, origin=[_xmin, _ymin], buffer_type=None,
                                     subsampling=subsampling, outputgrid=False, calcdens=calcdens)
        f2D[xs1[i]:xs2[i],ys1[i]:ys2[i]] = _f2d
        if verbose:
            MPI.mpi_print_zero(verbose_prefix+"DTFE subgrid:", "%i/%i" % (i+1,len(xs1)))
    if outputgrid:
        return x2D, y2D, f2D
    else:
        return f2D


def mpi_dtfe4grid3D(x, y, z, ngrid, boxsize, MPI, MPI_split, f=None, mass=None,
    buffer_type=None, buffer_length=0., buffer_val=0., origin=0., subsampling=4,
    outputgrid=False, calcdens=True, flush=True, verbose=False, verbose_prefix=""):
    """Returns the Delaunay tesselation density or field on a grid.

    Parameters
    ----------
    x, y, z : array
        Coordinates of particles.
    ngrid : int or int list
        Grid dimensions.
    boxsize : float or list
        Dimensions of the grid.
    MPI : class object
        MPIutils MPI class object.
    MPI_split : int or int list
        Determines how to split each axis for serial DTFE calculations.
    f : array, optional
        Field values, if None assumed output is density.
    mass : array, optional
        Particle mass values.
    buffer_type : str, optional
        Buffer particle type, either:
            - 'random' for random buffer particles.
            - 'periodic' for periodic buffer particles.
            - None for no buffer particles.
    buffer_length : float, optional
        Buffer length.
    buffer_val : float, optional
        Value given to random buffer particles.
    subsampling : int, optional
        The pixel subsampling rate. Each pixel is evaluated subsampling^2 points
        on a grid within each pixel. This is to ensure each pixel is assigned a
        mean pixel value and not the value at the center.
    outputgrid : bool, optional
        Outputs coordinate grid.
    calcdens : bool, optional
        Calculate density.
    flush : bool, optional
        Temporarily save data and load it up sequentially.
    verbose : bool, optional
        If True prints out statements
    verbose_prefix : str, optional
        Prefix for print statement.

    Returns
    -------
    f3D : ndarray
        Field values on a grid.
    x3D, y3D, z3D : ndarray, optional
        Pixel coordinate points.
    """
    # check basic inputs
    if np.isscalar(ngrid):
        nxgrid, nygrid, nzgrid = ngrid, ngrid, ngrid
    else:
        nxgrid, nygrid, nzgrid = ngrid[0], ngrid[1], ngrid[2]
    if np.isscalar(boxsize):
        xboxsize, yboxsize, zboxsize = boxsize, boxsize, boxsize
    else:
        xboxsize, yboxsize, zboxsize = boxsize[0], boxsize[1], boxsize[2]
    if np.isscalar(origin):
        xorigin, yorigin, zorigin = origin, origin, origin
    else:
        xorigin, yorigin, zorigin = origin[0], origin[1], origin[2]
    # check buffer length
    if buffer_type is not None:
        buffer_length = buffer_length
    else:
        buffer_length = 0.
    # collapse data
    if x is not None:
        if mass is None:
            if f is None:
                data = coords.coord2points([x, y, z, np.ones(len(x))])
            else:
                data = coords.coord2points([x, y, z, f])
        else:
            if f is None:
                data = coords.coord2points([x, y, z, np.ones(len(x)), mass])
            else:
                data = coords.coord2points([x, y, z, f, mass])
    else:
        data = None
    # check size of data and calculate densities
    size = MPI.collect(len(data), outlist=True)
    if MPI.rank == 0:
        npart = np.sum(np.array(size))
        MPI.send(npart, tag=11)
    else:
        npart = MPI.recv(0, tag=11)
    MPI.wait()
    # sort coordinates and distribute by coordinate system
    MPI_SBX = coords.MPI_SortByX(MPI)
    MPI_SBX.settings(xboxsize, nxgrid, origin=xorigin)
    MPI_SBX.input(data)
    MPI_SBX.limits4grid()
    # data = MPI_SBX.distribute()
    limits = [MPI_SBX.limits[0], MPI_SBX.limits[1], yorigin, yorigin+yboxsize, zorigin, zorigin+zboxsize]
    # add buffer particles.
    if buffer_type == 'random':
        _data = boundary.mpi_buffer_random_utils(data, limits, buffer_length, MPI)
        xr, yr, zr = boundary.mpi_buffer_random_3D(npart, boxsize, limits, buffer_length, MPI)
        _x, _y, _z = _data[:,0], _data[:,1], _data[:,2]
        _x = np.concatenate([_x, xr])
        _y = np.concatenate([_y, yr])
        _z = np.concatenate([_z, zr])
        _f = np.concatenate([_data[:,3], np.ones(len(xr))*buffer_val])
        if len(_data[0]) == 5:
            _m = np.concatenate([_data[:,4], np.ones(len(xr))*buffer_mass])
    elif buffer_type == 'periodic':
        _data = boundary.mpi_buffer_periodic_3D(data, boxsize, buffer_length, MPI, origin=origin)
        _x, _y, _z = _data[:,0], _data[:,1], _data[:,2]
        _f = _data[:,3]
        if len(data[0]) == 5:
            _m = _data[:,4]
    else:
        _x, _y, _z = data[:,0], data[:,1], data[:,2]
        _f = data[:,3]
        if len(data[0]) == 5:
            _m = data[:,4]
    # create grids
    xedges, xgrid = shift.cart.mpi_grid1D(xboxsize, nxgrid, MPI, origin=xorigin)
    yedges, ygrid = shift.cart.grid1D(yboxsize, nygrid, origin=yorigin)
    zedges, zgrid = shift.cart.grid1D(zboxsize, nzgrid, origin=zorigin)
    x3D, y3D, z3D = shift.cart.mpi_grid3D(boxsize, ngrid, MPI, origin=origin)
    shape = np.shape(x3D)
    nxgrid, nygrid, nzgrid = shape[0], shape[1], shape[2]
    f3D = np.zeros((nxgrid, nygrid, nzgrid))
    if np.isscalar(MPI_split):
        xs1, xs2 = 0, len(xgrid)
        ys1, ys2 = MPI.split(len(ygrid), size=MPI_split)
        zs1, zs2 = MPI.split(len(zgrid), size=MPI_split)
    else:
        xs1, xs2 = MPI.split(len(xgrid), size=MPI_split[0])
        ys1, ys2 = MPI.split(len(ygrid), size=MPI_split[1])
        zs1, zs2 = MPI.split(len(zgrid), size=MPI_split[2])
    xs1, ys1, zs1 = np.meshgrid(xs1, ys1, zs1, indexing='ij')
    xshape = np.shape(xs1)
    xs1, ys1, zs1 = xs1.flatten(), ys1.flatten(), zs1.flatten()
    xs2, ys2, zs2 = np.meshgrid(xs2, ys2, zs2, indexing='ij')
    xs2, ys2, zs2 = xs2.flatten(), ys2.flatten(), zs2.flatten()
    # calculate dtfe
    if flush:
        ii = np.arange(len(xs1))
        ii = ii.reshape(xshape)
        xs1 = xs1.reshape(xshape)
        ys1 = ys1.reshape(xshape)
        zs1 = zs1.reshape(xshape)
        xs2 = xs2.reshape(xshape)
        ys2 = ys2.reshape(xshape)
        zs2 = zs2.reshape(xshape)
        for i1 in range(0, len(ii)):
            _xmin, _xmax = xedges[xs1[i1, 0, 0]], xedges[xs2[i1, 0, 0]]
            c0 = np.where((_x >= _xmin-buffer_length) & (_x <= _xmax+buffer_length))[0]
            for i2 in range(0, len(ii[i1])):
                _ymin, _ymax = yedges[ys1[0, i2, 0]], yedges[ys2[0, i2, 0]]
                c1 = np.where((_y[c0] >= _ymin-buffer_length) & (_y[c0] <= _ymax+buffer_length))[0]
                for i3 in range(0, len(ii[i1,i2])):
                    _zmin, _zmax = zedges[zs1[0, 0, i3]], zedges[zs2[0, 0, i3]]
                    c2 = np.where((_z[c0[c1]] >= _zmin-buffer_length) & (_z[c0[c1]] <= _zmax+buffer_length))[0]
                    i = ii[i1, i2, i3]
                    _xx, _yy, _zz = _x[c0[c1[c2]]], _y[c0[c1[c2]]], _z[c0[c1[c2]]]
                    if calcdens:
                        _ff = None
                        _mm = _m[c0[c1[c2]]]
                    else:
                        _ff = _f[c0[c1[c2]]]
                        _mm = None
                    np.savez('temp_dtfe_MPI_%i_%i.npz'%(MPI.rank, i),
                        _xx=_xx, _yy=_yy, _zz=_zz, _ff=_ff, _mm=_mm)
                    if verbose:
                        MPI.mpi_print_zero(verbose_prefix+"Saving partitioned particles:", "%i/%i" % (i+1,int(xshape[0]
*xshape[1]*xshape[1])), "Npart=%i" % len(_xx))
        del _x
        del _y
        del _z
        if calcdens:
            del _m
        else:
            del _f
    xs1, ys1, zs1 = xs1.flatten(), ys1.flatten(), zs1.flatten()
    xs2, ys2, zs2 = xs2.flatten(), ys2.flatten(), zs2.flatten()
    for i in range(0, len(xs1)):
        _xmin, _xmax = xedges[xs1[i]], xedges[xs2[i]]
        _ymin, _ymax = yedges[ys1[i]], yedges[ys2[i]]
        _zmin, _zmax = zedges[zs1[i]], zedges[zs2[i]]
        _nxgrid, _nygrid, _nzgrid = xs2[i]-xs1[i], ys2[i]-ys1[i], zs2[i]-zs1[i]
        if flush:
            dat = np.load('temp_dtfe_MPI_%i_%i.npz'%(MPI.rank, i), allow_pickle=True)
            _xx, _yy, _zz, _ff, _mm = dat['_xx'], dat['_yy'], dat['_zz'], dat['_ff'], dat['_mm']
            if calcdens:
                _ff = None
            else:
                _mm = None
        else:
            cond = np.where((_x >= _xmin-buffer_length) &
                            (_x < _xmax+buffer_length) &
                            (_y >= _ymin-buffer_length) &
                            (_y < _ymax+buffer_length) &
                            (_z >= _zmin-buffer_length) &
                            (_z < _zmax+buffer_length))[0]
            _xx, _yy, _zz = _x[cond], _y[cond], _z[cond]
            if calcdens:
                _ff = None
                _mm = _m[cond]
            else:
                _ff = _f[cond]
                _mm = None
        _f3d = dtfe4grid.dtfe4grid3D(_xx, _yy, _zz, [_nxgrid, _nygrid, _nzgrid],
                                     [_xmax-_xmin, _ymax-_ymin, _zmax-_zmin],
                                     f=_ff, mass=_mm, origin=[_xmin, _ymin, _zmin],
                                     buffer_type=None, subsampling=subsampling,
                                     outputgrid=False, calcdens=calcdens)
        f3D[xs1[i]:xs2[i],ys1[i]:ys2[i],zs1[i]:zs2[i]] = _f3d
        if verbose:
            MPI.mpi_print_zero(verbose_prefix+"DTFE subgrid:", "%i/%i" % (i+1,len(xs1)))
    if flush:
        subprocess.call('rm -v temp_dtfe_MPI_%i_*.npz' % MPI.rank, shell=True)
    if outputgrid:
        return x3D, y3D, z3D, f3D
    else:
        return f3D
