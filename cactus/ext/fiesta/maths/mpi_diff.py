import numpy as np

from . import diff


def mpi_dfdx(xgrid, f, MPI, periodic=False):
    """Differentiate using a symmetric two-point derivative and the non-symmetric
    three point derivative estimator.

    Parameters
    ----------
    xgrid : array
        X-axis.
    f : array
        Function values at x.

    Returns
    -------
    dfdx : array
        Numerical differentiation values for f evaluated at points x.
    """
    dx = xgrid[1]-xgrid[0]
    _xgrid = np.zeros(len(xgrid)+2)
    _xgrid[1:-1] = xgrid
    _xgrid[0] = xgrid[0]-dx
    _xgrid[-1] = xgrid[-1]+dx
    shape = np.array(np.shape(f))
    shape[0] += 2
    _f = np.zeros(shape)
    _f[1:-1] = f
    f_send_up = MPI.send_up(f[-1])
    _f[0] = f_send_up
    f_send_down = MPI.send_down(f[0])
    _f[-1] = f_send_down
    if periodic is False:
        if MPI.rank == 0:
            _xgrid = _xgrid[1:]
            _f = _f[1:]
        elif MPI.rank == MPI.size-1:
            _xgrid = _xgrid[:-1]
            _f = _f[:-1]
    dfdx = diff.dfdx(_xgrid, _f, periodic=False)
    if periodic is True:
        dfdx = dfdx[1:-1]
    else:
        if MPI.rank == 0:
            dfdx = dfdx[:-1]
        elif MPI.rank == MPI.size-1:
            dfdx = dfdx[1:]
        else:
            dfdx = dfdx[1:-1]
    return dfdx


def mpi_dfdy(ygrid, f, MPI, periodic=False):
    """Differentiate using a symmetric two-point derivative and the non-symmetric
    three point derivative estimator.

    Parameters
    ----------
    ygrid : array
        Y-axis.
    f : array
        Function values at y.

    Returns
    -------
    dfdy : array
        Numerical differentiation values for f evaluated at points y.
    """
    dfdy = diff.dfdy(ygrid, f, periodic=periodic)
    return dfdy



def mpi_dfdz(zgrid, f, MPI, periodic=False):
    """Differentiate using a symmetric two-point derivative and the non-symmetric
    three point derivative estimator.

    Parameters
    ----------
    zgrid : array
        Z-axis.
    f : array
        Function values at z.

    Returns
    -------
    dfdz : array
        Numerical differentiation values for f evaluated at points z.
    """
    dfdz = diff.dfdz(zgrid, f, periodic=periodic)
    return dfdz
