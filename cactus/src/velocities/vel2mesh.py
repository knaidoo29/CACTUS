import numpy as np

from ...ext import fiesta


def p2g_vel2mesh3D(x, y, z, vx, vy, vz, boxsize, ngrid, method='TSC'):
    """Particle-to-grid velocity computation.

    Parameters
    ----------
    x, y, z : array
        Particle positions.
    vx, vy, vz : array
        Particle velocities.
    boxsize : float or list
        Box size.
    ngrid : int
        Grid divisions across one axis.
    method : str, optional
        Grid assignment scheme, either 'NGP', 'CIC' or 'TSC'.

    Returns
    -------
    vxf, vyf, vzf : 3darray
        Velocity fields.
    """

    rho = fiesta.p2g.part2grid3D(x, y, z, np.ones(len(x)), boxsize, ngrid, method=method)
    vxf = fiesta.p2g.part2grid3D(x, y, z, vx, boxsize, ngrid, method=method)
    vyf = fiesta.p2g.part2grid3D(x, y, z, vy, boxsize, ngrid, method=method)
    vzf = fiesta.p2g.part2grid3D(x, y, z, vz, boxsize, ngrid, method=method)

    vxf /= rho
    vyf /= rho
    vzf /= rho

    return vxf, vyf, vzf


def mpi_p2g_vel2mesh3D(x, y, z, vx, vy, vz, boxsize, ngrid, MPI, method='TSC'):
    """Particle-to-grid velocity computation.

    Parameters
    ----------
    x, y, z : array
        Particle positions.
    vx, vy, vz : array
        Particle velocities.
    boxsize : float or list
        Box size.
    ngrid : int
        Grid divisions across one axis.
    MPI : object
        MPIutils object.
    method : str, optional
        Grid assignment scheme, either 'NGP', 'CIC' or 'TSC'.

    Returns
    -------
    vxf, vyf, vzf : 3darray
        Velocity fields.
    """

    rho = fiesta.p2g.mpi_part2grid3D(x, y, z, np.ones(len(x)), boxsize, ngrid, MPI, method=method)
    vxf = fiesta.p2g.mpi_part2grid3D(x, y, z, vx, boxsize, ngrid, MPI, method=method)
    vyf = fiesta.p2g.mpi_part2grid3D(x, y, z, vy, boxsize, ngrid, MPI, method=method)
    vzf = fiesta.p2g.mpi_part2grid3D(x, y, z, vz, boxsize, ngrid, MPI, method=method)

    vxf /= rho
    vyf /= rho
    vzf /= rho

    return vxf, vyf, vzf


def dtfe_vel2mesh3D(x, y, z, vx, vy, vz, boxsize, ngrid, subsampling=1,
    buffer_type=None, buffer_length=0., verbose=True, verbose_prefix=""):
    """Particle-to-grid velocity computation.

    Parameters
    ----------
    x, y, z : array
        Particle positions.
    vx, vy, vz : array
        Particle velocities.
    boxsize : float or list
        Box size.
    ngrid : int
        Grid divisions across one axis.
    subsampling : int
        Number of points to evaluate at each cell.
    buffer_type : str, optional
        Buffer particle type, either:
            - 'random' for random buffer particles.
            - 'periodic' for periodic buffer particles.
            - None for no buffer particles.
    buffer_length : float, optional
        Buffer length.
    verbose : bool, optional
        If True prints out statements
    verbose_prefix : str, optional
        Prefix for print statement.

    Returns
    -------
    vxf, vyf, vzf : 3darray
        Velocity fields.
    """

    vxf = fiesta.dtfe.dtfe4grid3D(x, y, z, ngrid, boxsize, f=vx,
        buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    vyf = fiesta.dtfe.dtfe4grid3D(x, y, z, ngrid, boxsize, f=vy,
        buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    vzf = fiesta.dtfe.dtfe4grid3D(x, y, z, ngrid, boxsize, f=vz,
        buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    return vxf, vyf, vzf


def mpi_dtfe_vel2mesh3D(x, y, z, vx, vy, vz, boxsize, ngrid, MPI, MPI_split,
    subsampling=1, buffer_type=None, buffer_length=0., verbose=True, verbose_prefix=""):
    """Particle-to-grid velocity computation.

    Parameters
    ----------
    x, y, z : array
        Particle positions.
    vx, vy, vz : array
        Particle velocities.
    boxsize : float or list
        Box size.
    ngrid : int
        Grid divisions across one axis.
    MPI : class object
        MPIutils MPI class object.
    MPI_split : int or int list
        Determines how to split each axis for serial DTFE calculations.
    subsampling : int
        Number of points to evaluate at each cell.
    buffer_type : str, optional
        Buffer particle type, either:
            - 'random' for random buffer particles.
            - 'periodic' for periodic buffer particles.
            - None for no buffer particles.
    buffer_length : float, optional
        Buffer length.
    verbose : bool, optional
        If True prints out statements
    verbose_prefix : str, optional
        Prefix for print statement.

    Returns
    -------
    vxf, vyf, vzf : 3darray
        Velocity fields.
    """

    vxf = fiesta.dtfe.mpi_dtfe4grid3D(x, y, z, ngrid, boxsize, MPI, MPI_split,
        f=vx, buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    vyf = fiesta.dtfe.mpi_dtfe4grid3D(x, y, z, ngrid, boxsize, MPI, MPI_split,
        f=vy, buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    vzf = fiesta.dtfe.mpi_dtfe4grid3D(x, y, z, ngrid, boxsize, MPI, MPI_split,
        f=vz, buffer_type=buffer_type, buffer_length=buffer_length, buffer_val=0.,
        subsampling=sampling, outputgrid=False, verbose=verbose,
        verbose_prefix=verbose_prefix)

    return vxf, vyf, vzf
