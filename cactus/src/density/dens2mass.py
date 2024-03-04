import numpy as np


def norm_dens(dens):
    """Normalise density field.

    Parameters
    ----------
    dens : array
        Density field.
    """
    avgdens = np.mean(dens)
    if avgdens != 1.:
        dens /= avgdens
    return dens


def mpi_norm_dens(dens, MPI):
    """Normalise density field.

    Parameters
    ----------
    dens : array
        Density field.
    MPI : object
        MPI object.
    """
    avgdens = MPI.mean(dens)
    if avgdens != 1.:
        dens /= avgdens
    return dens


def average_mass_per_cell(Omega_m, boxsize, ngrid):
    """Get average mass in each cell.

    Parameters
    ----------
    Omega_m : float
        Matter density.
    boxsize : float
        Boxsize.
    ngrid : int
        Grid size along one axis.

    Returns
    -------
    avgmass : float
        Avgmass in a grid cell in units of 10^10 Msun/h.
    """
    Omega_m = Omega_m
    boxsize = boxsize
    ncells = ngrid**3
    G_const = 6.6743e-11
    avgmass = 3.*Omega_m*(boxsize**3.)/(8.*np.pi*G_const*ncells)
    avgmass *= 3.0857e2/1.9891
    avgmass /= 1e10
    return avgmass


def dens2mass(dens, Omega_m, boxsize, ngrid):
    """Convert density field to mass field.

    Parameters
    ----------
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Boxsize.
    ngrid : int
        Grid size along one axis.

    Returns
    -------
    mass : 3darray
        Mass field.
    """
    dens = norm_dens(dens)
    avgmass = average_mass_per_cell(Omega_m, boxsize, ngrid)
    mass = dens*avgmass
    return mass


def mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI):
    """Convert density field to mass field.

    Parameters
    ----------
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Boxsize.
    ngrid : int
        Grid size along one axis.
    MPI : object
        MPI object.

    Returns
    -------
    mass : 3darray
        Mass field.
    """
    dens = mpi_norm_dens(dens, MPI)
    avgmass = average_mass_per_cell(Omega_m, boxsize, ngrid)
    mass = dens*avgmass
    return mass
