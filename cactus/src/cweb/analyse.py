import numpy as np

from .. import density
from .. import groups


def get_vol_fraction(cweb):
    """Get the volume fraction for each cosmic web environment.

    Parameters
    ----------
    cweb : array
        Integer array with 0 = voids, 1 = walls, 2 = filaments and 3 = clusters.

    Returns
    -------
    vol_frac : array
        Volume fraction for each cosmic web environment.
    """
    _cweb = cweb.flatten()
    Ntotal = len(_cweb)
    cond = np.where(_cweb == 0.)[0]
    Nvoids = len(cond)
    cond = np.where(_cweb == 1.)[0]
    Nwalls = len(cond)
    cond = np.where(_cweb == 2.)[0]
    Nfilam = len(cond)
    cond = np.where(_cweb == 3.)[0]
    Nclust = len(cond)
    vol_frac = np.array([Nvoids, Nwalls, Nfilam, Nclust])/Ntotal
    return vol_frac


def get_mass_fraction(cweb, mass):
    """Get the mass fraction for each cosmic web environment.

    Parameters
    ----------
    cweb : array
        Integer array with 0 = voids, 1 = walls, 2 = filaments and 3 = clusters.
    mass : array
        Mass contained in each cell.

    Returns
    -------
    mass_frac : array
        Mass fraction for each cosmic web environment.
    """
    _cweb = cweb.flatten()
    _mass = mass.flatten()
    Mtotal = np.sum(mass)
    cond = np.where(_cweb == 0.)[0]
    Mvoids = np.sum(_mass[cond])
    cond = np.where(_cweb == 1.)[0]
    Mwalls = np.sum(_mass[cond])
    cond = np.where(_cweb == 2.)[0]
    Mfilam = np.sum(_mass[cond])
    cond = np.where(_cweb == 3.)[0]
    Mclust = np.sum(_mass[cond])
    mass_frac = np.array([Mvoids, Mwalls, Mfilam, Mclust])/Mtotal
    return mass_frac


def get_cweb_group_info(whichweb, cweb, dens, Omega_m, boxsize):
    """Group a specific cosmic web environment and output the groups number of
    cells, mass and average density (in units of the critical density).

    Parameters
    ----------
    whichweb : int
        Which cosmic web environment, either
    cweb : 3darray
        Cosmic web classification.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Boxsize of the grid.

    Returns
    -------
    group_npix : array
        Number of pixels in each group.
    group_vol : array
        Volume of each group.
    group_mass : array
        Mass of each group.
    group_dens : array
        Average density of each group.
    """
    ngrid = len(cweb)

    mass = density.dens2mass(dens, Omega_m, boxsize)

    dV = (boxsize/ngrid)**3.
    binmap = np.zeros(np.shape(cweb))

    cond = np.where(cweb == whichweb)
    binmap[cond] = 1.

    groupID = groups.groupfinder(binmap)

    group_npix = groups.get_ngroup(groupID)
    group_vol  = group_npix*dV
    group_mass = groups.sum4group(groupID, mass)
    group_dens = group_mass/group_vol

    avgmass = density.average_mass_per_cell(Omega_m, boxsize, ngrid)
    avgdens = avgmass/dV

    group_dens /= avgdens

    return group_npix, group_vol, group_mass, group_dens
