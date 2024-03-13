import numpy as np
from scipy.interpolate import interp1d

from ...ext import fiesta, magpie, shift

from .. import density, groups


def get_mass_below_logS(S, mass, nbins=100, mask=None):
    """Returns the mass contained in signatures values above logS.

    Parameters
    ----------
    S : 3darray
        Cosmic web signature.
    mass : 3darray
        Mass field.
    nbins : int, optional
        Number of bins to bin logS in.
    mask : int, optional
        An optional mask to mask out non-zero signature values.

    Returns
    -------
    logS_lim : array
        Thresholds in logS.
    sum_M : array
        Mass contained in signature values above logS_lim.
    """
    if mask is None:
        # find S > 0.
        cond = np.where(S > 0)
    else:
        cond = np.where((S > 0) & (mask == 1.))
    # convert to logS and flatten array.
    logS = np.log10(S[cond].flatten())
    # find min and max
    minlogS, maxlogS = np.min(logS), np.max(logS)
    # separate only mass in S > 0 and flatten array.
    _mass = mass[cond].flatten()
    # find pixels of logS in histogram.
    pixID = magpie.pixels.pos2pix_cart1d(logS, maxlogS-minlogS, nbins, origin=minlogS)
    # bin mass in histogram with mass weightings.
    dM = magpie.pixels.bin_pix(pixID, nbins, weights=_mass)
    # cumulative summation but from top down (hence the reversing)
    sum_M = np.cumsum(dM[::-1])[::-1]
    # get edges of histogram
    logS_edges, _ = shift.cart.grid1D(maxlogS-minlogS, nbins, origin=minlogS)
    # x values are the lower side of the histogram edges.
    logS_lim = logS_edges[:-1]
    return logS_lim, sum_M


def mpi_get_mass_below_logS(S, mass, MPI, nbins=100, mask=None):
    """Returns the mass contained in signatures values above logS.

    Parameters
    ----------
    S : 3darray
        Cosmic web signature.
    mass : 3darray
        Mass field.
    MPI : object
        MPIutils object.
    nbins : int, optional
        Number of bins to bin logS in.
    mask : int, optional
        An optional mask to mask out non-zero signature values.

    Returns
    -------
    logS_lim : array
        Thresholds in logS.
    sum_M : array
        Mass contained in signature values above logS_lim.
    """
    if mask is None:
        # find S > 0.
        cond = np.where(S > 0)
    else:
        cond = np.where((S > 0) & (mask == 1.))
    # convert to logS and flatten array.
    logS = np.log10(S[cond].flatten())
    # find min and max
    minlogS = MPI.min(logS)
    maxlogS = MPI.max(logS)
    # separate only mass in S > 0 and flatten array.
    _mass = mass[cond].flatten()
    # find pixels of logS in histogram.
    pixID = magpie.pixels.pos2pix_cart1d(logS, maxlogS-minlogS, nbins, origin=minlogS)
    # bin mass in histogram with mass weightings.
    dM = magpie.pixels.bin_pix(pixID, nbins, weights=_mass)
    dM = MPI.sum(dM)
    if MPI.rank == 0:
        # cumulative summation but from top down (hence the reversing)
        sum_M = np.cumsum(dM[::-1])[::-1]
        # get edges of histogram
        logS_edges, _ = shift.cart.grid1D(maxlogS-minlogS, nbins, origin=minlogS)
        # x values are the lower side of the histogram edges.
        logS_lim = logS_edges[:-1]
        MPI.send(logS_lim, tag=11)
        MPI.send(sum_M, tag=12)
    else:
        logS_lim = MPI.recv(0, tag=11)
        sum_M = MPI.recv(0, tag=12)
    return logS_lim, sum_M


def get_Sc_group_info(Sc, dens, Omega_m, boxsize, ngrid, minvol, mindens, minmass,
    neval=10, overide_min_sum_M=None, overide_max_sum_M=None, periodic=True,
    verbose=True, prefix=''):
    """Computes group information for different thresholds of the cluster signature.

    Parameters
    ----------
    Sc : 3darray
        Cluster nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    mindens : float
        Minimum density for a group.
    minmass : float
        Minimum mass for a group.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sc_lims : array
        Cluster signature limits.
    Num : array
        Number of valid groups (i.e. with a size larger than the minimum volume).
    Num_mlim : array
        Number of valid groups with a mass larger than minmass.
    Num_dlim : array
        Number of valid groups with a density larger than mindens.
    Num_mlim_dlim : array
        Number of valid groups with a mass larger than minmass and density larger
        than mindens.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    avgmass = density.average_mass_per_cell(Omega_m, boxsize, ngrid)
    avgdens = avgmass/dV

    mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)

    logS_lim, sum_M = get_mass_below_logS(Sc, mass)

    if overide_max_sum_M is None:
        max_sum_M = np.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = np.min(sum_M)
        if min_sum_M < minmass:
            min_sum_M = minmass
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sc_lims = 10.**interp_logS_lim(interp_vals)

    Num = []
    Num_mlim = []
    Num_dlim = []
    Num_mlim_dlim = []

    if verbose:
        print(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["log10(Sc)", "Num(Groups)", "Num(d>dmin)", "Num(M>Mmin)", "Num(M>Mmin,d>dmin)", "Valid Frac."]))
        print(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["-"*16, "-"*16, "-"*16, "-"*16, "-"*20, "-"*16]))

    for Sc_lim in Sc_lims:

        binmap = np.zeros(np.shape(Sc))
        cond = np.where(Sc > Sc_lim)
        binmap[cond] = 1.

        sumbinmap = np.sum(binmap)

        if sumbinmap > 0.:

            groupID = groups.groupfinder(binmap, periodic=periodic)

            group_N = groups.get_ngroup(groupID)
            group_mass = groups.sum4group(groupID, mass)
            # true density rho
            group_dens = group_mass/(dV*group_N)
            # in units of mean density
            group_dens /= avgdens

            cond = np.where((group_N >= minpix))[0]

            Num.append(len(cond))

            group_N = group_N[cond]
            group_mass = group_mass[cond]
            group_dens = group_dens[cond]

            cond = np.where((group_mass >= minmass))[0]
            Num_mlim.append(len(cond))

            cond = np.where((group_dens >= mindens))[0]
            Num_dlim.append(len(cond))

            cond = np.where((group_mass >= minmass) & (group_dens >= mindens))[0]
            Num_mlim_dlim.append(len(cond))

        else:

            Num.append(0.)
            Num_mlim.append(0.)
            Num_dlim.append(0.)
            Num_mlim_dlim.append(0.)

        if Num_mlim[-1] != 0:
            frac = Num_mlim_dlim[-1]/Num_mlim[-1]
        else:
            if Num_mlim[-1] != 0:
                frac = np.inf
            else:
                frac = 1.

        if verbose:
            print(prefix + "{:>16.6} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16.6}".format(*[np.log10(Sc_lim), Num[-1], Num_dlim[-1], Num_mlim[-1], Num_mlim_dlim[-1], frac]))

    if verbose:
        print(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["-"*16, "-"*16, "-"*16, "-"*16, "-"*20, "-"*16]))

    Num = np.array(Num)
    Num_dlim = np.array(Num_dlim)
    Num_mlim = np.array(Num_mlim)
    Num_mlim_dlim = np.array(Num_mlim_dlim)

    return Sc_lims, Num, Num_dlim, Num_mlim, Num_mlim_dlim


def mpi_get_Sc_group_info(Sc, dens, Omega_m, boxsize, ngrid, minvol, mindens,
    minmass, MPI, neval=10, overide_min_sum_M=None, overide_max_sum_M=None,
    periodic=True, verbose=True, prefix=''):
    """Computes group information for different thresholds of the cluster signature.

    Parameters
    ----------
    Sc : 3darray
        Cluster nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    mindens : float
        Minimum density for a group.
    minmass : float
        Minimum mass for a group.
    MPI : object
        MPIutils object.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sc_lims : array
        Cluster signature limits.
    Num : array
        Number of valid groups (i.e. with a size larger than the minimum volume).
    Num_mlim : array
        Number of valid groups with a mass larger than minmass.
    Num_dlim : array
        Number of valid groups with a density larger than mindens.
    Num_mlim_dlim : array
        Number of valid groups with a mass larger than minmass and density larger
        than mindens.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    avgmass = density.average_mass_per_cell(Omega_m, boxsize, ngrid)
    avgdens = avgmass/dV

    mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)

    logS_lim, sum_M = mpi_get_mass_below_logS(Sc, mass, MPI)

    if overide_max_sum_M is None:
        max_sum_M = MPI.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = MPI.min(sum_M)
        if min_sum_M < minmass:
            min_sum_M = minmass
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sc_lims = 10.**interp_logS_lim(interp_vals)

    Num = []
    Num_mlim = []
    Num_dlim = []
    Num_mlim_dlim = []

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["log10(Sc)", "Num(Groups)", "Num(d>dmin)", "Num(M>Mmin)", "Num(M>Mmin,d>dmin)", "Valid Frac."]))
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["-"*16, "-"*16, "-"*16, "-"*16, "-"*20, "-"*16]))

    for Sc_lim in Sc_lims:

        binmap = np.zeros(np.shape(Sc))
        cond = np.where(Sc > Sc_lim)
        binmap[cond] = 1.

        sumbinmap = MPI.sum(np.sum(binmap))
        sumbinmap = MPI.broadcast(sumbinmap)

        if sumbinmap > 0.:

            groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

            group_N = groups.mpi_get_ngroup(groupID, MPI)
            group_mass = groups.mpi_sum4group(groupID, mass, MPI)

            group_N = MPI.broadcast(group_N)
            group_mass = MPI.broadcast(group_mass)

            # true density rho
            group_dens = group_mass/(dV*group_N)
            # in units of mean density
            group_dens /= avgdens

            cond = np.where((group_N >= minpix))[0]

            Num.append(len(cond))

            group_N = group_N[cond]
            group_mass = group_mass[cond]
            group_dens = group_dens[cond]

            cond = np.where((group_mass >= minmass))[0]
            Num_mlim.append(len(cond))

            cond = np.where((group_dens >= mindens))[0]
            Num_dlim.append(len(cond))

            cond = np.where((group_mass >= minmass) & (group_dens >= mindens))[0]
            Num_mlim_dlim.append(len(cond))

        else:

            Num.append(0.)
            Num_mlim.append(0.)
            Num_dlim.append(0.)
            Num_mlim_dlim.append(0.)

        if Num_mlim[-1] != 0:
            frac = Num_mlim_dlim[-1]/Num_mlim[-1]
        else:
            if Num_mlim[-1] != 0:
                frac = np.inf
            else:
                frac = 1.

        if verbose:
            MPI.mpi_print_zero(prefix + "{:>16.6} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16.6}".format(*[np.log10(Sc_lim), Num[-1], Num_dlim[-1], Num_mlim[-1], Num_mlim_dlim[-1], frac]))

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16} | {:>16} | {:>16} | {:>20} | {:>16}".format(*["-"*16, "-"*16, "-"*16, "-"*16, "-"*20, "-"*16]))

    Num = np.array(Num)
    Num_dlim = np.array(Num_dlim)
    Num_mlim = np.array(Num_mlim)
    Num_mlim_dlim = np.array(Num_mlim_dlim)

    return Sc_lims, Num, Num_dlim, Num_mlim, Num_mlim_dlim


def get_clust_threshold(Sc_lims, Num_mlim, Num_mlim_dlim):
    """Returns the cluster significance threshold.

    Parameters
    ----------
    Sc_lims : array
        Cluster signature limits.
    Num_mlim : array
        Number of valid groups with a mass larger than minmass.
    Num_mlim_dlim : array
        Number of valid groups with a mass larger than minmass and density larger
        than mindens.

    Returns
    -------
    Sc_lim : float
        The cluster significance threshold.
    """
    frac = np.ones(len(Num_mlim_dlim))
    cond = np.where(Num_mlim > 0)[0]
    frac[cond] = Num_mlim_dlim[cond]/Num_mlim[cond]
    f = interp1d(frac, np.log10(Sc_lims))
    Sc_lim = 10.**f(0.5)
    return Sc_lim


def get_clust_map(Sc, Sc_lim, dens, Omega_m, boxsize, ngrid, minvol, minmass,
    periodic=True):
    """Apply the cluster significance threshold to find cluster environments.
    Only environment groups larger than the minvol and minmass are kept.

    Parameters
    ----------
    Sc : 3darray
        Cluster nexus signature.
    Sc_lim : float
        The cluster significance threshold.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    mindens : float
        Minimum density for a group.
    minmass : float
        Minimum mass for a group.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    clust_map : 3darray
        Binary map of the cluster environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    avgmass = density.average_mass_per_cell(Omega_m, boxsize, ngrid)
    avgdens = avgmass/dV

    mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)

    binmap = np.zeros(np.shape(Sc))
    cond = np.where(Sc > Sc_lim)
    binmap[cond] = 1.

    groupID = groups.groupfinder(binmap, periodic=periodic)

    group_N = groups.get_ngroup(groupID)
    group_mass = groups.sum4group(groupID, mass)

    # true density rho
    group_dens = group_mass/(dV*group_N)
    # in units of mean density
    group_dens /= avgdens

    #Edit
    cond = np.where((group_N >= minpix) & (group_mass > minmass))[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    clust_map = mask[groupID.astype('int')]

    return clust_map


def mpi_get_clust_map(Sc, Sc_lim, dens, Omega_m, boxsize, ngrid, minvol, minmass,
    MPI, periodic=True):
    """Apply the cluster significance threshold to find cluster environments.
    Only environment groups larger than the minvol and minmass are kept.

    Parameters
    ----------
    Sc : 3darray
        Cluster nexus signature.
    Sc_lim : float
        The cluster significance threshold.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    minmass : float
        Minimum mass for a group.
    MPI : object
        MPIutils object.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    clust_map : 3darray
        Binary map of the cluster environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    avgmass = density.average_mass_per_cell(Omega_m, boxsize, ngrid)
    avgdens = avgmass/dV

    mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)

    binmap = np.zeros(np.shape(Sc))
    cond = np.where(Sc > Sc_lim)
    binmap[cond] = 1.

    groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

    group_N = groups.mpi_get_ngroup(groupID, MPI)
    group_mass = groups.mpi_sum4group(groupID, mass, MPI)

    group_N = MPI.broadcast(group_N)
    group_mass = MPI.broadcast(group_mass)

    # true density rho
    group_dens = group_mass/(dV*group_N)
    # in units of mean density
    group_dens /= avgdens

    cond = np.where((group_N >= minpix) & (group_mass > minmass))[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    clust_map = mask[groupID.astype('int')]

    return clust_map


def get_Sf_group_info(Sf, dens, Omega_m, boxsize, ngrid, minvol, clust_map, neval=10,
    overide_min_sum_M=None, overide_max_sum_M=None, periodic=True, verbose=True, prefix=''):
    """Computes group information for different thresholds of the filament signature.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sf_lims : array
        Filament signature limits.
    SumM : array
        The sum of masses with filament signatures above the signature threshold.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)

    logS_lim, sum_M = get_mass_below_logS(Sf, mass)

    if overide_max_sum_M is None:
        max_sum_M = np.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = np.min(sum_M)
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sf_lims = 10.**interp_logS_lim(interp_vals)

    SumM = []

    if verbose:
        print(prefix + "{:>16} | {:>16}".format(*["log10(Sf)", "Sum(Mass)"]))
        print(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    for Sf_lim in Sf_lims:

        binmap = np.zeros(np.shape(Sf))
        cond = np.where((Sf > Sf_lim) & (clust_map == 0.))
        binmap[cond] = 1.

        sumbinmap = np.sum(binmap)

        if sumbinmap > 0.:

            groupID = groups.groupfinder(binmap, periodic=periodic)

            group_N = groups.get_ngroup(groupID)
            group_mass = groups.sum4group(groupID, mass)

            cond = np.where((group_N >= minpix))[0]

            SumM.append(np.sum(group_mass[cond]))

        else:

            SumM.append(0.)

        if verbose:
            print(prefix + "{:>16.6} | {:>16.6}".format(*[np.log10(Sf_lim), SumM[-1]]))

    if verbose:
        print(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    SumM = np.array(SumM)

    return Sf_lims, SumM


def get_filam_threshold(Sf, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    neval=10, periodic=True, verbose=True, prefix=''):
    """Returns the filament significance threshold.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sf_lim : float
        Filament signature threshold.
    logSf_lim : array
        The filament signature thresholds used to compute dM2.
    dM2 : array
        Derivative of the cumulated mass squared (i.e the mass contained above
        a given significance threshold).
    """

    # mask = np.ones(np.shape(clust_map))
    # cond = np.where(clust_map == 1.)
    # mask[cond] = 0
    #
    # mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)
    #
    # logSf_lim, sum_M = get_mass_below_logS(Sf, mass, nbins=nbins, mask=mask)
    #
    # dM2 = abs(fiesta.maths.dfdx(logSf_lim, sum_M**2.))
    #
    # Sf_lim = 10.**logSf_lim[np.argmax(dM2)]

    Sf_lims, sumM = get_Sf_group_info(Sf, dens, Omega_m, boxsize, ngrid,
        minvol, clust_map, neval=neval, periodic=periodic, verbose=verbose,
        prefix=prefix)

    interp_sumM = interp1d(np.log10(Sf_lims), sumM)
    logSf_lim = np.linspace(np.log10(Sf_lims[0]), np.log10(Sf_lims[-1]), 1000)
    M = interp_sumM(logSf_lim)

    dM2 = abs(fiesta.maths.dfdx(logSf_lim, M**2.))

    logSf_range = logSf_lim[-1] - logSf_lim[0]
    logSf_range += logSf_lim[1] - logSf_lim[0]
    sigma = 0.1

    dM2k = shift.cart.dct1D(dM2, logSf_range)
    k = shift.cart.kgrid1D(logSf_range, len(dM2))
    dM2k *= shift.cart.convolve_gaussian(k, sigma)
    dM2_gaussian = shift.cart.idct1D(dM2k, logSf_range)

    #Sf_lim = 10.**logSf_lim[np.argmax(dM2)]
    Sf_lim = 10.**logSf_lim[np.argmax(dM2_gaussian)]

    return Sf_lim, logSf_lim, dM2


def mpi_get_Sf_group_info(Sf, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    MPI, neval=10, overide_min_sum_M=None, overide_max_sum_M=None, periodic=True,
    verbose=True, prefix=''):
    """Computes group information for different thresholds of the filament signature.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    MPI : object
        MPI object.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sf_lims : array
        Filament signature limits.
    SumM : array
        The sum of masses with filament signatures above the signature threshold.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)

    logS_lim, sum_M = mpi_get_mass_below_logS(Sf, mass, MPI)

    if overide_max_sum_M is None:
        max_sum_M = MPI.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = MPI.min(sum_M)
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sf_lims = 10.**interp_logS_lim(interp_vals)

    SumM = []

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["log10(Sf)", "Sum(Mass)"]))
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    for Sf_lim in Sf_lims:

        binmap = np.zeros(np.shape(Sf))
        cond = np.where((Sf > Sf_lim) & (clust_map == 0.))
        binmap[cond] = 1.

        sumbinmap = MPI.sum(np.sum(binmap))
        sumbinmap = MPI.broadcast(sumbinmap)

        if sumbinmap > 0.:

            groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

            group_N = groups.mpi_get_ngroup(groupID, MPI)
            group_mass = groups.mpi_sum4group(groupID, mass, MPI)

            group_N = MPI.broadcast(group_N)
            group_mass = MPI.broadcast(group_mass)

            cond = np.where((group_N >= minpix))[0]

            SumM.append(np.sum(group_mass[cond]))

        else:

            SumM.append(0.)

        if verbose:
            MPI.mpi_print_zero(prefix + "{:>16.6} | {:>16.6}".format(*[np.log10(Sf_lim), SumM[-1]]))

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    SumM = np.array(SumM)

    return Sf_lims, SumM


def mpi_get_filam_threshold(Sf, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    MPI, neval=10, periodic=True, verbose=True, prefix=''):
    """Returns the filament significance threshold.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    MPI : object
        MPI object.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sf_lim : float
        Filament signature threshold.
    logSf_lim : array
        The filament signature thresholds used to compute dM2.
    dM2 : array
        Derivative of the cumulated mass squared (i.e the mass contained above
        a given significance threshold).
    """

    # mask = np.ones(np.shape(clust_map))
    # cond = np.where(clust_map == 1.)
    # mask[cond] = 0
    #
    # mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)
    #
    # logSf_lim, sum_M = mpi_get_mass_below_logS(Sf, mass, MPI, nbins=nbins, mask=mask)
    #
    # dM2 = abs(fiesta.maths.dfdx(logSf_lim, sum_M**2.))
    #
    # Sf_lim = 10.**logSf_lim[np.argmax(dM2)]

    Sf_lims, sumM = mpi_get_Sf_group_info(Sf, dens, Omega_m, boxsize, ngrid,
        minvol, clust_map, MPI, neval=neval, periodic=periodic, verbose=verbose,
        prefix=prefix)

    interp_sumM = interp1d(np.log10(Sf_lims), sumM)
    logSf_lim = np.linspace(np.log10(Sf_lims[0]), np.log10(Sf_lims[-1]), 1000)
    M = interp_sumM(logSf_lim)

    dM2 = abs(fiesta.maths.dfdx(logSf_lim, M**2.))

    logSf_range = logSf_lim[-1] - logSf_lim[0]
    logSf_range += logSf_lim[1] - logSf_lim[0]
    sigma = 0.1

    dM2k = shift.cart.dct1D(dM2, logSf_range)
    k = shift.cart.kgrid1D(logSf_range, len(dM2))
    dM2k *= shift.cart.convolve_gaussian(k, sigma)
    dM2_gaussian = shift.cart.idct1D(dM2k, logSf_range)

    #Sf_lim = 10.**logSf_lim[np.argmax(dM2)]
    Sf_lim = 10.**logSf_lim[np.argmax(dM2_gaussian)]

    return Sf_lim, logSf_lim, dM2


def get_filam_map(Sf, Sf_lim, dens, boxsize, ngrid, minvol, clust_map, periodic=True):
    """Apply the filament significance threshold to find filament environments.
    Only environment groups larger than the minvol are kept.

    Parameters
    ----------
    Sf : 3darray
        Cluster nexus signature.
    Sf_lim : float
        The filament significance threshold.
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    filam_map : 3darray
        Binary map of the filament environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    binmap = np.zeros(np.shape(Sf))
    cond = np.where((Sf > Sf_lim) & (clust_map == 0.))
    binmap[cond] = 1.

    groupID = groups.groupfinder(binmap, periodic=periodic)

    group_N = groups.get_ngroup(groupID)

    cond = np.where(group_N >= minpix)[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    filam_map = mask[groupID.astype('int')]

    return filam_map


def mpi_get_filam_map(Sf, Sf_lim, dens, boxsize, ngrid, minvol, clust_map, MPI,
    periodic=True):
    """Apply the filament significance threshold to find filament environments.
    Only environment groups larger than the minvol are kept.

    Parameters
    ----------
    Sf : 3darray
        Cluster nexus signature.
    Sf_lim : float
        The filament significance threshold.
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    MPI : object
        MPI object.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    filam_map : 3darray
        Binary map of the filament environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    binmap = np.zeros(np.shape(Sf))
    cond = np.where((Sf > Sf_lim) & (clust_map == 0.))
    binmap[cond] = 1.

    groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

    group_N = groups.mpi_get_ngroup(groupID, MPI)

    group_N = MPI.broadcast(group_N)

    cond = np.where(group_N >= minpix)[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    filam_map = mask[groupID.astype('int')]

    return filam_map


def get_Sw_group_info(Sw, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    filam_map, neval=10, overide_min_sum_M=None, overide_max_sum_M=None,
    periodic=True, verbose=True, prefix=''):
    """Computes group information for different thresholds of the sheet signature.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sw_lims : array
        Sheet signature limits.
    SumM : array
        The sum of masses with sheet signatures above the signature threshold.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)

    logS_lim, sum_M = get_mass_below_logS(Sw, mass)

    if overide_max_sum_M is None:
        max_sum_M = np.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = np.min(sum_M)
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sw_lims = 10.**interp_logS_lim(interp_vals)

    SumM = []

    if verbose:
        print(prefix + "{:>16} | {:>16}".format(*["log10(Sw)", "Sum(Mass)"]))
        print(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    for Sw_lim in Sw_lims:

        binmap = np.zeros(np.shape(Sw))
        cond = np.where((Sw > Sw_lim) & (clust_map == 0.) & (filam_map == 0.))
        binmap[cond] = 1.

        sumbinmap = np.sum(binmap)

        if sumbinmap > 0.:

            groupID = groups.groupfinder(binmap, periodic=periodic)

            group_N = groups.get_ngroup(groupID)
            group_mass = groups.sum4group(groupID, mass)

            cond = np.where((group_N >= minpix))[0]

            SumM.append(np.sum(group_mass[cond]))

        else:

            SumM.append(0.)

        if verbose:
            print(prefix + "{:>16.6} | {:>16.6}".format(*[np.log10(Sw_lim), SumM[-1]]))

    if verbose:
        print(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    SumM = np.array(SumM)

    return Sw_lims, SumM


def get_sheet_threshold(Sw, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    filam_map, neval=10, periodic=True, verbose=True, prefix=''):
    """Returns the sheet significance threshold.

    Parameters
    ----------
    Sw : 3darray
        Wall nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sw_lim : float
        Wall signature threshold.
    logSw_lim : array
        The wall signature thresholds used to compute dM2.
    dM2 : array
        Derivative of the cumulated mass squared (i.e the mass contained above
        a given significance threshold).
    """

    # mask = np.ones(np.shape(clust_map))
    # cond = np.where((clust_map == 1.) | (filam_map == 1.))
    # mask[cond] = 0
    #
    # mass = density.dens2mass(dens, Omega_m, boxsize, ngrid)
    #
    # logSw_lim, sum_M = get_mass_below_logS(Sw, mass, nbins=nbins, mask=mask)
    #
    # dM2 = abs(fiesta.maths.dfdx(logSw_lim, sum_M**2.))
    #
    # Sw_lim = 10.**logSw_lim[np.argmax(dM2)]

    Sw_lims, sumM = get_Sw_group_info(Sw, dens, Omega_m, boxsize, ngrid,
        minvol, clust_map, filam_map, neval=neval, periodic=periodic, verbose=verbose,
        prefix=prefix)

    interp_sumM = interp1d(np.log10(Sw_lims), sumM)
    logSw_lim = np.linspace(np.log10(Sw_lims[0]), np.log10(Sw_lims[-1]), 1000)
    M = interp_sumM(logSw_lim)

    dM2 = abs(fiesta.maths.dfdx(logSw_lim, M**2.))

    logSw_range = logSw_lim[-1] - logSw_lim[0]
    logSw_range += logSw_lim[1] - logSw_lim[0]
    sigma = 0.1

    dM2k = shift.cart.dct1D(dM2, logSw_range)
    k = shift.cart.kgrid1D(logSw_range, len(dM2))
    dM2k *= shift.cart.convolve_gaussian(k, sigma)
    dM2_gaussian = shift.cart.idct1D(dM2k, logSw_range)

    #Sw_lim = 10.**logSw_lim[np.argmax(dM2)]
    Sw_lim = 10.**logSw_lim[np.argmax(dM2_gaussian)]

    return Sw_lim, logSw_lim, dM2


def mpi_get_Sw_group_info(Sw, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    filam_map, MPI, neval=10, overide_min_sum_M=None, overide_max_sum_M=None,
    periodic=True, verbose=True, prefix=''):
    """Computes group information for different thresholds of the wall signature.

    Parameters
    ----------
    Sf : 3darray
        Filament nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    MPI : object
        MPI object.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    overide_min_sum_M, overide_max_sum_M : float, optional
        Overide min/max mass thresholds for computing signature thresholds.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sw_lims : array
        Sheet signature limits.
    SumM : array
        The sum of masses with sheet signatures above the signature threshold.
    """

    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)

    logS_lim, sum_M = mpi_get_mass_below_logS(Sw, mass, MPI)

    if overide_max_sum_M is None:
        max_sum_M = MPI.max(sum_M)
    else:
        max_sum_M = overide_max_sum_M

    if overide_min_sum_M is None:
        min_sum_M = MPI.min(sum_M)
    else:
        min_sum_m = overide_min_sum_M

    interp_vals = np.linspace(max_sum_M, min_sum_M, neval)

    interp_logS_lim = interp1d(sum_M, logS_lim)

    Sw_lims = 10.**interp_logS_lim(interp_vals)

    SumM = []

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["log10(Sw)", "Sum(Mass)"]))
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    for Sw_lim in Sw_lims:

        binmap = np.zeros(np.shape(Sw))
        cond = np.where((Sw > Sw_lim) & (clust_map == 0.) & (filam_map == 0.))
        binmap[cond] = 1.

        sumbinmap = MPI.sum(np.sum(binmap))
        sumbinmap = MPI.broadcast(sumbinmap)

        if sumbinmap > 0.:

            groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

            group_N = groups.mpi_get_ngroup(groupID, MPI)
            group_mass = groups.mpi_sum4group(groupID, mass, MPI)

            group_N = MPI.broadcast(group_N)
            group_mass = MPI.broadcast(group_mass)

            cond = np.where((group_N >= minpix))[0]

            SumM.append(np.sum(group_mass[cond]))

        else:

            SumM.append(0.)

        if verbose:
            MPI.mpi_print_zero(prefix + "{:>16.6} | {:>16.6}".format(*[np.log10(Sw_lim), SumM[-1]]))

    if verbose:
        MPI.mpi_print_zero(prefix + "{:>16} | {:>16}".format(*["-"*16, "-"*16]))

    SumM = np.array(SumM)

    return Sw_lims, SumM


def mpi_get_sheet_threshold(Sw, dens, Omega_m, boxsize, ngrid, minvol, clust_map,
    filam_map, MPI, neval=10, periodic=True, verbose=True, prefix=''):
    """Returns the sheet significance threshold.

    Parameters
    ----------
    Sw : 3darray
        Wall nexus signature.
    dens : 3darray
        Density field.
    Omega_m : float
        Matter density.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    MPI : object
        MPI object.
    neval : int, optional
        Number of signature threshold levels to compute group information. Levels
        are equal along a mass weighted cumulative distribution of the cluster
        signature.
    periodic : bool, optional
        Periodic boundary.
    verbose : bool, optional
        Print progress statements.
    prefix : str, optional
        Print statement prefix.

    Returns
    -------
    Sw_lim : float
        Wall signature threshold.
    logSw_lim : array
        The wall signature thresholds used to compute dM2.
    dM2 : array
        Derivative of the cumulated mass squared (i.e the mass contained above
        a given significance threshold).
    """

    # mask = np.ones(np.shape(clust_map))
    # cond = np.where((clust_map == 1.) | (filam_map == 1.))
    # mask[cond] = 0
    #
    # mass = density.mpi_dens2mass(dens, Omega_m, boxsize, ngrid, MPI)
    #
    # logSw_lim, sum_M = mpi_get_mass_below_logS(Sw, mass, MPI, nbins=nbins, mask=mask)
    #
    # dM2 = abs(fiesta.maths.dfdx(logSw_lim, sum_M**2.))
    #
    # Sw_lim = 10.**logSw_lim[np.argmax(dM2)]

    Sw_lims, sumM = mpi_get_Sw_group_info(Sw, dens, Omega_m, boxsize, ngrid,
        minvol, clust_map, filam_map, MPI, neval=neval, periodic=periodic,
        verbose=verbose, prefix=prefix)

    interp_sumM = interp1d(np.log10(Sw_lims), sumM)
    logSw_lim = np.linspace(np.log10(Sw_lims[0]), np.log10(Sw_lims[-1]), 1000)
    M = interp_sumM(logSw_lim)

    dM2 = abs(fiesta.maths.dfdx(logSw_lim, M**2.))

    logSw_range = logSw_lim[-1] - logSw_lim[0]
    logSw_range += logSw_lim[1] - logSw_lim[0]
    sigma = 0.1

    dM2k = shift.cart.dct1D(dM2, logSw_range)
    k = shift.cart.kgrid1D(logSw_range, len(dM2))
    dM2k *= shift.cart.convolve_gaussian(k, sigma)
    dM2_gaussian = shift.cart.idct1D(dM2k, logSw_range)

    Sw_lim = 10.**logSw_lim[np.argmax(dM2)]
    Sw_lim = 10.**logSw_lim[np.argmax(dM2_gaussian)]
    
    return Sw_lim, logSw_lim, dM2


def get_sheet_map(Sw, Sw_lim, dens, boxsize, ngrid, minvol, clust_map,
    filam_map, periodic=True):
    """Apply the sheet significance threshold to find sheet environments. Only
    environment groups larger than the minvol are kept.

    Parameters
    ----------
    Sf : 3darray
        Cluster nexus signature.
    Sf_lim : float
        The filament significance threshold.
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    sheet_map : 3darray
        Binary map of the sheet environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    binmap = np.zeros(np.shape(Sw))
    cond = np.where((Sw > Sw_lim) & (clust_map == 0.) & (filam_map == 0.))
    binmap[cond] = 1.

    groupID = groups.groupfinder(binmap, periodic=periodic)

    group_N = groups.get_ngroup(groupID)

    cond = np.where(group_N >= minpix)[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    sheet_map = mask[groupID.astype('int')]

    return sheet_map



def mpi_get_sheet_map(Sw, Sw_lim, dens, boxsize, ngrid, minvol, clust_map,
    filam_map, MPI, periodic=True):
    """Apply the sheet significance threshold to find sheet environments. Only
    environment groups larger than the minvol are kept.

    Parameters
    ----------
    Sf : 3darray
        Cluster nexus signature.
    Sf_lim : float
        The filament significance threshold.
    dens : 3darray
        Density field.
    boxsize : float
        Size of the box.
    ngrid : int
        Grid size along each axis.
    minvol : float
        Minimum volume for a group.
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    MPI : object
        MPI object.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    sheet_map : 3darray
        Binary map of the sheet environments.
    """
    dV = (boxsize/ngrid)**3.
    minpix = minvol/dV

    binmap = np.zeros(np.shape(Sw))
    cond = np.where((Sw > Sw_lim) & (clust_map == 0.) & (filam_map == 0.))
    binmap[cond] = 1.

    groupID = groups.mpi_groupfinder(binmap, MPI, periodic=periodic)

    group_N = groups.mpi_get_ngroup(groupID, MPI)

    group_N = MPI.broadcast(group_N)

    cond = np.where(group_N >= minpix)[0]

    mask = np.zeros(len(group_N)+1)
    mask[cond+1] = 1.

    sheet_map = mask[groupID.astype('int')]

    return sheet_map


def get_cweb_map(clust_map, filam_map, sheet_map):
    """Combines cluster, filament and sheet map to create a unified cosmic web
    environment map.

    Parameters
    ----------
    clust_map : 3darray
        Binary map of the cluster environments.
    filam_map : 3darray
        Binary map of the filament environments.
    sheet_map : 3darray
        Binary map of the sheet environments.
    """
    cweb = np.zeros(np.shape(clust_map))
    cond = np.where(clust_map == 1.)
    cweb[cond] = 3.
    cond = np.where((clust_map == 0.) & (filam_map == 1.))
    cweb[cond] = 2.
    cond = np.where((clust_map == 0.) & (filam_map == 0.) & (sheet_map == 1.))
    cweb[cond] = 1.
    return cweb
