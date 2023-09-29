import numpy as np

from ..ext import shift

from scipy.interpolate import interp1d

from .. import maths


def get_theta(x):
    """Heaviside function with center at 0.

    Parameters
    ----------
    x : float or array
        Array.
    """
    return np.heaviside(x, 0.)


def get_Theta(x):
    """Positive linear function.

    Parameters
    ----------
    x : float or array
        Array.
    """
    return x*get_theta(x)


def get_nexus_strength(eig):
    """Nexus+ strenght I.

    Parameters
    ----------
    eig : array or 2darray
        Single eigenvalues or a set of eigenvalues.

    Returns
    -------
    Ic, If, Iw : array
        Strength for clusters (Ic), filament (If), walls (Iw).
    """
    if len(np.shape(eig)) == 1:
        Ic = abs(eig[2]/eig[0])
        If = abs(eig[1]/eig[0])*get_Theta(1.-abs(eig[2]/eig[0]))
        Iw = get_Theta(1.-abs(eig[1]/eig[0]))*get_Theta(1.-abs(eig[2]/eig[0]))
    else:
        Ic = abs(eig[:,2]/eig[:,0])
        If = abs(eig[:,1]/eig[:,0])*get_Theta(1.-abs(eig[:,2]/eig[:,0]))
        Iw = get_Theta(1.-abs(eig[:,1]/eig[:,0]))*get_Theta(1.-abs(eig[:,2]/eig[:,0]))
    return Ic, If, Iw


def get_nexus_signature(eig, Ic, If, Iw):
    """Nexus+ strenght S.

    Parameters
    ----------
    eig : array or 2darray
        Single eigenvalues or a set of eigenvalues.
    Ic, If, Iw : array
        Strength for clusters (Ic), filament (If), walls (Iw).

    Returns
    -------
    Sc, Sf, Sw : array
        Signature for clusters (Sc), filament (Sf), walls (Sw).
    """
    if len(np.shape(eig)) == 1:
        Sc = Ic*abs(eig[2])*get_theta(-eig[0])*get_theta(-eig[1])*get_theta(-eig[2])
        Sf = If*abs(eig[1])*get_theta(-eig[0])*get_theta(-eig[1])
        Sw = Iw*abs(eig[0])*get_theta(-eig[0])
    else:
        Sc = Ic*abs(eig[:,2])*get_theta(-eig[:,0])*get_theta(-eig[:,1])*get_theta(-eig[:,2])
        Sf = If*abs(eig[:,1])*get_theta(-eig[:,0])*get_theta(-eig[:,1])
        Sw = Iw*abs(eig[:,0])*get_theta(-eig[:,0])
    return Sc, Sf, Sw


def get_nexus_sig(dens, boxsize, logsmooth=True, R0=0.5, Nmax=7, verbose=True,
    verbose_prefix='', debug=True):
    """Classifies simulation based on multiscale Hessian of a log-Gaussian
    smoothed density field.

    Parameters
    ----------
    dens : 3darray
        Density of the 3d field.
    boxsize : float
        Size of the box.
    output :
    R0 : float, optional
        Minimum smoothing scale.
    Nmax : int, optional
        Number of smoothing scales, going by sqrt(2)^N * R0.
    verbose : bool, optional
        Determines whether to print updates about NEXUS+ calculation.
    verbose_prefix : str, optional
        Prefix to verbose print statements.

    Returns
    -------
    Sc, Sf, Sw : 3darray
        Nexus signature for cluster, filament and wall environment.
    """
    if verbose:
        print(verbose_prefix+"Checking dens > 0")
    # check if there are zeros
    cond = np.where(dens.flatten() <= 0.)[0]
    assert len(cond) == 0., 'dens cannot have values <= 0.'
    if verbose:
        print(verbose_prefix+"Constructing Fourier grid")
    # determine shape of the dens grid
    dshape = np.shape(dens)
    ngrid = dshape[0]
    # construct fourier grid
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # determine mean of the field and logarithm
    dmean = np.mean(dens)
    if logsmooth:
        log10d = np.log10(dens)
        # FFT log field.
        log10dk = shift.cart.fft3D(log10d, boxsize)
    else:
        dk = shift.cart.fft3D(dens, boxsize)
    # smoothing scales
    N = np.arange(Nmax)
    Rns = R0*np.sqrt(2.)**N
    # loop over smoothing scales
    for Rn in Rns:
        if logsmooth:
            if verbose:
                print()
                print(verbose_prefix+"Smooth log(dens) at Rn = %.4f" % Rn)
            # log-Gaussian smoothing
            log10dk_smooth = np.copy(log10dk)*shift.cart.convolve_gaussian(kmag, Rn)
            log10d_smooth = shift.cart.ifft3D(log10dk_smooth, boxsize)
            d_smooth = 10.**log10d_smooth
            dmean_smooth = np.mean(d_smooth)
            if verbose:
                print(verbose_prefix+"Mean of smoothed dens = %.4f" % dmean_smooth)
            #d_smooth *= dmean/dmean_smooth
        else:
            if verbose:
                print()
                print(verbose_prefix+"Smooth dens at Rn = %.4f" % Rn)
            dk_smooth = np.copy(dk)*shift.cart.convolve_gaussian(kmag, Rn)
            d_smooth = shift.cart.ifft3D(dk_smooth, boxsize)
            dmean_smooth = np.mean(d_smooth)
        if verbose:
            print(verbose_prefix+"Compute Hessian matrix at Rn = %.4f" % Rn)
        # FFT again to compute things in Fourier space
        dk_smooth = shift.cart.fft3D(d_smooth, boxsize)
        # Compute Hessian in Fourier space
        Hkxx = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth)
        Hkxy = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth, k2=ky3d)
        Hkxz = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth, k2=kz3d)
        Hkyy = (Rn**2.) * shift.cart.dfdk2(ky3d, dk_smooth)
        Hkyz = (Rn**2.) * shift.cart.dfdk2(ky3d, dk_smooth, k2=kz3d)
        Hkzz = (Rn**2.) * shift.cart.dfdk2(kz3d, dk_smooth)
        # Backward FFT for Hessian fields
        Hxx = shift.cart.ifft3D(Hkxx, boxsize)
        Hxy = shift.cart.ifft3D(Hkxy, boxsize)
        Hxz = shift.cart.ifft3D(Hkxz, boxsize)
        Hyy = shift.cart.ifft3D(Hkyy, boxsize)
        Hyz = shift.cart.ifft3D(Hkyz, boxsize)
        Hzz = shift.cart.ifft3D(Hkzz, boxsize)
        Hxx = Hxx.flatten()
        Hxy = Hxy.flatten()
        Hxz = Hxz.flatten()
        Hyy = Hyy.flatten()
        Hyz = Hyz.flatten()
        Hzz = Hzz.flatten()
        if verbose:
            print(verbose_prefix+"Compute Nexus signature at Rn = %.4f" % Rn)
        # Computes eigenvalues
        eig = maths.get_eig_3by3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)
        _Ic, _If, _Iw = get_nexus_strength(eig)
        _Sc, _Sf, _Sw = get_nexus_signature(eig, _Ic, _If, _Iw)
        # Find maximum across already evaluated smoothing scales
        if Rn == Rns[0]:
            Sc = np.copy(_Sc)
            Sf = np.copy(_Sf)
            Sw = np.copy(_Sw)
        else:
            cond = np.where(_Sc > Sc)[0]
            Sc[cond] = _Sc[cond]
            cond = np.where(_Sf > Sf)[0]
            Sf[cond] = _Sf[cond]
            cond = np.where(_Sw > Sw)[0]
            Sw[cond] = _Sw[cond]
    if verbose:
        print()
        print(verbose_prefix+"Output Nexus signature")
    Sc = Sc.reshape(dshape)
    Sf = Sf.reshape(dshape)
    Sw = Sw.reshape(dshape)
    if debug:
        return Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, eig
    else:
        return Sc, Sf, Sw



def mpi_get_nexus_sig(dens, ngrid, boxsize, MPI, logsmooth=True, R0=0.5,
    Nmax=7, verbose=True, verbose_prefix=''):
    """Classifies simulation based on multiscale Hessian of a log-Gaussian
    smoothed density field.

    Parameters
    ----------
    dens : 3darray
        Density of the 3d field.
    ngrid : int
        Ngrid of the box.
    boxsize : float
        Size of the box.
    MPI : object
        MPIutils mpi object.
    R0 : float, optional
        Minimum smoothing scale.
    Nmax : int, optional
        Number of smoothing scales, going by sqrt(2)^N * R0.
    verbose : bool, optional
        Determines whether to print updates about NEXUS+ calculation.
    verbose_prefix : str, optional
        Prefix to verbose print statements.

    Returns
    -------
    Sc, Sf, Sw : 3darray
        Nexus signature for cluster, filament and wall environment.
    """
    if verbose:
        MPI.mpi_print_zero(verbose_prefix+"Checking dens > 0")
    # check if there are zeros
    cond = np.where(dens.flatten() <= 0.)[0]
    if len(cond) == 0:
        check = True
    else:
        check = False
    checks = MPI.collect(check)
    if MPI.rank == 0:
        cond = np.where(checks == False)[0]
        if len(cond) == 0:
            check = True
        else:
            check = False
        MPI.send(check, tag=11)
    else:
        check = MPI.recv(0, tag=11)
    MPI.wait()
    assert check, 'dens cannot have values <= 0.'
    if verbose:
        MPI.mpi_print_zero(verbose_prefix+"Constructing Fourier grid")
    # determine shape of the dens grid
    dshape = np.shape(dens)
    # determine mean of the field and logarithm
    dmean = MPI.mean(dens)
    # construct fourier grid
    kx3d, ky3d, kz3d = shift.cart.mpi_kgrid3D(boxsize, ngrid, MPI)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # smoothing scales
    N = np.arange(Nmax)
    Rns = R0*np.sqrt(2.)**N
    # loop over smoothing scales
    for Rn in Rns:
        if logsmooth:
            if verbose:
                MPI.mpi_print_zero()
                MPI.mpi_print_zero(verbose_prefix+"Smooth log(dens) at Rn = %.4f" % Rn)
            log10d = np.log10(dens)
            # FFT log field.
            log10dk = shift.cart.mpi_fft3D(log10d, boxsize, ngrid, MPI)
            log10dk *= shift.cart.convolve_gaussian(kmag, Rn)
            log10d = shift.cart.mpi_ifft3D(log10dk, boxsize, ngrid, MPI)
            d_smooth = 10.**log10d
            dmean_smooth = MPI.mean(d_smooth)
            if verbose:
                MPI.mpi_print_zero(verbose_prefix+"Mean of smoothed dens = %.4f" % dmean_smooth)
            d_smooth *= dmean/dmean_smooth
        else:
            if verbose:
                MPI.mpi_print_zero()
                MPI.mpi_print_zero(verbose_prefix+"Smooth dens at Rn = %.4f" % Rn)
            dk_smooth = shift.cart.mpi_fft3D(dens, boxsize, ngrid, MPI)
            dk_smooth *= shift.cart.convolve_gaussian(kmag, Rn)
            d_smooth = shift.cart.mpi_ifft3D(dk_smooth, boxsize, ngrid, MPI)
        if verbose:
            MPI.mpi_print_zero(verbose_prefix+"Compute Hessian matrix at Rn = %.4f" % Rn)
        # FFT again to compute things in Fourier space
        dk_smooth = shift.cart.mpi_fft3D(d_smooth, boxsize, ngrid, MPI)
        # Compute Hessian in Fourier space
        Hkxx = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth)
        Hkxy = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth, k2=ky3d)
        Hkxz = (Rn**2.) * shift.cart.dfdk2(kx3d, dk_smooth, k2=kz3d)
        Hkyy = (Rn**2.) * shift.cart.dfdk2(ky3d, dk_smooth)
        Hkyz = (Rn**2.) * shift.cart.dfdk2(ky3d, dk_smooth, k2=kz3d)
        Hkzz = (Rn**2.) * shift.cart.dfdk2(kz3d, dk_smooth)
        # Backward FFT for Hessian fields
        Hxx = shift.cart.mpi_ifft3D(Hkxx, boxsize, ngrid, MPI)
        Hxy = shift.cart.mpi_ifft3D(Hkxy, boxsize, ngrid, MPI)
        Hxz = shift.cart.mpi_ifft3D(Hkxz, boxsize, ngrid, MPI)
        Hyy = shift.cart.mpi_ifft3D(Hkyy, boxsize, ngrid, MPI)
        Hyz = shift.cart.mpi_ifft3D(Hkyz, boxsize, ngrid, MPI)
        Hzz = shift.cart.mpi_ifft3D(Hkzz, boxsize, ngrid, MPI)
        Hxx = Hxx.flatten()
        Hxy = Hxy.flatten()
        Hxz = Hxz.flatten()
        Hyy = Hyy.flatten()
        Hyz = Hyz.flatten()
        Hzz = Hzz.flatten()
        if verbose:
            MPI.mpi_print_zero(verbose_prefix+"Compute Nexus signature at Rn = %.4f" % Rn)
        # Computes eigenvalues
        eig = maths.get_eig_3by3(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)
        _Ic, _If, _Iw = get_nexus_strength(eig)
        _Sc, _Sf, _Sw = get_nexus_signature(eig, _Ic, _If, _Iw)
        # Find maximum across already evaluated smoothing scales
        if Rn == Rns[0]:
            Sc = _Sc
            Sf = _Sf
            Sw = _Sw
            cond = np.where(np.isfinite(Sc) == False)[0]
            Sc[cond] = 0.
            cond = np.where(np.isfinite(Sf) == False)[0]
            Sf[cond] = 0.
            cond = np.where(np.isfinite(Sw) == False)[0]
            Sw[cond] = 0.
        else:
            cond = np.where(_Sc > Sc)[0]
            Sc[cond] = _Sc[cond]
            cond = np.where(_Sf > Sf)[0]
            Sf[cond] = _Sf[cond]
            cond = np.where(_Sw > Sw)[0]
            Sw[cond] = _Sw[cond]
    if verbose:
        MPI.mpi_print_zero()
        MPI.mpi_print_zero(verbose_prefix+"Output Nexus signature")
    Sc = Sc.reshape(dshape)
    Sf = Sf.reshape(dshape)
    Sw = Sw.reshape(dshape)
    return Sc, Sf, Sw
