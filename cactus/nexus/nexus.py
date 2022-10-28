import numpy as np
import magpie
import shift

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
    return x*gettheta(x)


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
    if len(np.shape(eigs)) == 1:
        Ic = abs(eig[2]/eig[0])
        If = abs(eig[1]/eig[0])*getTheta(1.-abs(eig[2]/eig[0]))
        Iw = getTheta(1.-abs(eig[1]/eig[0]))*getTheta(1.-abs(eig[2]/eig[0]))
    else:
        Ic = abs(eig[:,2]/eig[:,0])
        If = abs(eig[:,1]/eig[:,0])*getTheta(1.-abs(eig[:,2]/eig[:,0]))
        Iw = getTheta(1.-abs(eig[:,1]/eig[:,0]))*getTheta(1.-abs(eig[:,2]/eig[:,0]))
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
    Sc = Ic*abs(eig[2])*gettheta(-eig[0])*gettheta(-eig[1])*gettheta(-eig[2])
    Sf = If*abs(eig[1])*gettheta(-eig[0])*gettheta(-eig[1])
    Sw = Iw*abs(eig[0])*gettheta(-eig[0])
    return Sc, Sf, Sw


def run_nexus_plus(dens, boxsize, R0=0.5, Nmax=7, virdens=370., verbose=True):
    """Classifies simulation based on multiscale Hessian of a log-Gaussian
    smoothed density field.

    Parameters
    ----------
    dens : 3darray
        Density of the 3d field.
    boxsize : float
        Size of the box.
    R0 : float, optional
        Minimum smoothing scale.
    Nmax : int, optional
        Number of smoothing scales, going by sqrt(2)^N * R0.
    virdens : float, optional
        Virilised density. Assumed to be 370., given by spherical collapse
        models at z=0.
    verbose : bool, optional
        Determines whether to print updates about NEXUS+ calculation.

    Returns
    -------
    cweb : 3darray
        Cosmic web classification where: 0 = void, 1 = wall, 2 = filament and
        3 = cluster.
    """
    # check if there are zeros
    cond = np.where(dens.flatten() <= 0.)[0]
    assert len(cond), 'dens cannot have values <= 0.'
    # determine shape of the dens grid
    dshape = np.shape(dens)
    ngrid = dshape[0]
    # construct fourier grid
    x3d, y3d, z3d = shift.cart.grid3D(boxsize, ngrid)
    kx3d, ky3d, kz3d = shift.cart.kgrid3D(boxsize, ngrid)
    kmag = np.sqrt(kx3d**2. + ky3d**2. + kz3d**2.)
    # determine mean of the field and logarithm
    dmean = np.mean(dens)
    log10d = np.log10(dens)
    # FFT log field.
    log10dk = shift.cart.fft3D(log10d, boxsize)
    # smoothing scales
    N = np.arange(Nmax)
    Rns = R0*np.sqrt(2.)**N
    # loop over smoothing scales
    for Rn in Rns:
        # log-Gaussian smoothing
        log10dk_smooth = np.copy(log10dk)*shift.cart.convolve_gaussian(kmag, Rn)
        log10d_smooth = shift.cart.ifft3D(log10dk_smooth, boxsize)
        d_smooth = 10.**log10d_smooth
        dmean_smooth = np.mean(d_smooth)
        d_smooth *= dmean/dmean_smooth
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
        # Computes eigenvalues
        eigs = maths.get_eig_3by3(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz)
        _Ic, _If, _Iw = get_nexus_strength(eig)
        _Sc, _Sf, _Sw = get_nexus_signature(eig, Ic, If, Iw)
        # Find maximum across already evaluated smoothing scales
        if Rn == Rns[0]:
            S_c = _S_c
            S_f = _S_f
            S_w = _S_w
        else:
            cond = np.where(_S_c > S_c)[0]
            S_c[cond] = _S_c[cond]
            cond = np.where(_S_f > S_f)[0]
            S_f[cond] = _S_f[cond]
            cond = np.where(_S_w > S_w)[0]
            S_w[cond] = _S_w[cond]
    S_c = S_c.flatten()
    S_f = S_f.flatten()
    S_w = S_w.flatten()
    cweb = np.zeros(len(S_c))
    # Determine Mass in Cells
    den = dens.flatten()
    dx = boxsize/ngrid
    mass = dens * dx**3.
    # get cluster environment
    virialised = np.zeros(len(dens))
    cond = np.where(np.log10(dens) >= np.log10(virdens))[0]
    virialised[cond] = 1.
    cond = np.where(S_c > 0.)[0]
    logSc_cut = np.log10(S_c[cond])
    xlogSc_min = np.min(np.log10(S_c[cond]))
    xlogSc_max = np.max(np.log10(S_c[cond]))
    xlogSc = magpie.grids.grid1d(xlogsc_max-xlogsc_min, 100,
                                 origin=xlogsc_min, return_edges=False)
    pixID = magpie.pixels.pos2pix_cart1d(logSc_cut),
                                         xlogsc_max-xlogsc_min, 100,
                                         origin=xlogsc_min))
    hist_Sc_all = magpie.pixels.bin_pix(pixID, 100, weights=np.ones(len(pixID)))
    hist_Sc_vir = magpie.pixels.bin_pix(pixID, 100, weights=virialised[cond])
    y_Sc_vir = np.cumsum(hist_Sc_vir[::-1])[::-1]
    y_Sc_all = np.cumsum(hist_Sc_all[::-1])[::-1]
    y_Sc = y_Sc_vir/y_Sc_all
    interp_y_Sc = interp1d(y_Sc, xlogSc)
    Sc_threshold = 10.**interpcdf(0.5)
    cond = np.where(S_c >= Sc_lim)[0]
    cweb[cond] = 3.
    # get filament environment
    cond = np.where((S_f > 0.) & (cweb == 0.))[0]
    logSf_cut = np.log10(S_f[cond])
    mass_Sf_cut = mass[cond]
    xlogSf_min = np.min(logSf_cut)
    xlogSf_max = np.max(logSf_cut)
    xlogSf = magpie.grids.grid1d(xlogSf_max-xlogSf_min, 100, origin=xlogSf_min,
                                 return_edges=False)
pixID = magpie.pixels.pos2pix_cart1d(logSf_cut, xlogSf_max-log_Sf_min, 100, origin=log_Sf_min)
W_Sf = magpie.pixels.bin_pix(pixID, 100, weights=mass_Sf_cut)

M_Sf = np.cumsum(W_Sf[::-1])[::-1]
#M_Sf /= np.max(M_Sf)
#M_Sf = 1 - M_Sf

plt.plot(logSf_grid, M_Sf)
plt.show()

M2_Sf = M_Sf**2.

import TheoryCL

DeltaM2_Sf = abs(TheoryCL.maths.numerical_differentiate(logSf_grid, M2_Sf))
