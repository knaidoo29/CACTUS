import numpy as np


def save_dens(prefix, rank, x3D, y3D, z3D, dens, Ngrid, Boxsize):
    """Save density in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.
    x3D, y3D, z3D : array
        Cartesian grid.
    dens : array
        Density values.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.
    """
    fname = prefix + str(rank) + ".npz"
    np.savez(fname, Ngrid=Ngrid, Boxsize=Boxsize, x3D=x3D, y3D=y3D, z3D=z3D, dens=dens)


def check_file(prefix, rank, Ngrid, Boxsize, MPI):
    """Load file in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.

    Returns
    -------
    ERROR: bool
        If file matches Ngrid.
    """
    fname = prefix + str(rank) + ".npz"
    data = np.load(fname)
    if data['Ngrid'] == Ngrid and data['Boxsize'] == Boxsize:
        ERROR = False
    else:
        MPI.mpi_print_zero(" ERROR: Ngrid and Boxsize of file:", data['Ngrid'], "and", data["Boxsize"],
                            "does not match input:", Ngrid, "and", Boxsize)
        ERROR = True
    return ERROR


def load_dens(prefix, rank):
    """Load density in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.

    Returns
    -------
    x3D, y3D, z3D : array
        Cartesian grid.
    dens : array
        Density values.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.
    """
    fname = prefix + str(rank) + ".npz"
    data = np.load(fname)
    Ngrid = data['Ngrid']
    Boxsize = data['Boxsize']
    x3D = data['x3D']
    y3D = data['y3D']
    z3D = data['z3D']
    dens = data['dens']
    return x3D, y3D, z3D, dens, Ngrid, Boxsize



def save_nexus_sig(prefix, rank, x3D, y3D, z3D, Sc, Sf, Sw, Ngrid, Boxsize):
    """Save density in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.
    x3D, y3D, z3D : array
        Cartesian grid.
    Sc, Sf, Sw : array
        Signature values.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.
    """
    fname = prefix + str(rank) + ".npz"
    np.savez(fname, Ngrid=Ngrid, Boxsize=Boxsize, x3D=x3D, y3D=y3D, z3D=z3D, Sc=Sc, Sf=Sf, Sw=Sw)


def load_nexus_sig(prefix, rank):
    """Save density in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.
    x3D, y3D, z3D : array
        Cartesian grid.
    Sc, Sf, Sw : array
        Signature values.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.
    """
    fname = prefix + str(rank) + ".npz"
    data = np.load(fname)
    Ngrid = data['Ngrid']
    Boxsize = data['Boxsize']
    x3D = data['x3D']
    y3D = data['y3D']
    z3D = data['z3D']
    Sc = data['Sc']
    Sf = data['Sf']
    Sw = data['Sw']
    return Ngrid, Boxsize, x3D, y3D, z3D, Sc, Sf, Sw
