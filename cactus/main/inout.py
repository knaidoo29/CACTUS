import copy

import h5py
import numpy as np


def save_dens(prefix, rank, dens, Ngrid, Boxsize, precision='single'):
    """Save density in NPZ format.

    Parameters
    ----------
    prefix : str
        Filename prefix.
    rank : int
        MPI node.
    dens : array
        Density values.
    Ngrid : int
        Integer grid.
    Boxsize : float
        Boxsize length.
    precision : str, optional
        Precision of saved data.
    """
    fname = prefix + str(rank) + ".npz"
    np.savez(fname, Ngrid=Ngrid, Boxsize=Boxsize, dens=dens.astype(precision))


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
    dens = data['dens']
    return dens, Ngrid, Boxsize



def save_nexus_sig(prefix, rank, Sc, Sf, Sw, Ngrid, Boxsize, precision='single'):
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
    precision : str, optional
        Precision of saved data.
    """
    fname = prefix + str(rank) + ".npz"
    np.savez(fname, Ngrid=Ngrid, Boxsize=Boxsize, Sc=Sc.astype(precision),
        Sf=Sf.astype(precision), Sw=Sw.astype(precision))


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
    Sc = data['Sc']
    Sf = data['Sf']
    Sw = data['Sw']
    return Ngrid, Boxsize, Sc, Sf, Sw


class OutputCosmicWeb(object):

    def __init__(self, prefix, rank, output_data) -> None:
        """OutputCosmicWeb class to handle writing of cosmic web flags.

        Args:
            prefix (str): Prefix of output filename.
            rank (str): MPI rank.
            output_data (arr): Array of data to write to file.
        """
        self.filename_prefix = prefix + str(rank)
        self.output_data = output_data

    def save_npz(self):
        """Saves output data in numpy binary format.
        """
        out_filename = self.filename_prefix + ".npz"
        np.savez(out_filename, web_flag=self.output_data)
        return None

    def save_hdf5(self):
        """Saves output data in HDF5 format.
        """
        out_filename = self.filename_prefix + ".hdf5"

        with h5py.File(out_filename, 'w') as f:
            f.create_dataset(name="web_flag", data=self.output_data)
        return None

    def save_cautun_nexus(self, header_bytes=1048, array_order='C'):
        """Saves output data in NEXUS+ binary format.
            Note: Modifies the segmentation flags match the NEXUS+ style
            of flag assignment, i.e. flag_id==1 is 'undefined'.

        Args:
            header_bytes (int, optional): Size of the header (in bytes)
                to be written to the binary file. Defaults to 1048.
            array_order (str, optional): Array order of the data in the
                file. Defaults to 'C'-order but 'F'ortran order is also
                accepted. The numpy write functions always write in 'C'
                order so when 'F' is supplied the output data array
                itself is modified to account for this.
        """
        out_filename = self.filename_prefix + ".MMF"

        # Prepare empty header array
        dtype = np.dtype(np.ushort)
        n_bytes = dtype.itemsize
        header_array = np.empty(header_bytes // n_bytes, dtype=dtype)
        header_array[:] = 256

        # Prepare the output data in the NEXUS+ flag format (flag_id==1
        # is 'undefined')
        cautun_nexus_data = copy.deepcopy(self.output_data)
        cautun_nexus_data[cautun_nexus_data > 0] += 1
        cautun_nexus_data = np.asarray(cautun_nexus_data, dtype=dtype)

        # Concatenate header array with output data
        output_data = np.concatenate(
            (header_array, cautun_nexus_data.reshape(-1)))

        # Switches row-column order of binary data. Has to be done
        # manually because numpy always writes in 'C'-order.
        if array_order == 'F':
            output_data = output_data.T

        # Write to file
        with open(out_filename, 'wb') as f:
            output_data.tofile(f)
        return None
