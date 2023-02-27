import numpy as np
import yaml
import pygadgetreader as pyg


from . import files


def read_paramfile(yaml_fname, MPI):
    """Read yaml parameter file.

    Parameters
    ----------
    yaml_fname : str
        Yaml parameter filename.
    MPI : object
        mpi4py object.

    Returns
    -------
    params : dict
        Parameter file dictionary.
    ERROR : bool
        Error tracker.
    """
    MPI.mpi_print_zero(" Reading parameter file: "+yaml_fname)
    if files.check_exist(yaml_fname) is False:
        MPI.mpi_print_zero(" ERROR: Yaml file does not exist, aborting.")
        ERROR = True
    else:
        ERROR = False
    if ERROR is False:
        if MPI.rank == 0:
            with open(yaml_fname, mode="r") as file:
                params = yaml.safe_load(file)
            MPI.send(params, to_rank=None, tag=11)
        else:
            params = MPI.recv(0, tag=11)
        MPI.wait()
        return params, ERROR
    else:
        return 0, ERROR


def read_ascii(fname, columns):
    """Read positions from ascii file.

    Parameters
    ----------
    fname : str
        Filename.
    columns : int list
        Columns for x, y, z positions.
    """
    data = np.loadtxt(fname, unpack=True)
    pos = np.column_stack([data[columns[0]], data[columns[1]], data[columns[2]]])
    return pos


def read_npz(fname, npz_keys):
    """Read positions from ascii file.

    Parameters
    ----------
    fname : str
        Filename.
    columns : int list
        Columns for x, y, z positions.
    """
    data = np.load(fname)
    pos = np.column_stack([data[npz_keys[0]], data[npz_keys[1]], data[npz_keys[2]]])
    return pos


def read_gadget(fname, return_pos=True, return_vel=False, return_pid=False, part='dm',
                single=1, suppress=1):
    """Reads Gadget snapshot file.

    Parameters
    ----------
    fname : str
        Gadget file name.
    return_pos : bool, optional
        Reads and outputs the positions from a GADGET file.
    return_vel : bool, optional
        Reads and outputs the velocities from a GADGET file.
    return_pid : bool, optional
        Reads and outputs the particle IDs from a GADGET file.
    part : str, optional
        Particle type, default set to 'dm' (dark matter).
    single : int, optional
        If 1 opens a single snapshot part, otherwise opens them all.
    suppress : int, optional
        Suppresses print statements from pygadgetreader.
    """
    if return_pos == True:
        pos = pyg.readsnap(fname, 'pos', part, single=single, suppress=suppress)
    if return_vel == True:
        vel = pyg.readsnap(fname, 'vel', part, single=single, suppress=suppress)
    if return_pid == True:
        pid = pyg.readsnap(fname, 'pid', part, single=single, suppress=suppress)
    if return_pid == True:
        if return_pos == True and return_vel == True:
            return [pos, vel, pid]
        elif return_pos == True and return_vel == False:
            return [pos, pid]
        elif return_pos == False and return_vel == True:
            return [vel, pid]
    else:
        if return_pos == True and return_vel == True:
            return [pos, vel]
        elif return_pos == True and return_vel == False:
            return pos
        elif return_pos == False and return_vel == True:
            return vel
