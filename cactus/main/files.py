import os


def check_exist(fname):
    """Checks file exist.

    Parameters
    ----------
    fname : str
        Filename.
    """
    return os.path.exists(fname)
