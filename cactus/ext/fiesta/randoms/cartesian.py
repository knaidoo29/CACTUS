import numpy as np


def random_uniform(size, xmin, xmax):
    """Generates uniform randoms along one axis.

    Parameters
    ----------
    size : int
        Number of randoms.
    xmin : float
        Minimum X for the box.
    xmax : float
        Maximum X for the box.

    Returns
    -------
    x : array
        Random x-values.
    """
    x = np.random.random_sample(size)
    x *= xmax - xmin
    x += xmin
    return x


def random_box(size, xmin, xmax, ymin, ymax):
    """Generates random particles inside a box.

    Parameters
    ----------
    size : int
        Number of randoms.
    xmin : float
        Minimum X for the box.
    xmax : float
        Maximum X for the box.
    ymin : float
        Minimum Y for the box.
    ymax : float
        Maximum Y for the box.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    """
    x = random_uniform(size, xmin, xmax)
    y = random_uniform(size, ymin, ymax)
    return x, y


def random_cube(size, xmin, xmax, ymin, ymax, zmin, zmax):
    """Generates random particles inside a box.

    Parameters
    ----------
    size : int
        Number of randoms.
    xmin : float
        Minimum X for the box.
    xmax : float
        Maximum X for the box.
    ymin : float
        Minimum Y for the box.
    ymax : float
        Maximum Y for the box.
    zmin : float
        Minimum Z for the box.
    zmax : float
        Maximum Z for the box.

    Returns
    -------
    x : array
        Random x-values.
    y : array
        Random y-values.
    z : array
        Random z-values.
    """
    x = random_uniform(size, xmin, xmax)
    y = random_uniform(size, ymin, ymax)
    z = random_uniform(size, zmin, zmax)
    return x, y, z
