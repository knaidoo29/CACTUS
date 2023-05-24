import numpy as np


def x2points(x):
    """Column stacks input coordinates.

    Parameters
    ----------
    x : array
        X coordinates.

    Return
    ------
    points : 1darray
        Column stacked array.
    """
    if np.isscalar(x) == True:
        points = np.array([[x]])
    else:
        points = np.column_stack([x])
    return points


def points2x(points):
    """Unstacks input coordinates.

    Parameters
    ----------
    points : 2darray
        Column stacked array.

    Return
    ------
    x : array
        X coordinates.
    """
    x = points[:, 0]
    return x


def xy2points(x, y):
    """Column stacks input coordinates.

    Parameters
    ----------
    x : array
        X coordinates.
    y : array
        Y coordinates.

    Return
    ------
    points : 2darray
        Column stacked array.
    """
    if np.isscalar(x) == True:
        points = np.array([[x, y]])
    else:
        points = np.column_stack([x, y])
    return points


def points2xy(points):
    """Unstacks input coordinates.

    Parameters
    ----------
    points : 2darray
        Column stacked array.

    Return
    ------
    x : array
        X coordinates.
    y : array
        Y coordinates.
    """
    x, y = points[:, 0], points[:, 1]
    return x, y


def xyz2points(x, y, z):
    """Column stacks input coordinates.

    Parameters
    ----------
    x : array
        X coordinates.
    y : array
        Y coordinates.
    z : array
        Z coordinates.

    Return
    ------
    points : 2darray
        Column stacked array.
    """
    if np.isscalar(x) == True:
        points = np.array([[x, y, z]])
    else:
        points = np.column_stack([x, y, z])
    return points


def points2xyz(points):
    """Unstacks input coordinates.

    Parameters
    ----------
    points : 2darray
        Column stacked array.

    Return
    ------
    x : array
        X coordinates.
    y : array
        Y coordinates.
    z : array
        Z coordinates.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return x, y, z


def coord2points(xlist):
    """Column stacks input coordinates.

    Parameters
    ----------
    xlist : array
        List of coordinates and extra informations.

    Return
    ------
    data : 2darray
        Column stacked data array.
    """
    if np.isscalar(xlist[0]) == True:
        data = np.array([xlist])
    else:
        data = np.column_stack(xlist)
    return data
