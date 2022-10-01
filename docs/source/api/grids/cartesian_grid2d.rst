
magpie.grids.grid2d
^^^^^^^^^^^^^^^^^^^

.. function:: magpie.grids.grid2d(lengths, ngrids, origin=[None, None], return1d=False)

    Returns the mid-point of a uniform grid.

    :Parameters:
      lengths : float or list[float]
          Length of the grid.
      ngrids : int or list[int]
          Number of grid points.
      origin : list[float], optional
          Minimum values of the grid along each axis.
      return1d : bool, optional
          Returns 1d mid-points.

    :Returns:
      x2d, y2d : 2darray
          The uniform 2d grid.
      xmid, ymid : array, optional
          The mid-point of the uniform grid.
