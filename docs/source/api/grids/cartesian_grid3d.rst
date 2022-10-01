
magpie.grids.grid3d
^^^^^^^^^^^^^^^^^^^

.. function:: magpie.grids.grid3d(lengths, ngrids, origin=[None, None, None], return1d=False)

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
      x3d, y3d, z3d : 3darray
          The uniform 3d grid.
      xmid, ymid, zmid : array, optional
          The mid-point of the uniform grid.
