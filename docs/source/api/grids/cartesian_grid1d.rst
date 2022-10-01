
magpie.grids.grid1d
^^^^^^^^^^^^^^^^^^^

.. function:: magpie.grids.grid1d(length, ngrid, origin=0., return_edges=False)

    Returns the edges of a uniform grid along one axis.

    :Parameters:
      Length : float
          Length of the grid.
      ngrid : int
          Minimum value of the grid.
      origin : float, optional
          Minimum value of the grid.
      return_edges : bool, optional
          If True then the edges of the uniform grid are also supplied.

    :Returns:
      xmid : array
          The mid-point of a uniform grid.
      xedges : array, optional
          The edges of a uniform grid along one axis.
