
magpie.remap.grid2grid1d
^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.remap.grid2grid1d(f, boxsize, ngridout, origin=0., originout=0., boxsizeout=None)

    Remaps a dataset defined on 1D a uniform cartesian grid onto a new 1D
    uniform cartesian grid.

    :Parameters:
      f : array
          The values of the 1D grid pixels.
      boxsize : float
          Box size.
      ngridout : int
          Grid division of the output grid.
      origin : float, optional
          Origin of the grid.
      originout : float, optional
          Origin of the output grid.
      boxsizeout : float
          Box size of the output grid.

    :Returns:
      fout : array
          The remapped pixel values for the output 1D grid.
