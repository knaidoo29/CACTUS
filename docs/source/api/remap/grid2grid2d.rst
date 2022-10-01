
magpie.remap.grid2grid2d
^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.remap.grid2grid2d(f, boxsize, ngridout, origin=0., originout=0., boxsizeout=None)

    Remaps a dataset defined on a uniform 2D cartesian grid onto a new
    uniform 2D cartesian grid.

    :Parameters:
      f : 2darray
          The values of the 2D grid pixels.
      boxsize : float or list[float]
          Box size or list of length along each axis.
      ngridout : int or list[int]
          Grid division of the output grid along one axis or a list of the grid
          divisions along each axis.
      origin : float or list[float], optional
          Origin of the grid. If all axes begin at the same origin this can be a
          scalar, if you instead wish to specify different origins for each axis
          this should be added as a list.
      originout : float or list[float], optional
          Origin of the output grid. If all axes begin at the same origin this
          can be a scalar, if you instead wish to specify different origins for
          each axis this should be added as a list.
      boxsizeout : float or list[float], optional
          Box size or list of length along each axis for the output grid.

    :Returns:
      fout : 2darray
          The resampled pixel values for the output 2D cartesian grid.
