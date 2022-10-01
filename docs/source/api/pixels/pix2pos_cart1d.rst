
magpie.pixels.pix2pos_cart1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.pix2pos_cart1d(pixID, length, ngrid, origin=0.)

    Returns the 1D pixel coordinate from a pixel index.

    :Parameters:
      pixID : int or array
          1D pixel index.
      length : float
          Length of the grid.
      ngrid : int
          Number of grid points.
      origin : float, optional
          Minimum value of the grid.

    :Returns:
      x : float or array
          Mid-point of the pixel points.
      dx : float
          Pixel width.
