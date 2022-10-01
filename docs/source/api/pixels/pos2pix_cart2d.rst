
magpie.pixels.pos2pix_cart2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.pos2pix_cart2d(x, y, lengths, ngrids, origin=[0., 0.])

    X and Y-coordinate to 2D pixel.

    :Parameters:
      x, y : float or array
          X and Y-coordinate.
      lengths : float or list[float]
          Lengths of the 2D grids.
      ngrids : int or list[int]
          The grid dimension.
      origin : list[float], optional
          Minimum value along the grids.

    :Returns:
      pixID : int or array
          The index of the pixel the points are located in 2D.
