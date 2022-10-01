
magpie.pixels.pos2pix_cart3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.pos2pix_cart3d(x, y, z, lengths, ngrids, origin=[0., 0., 0.])

    X, Y and Z-coordinate to 3D pixel.

    :Parameters:
      x, y, z : float or array
          X, Y and Z-coordinate.
      lengths : float or list[float]
          Lengths of the 3D grids.
      ngrids : int or list[int]
          The grid dimension.
      origin : list[float], optional
          Minimum value along the grids.

    :Returns:
      pixID : int or array
          The index of the pixel the points are located in 3D.
