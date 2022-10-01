
magpie.pixels.pix2pos_cart3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.pix2pos_cart3d(pixID, lengths, ngrids, origin=[0., 0., 0.], return1d_pixID=False)

    Returns the 3D pixel coordinate from a pixel index.

    :Parameters:
      pixID : int or array
          3D pixel index.
      lengths : float or list[float]
          Length of the grid.
      ngrids : int or list[int]
          Number of grid points.
      origin : list[float], optional
          Minimum values of the grid along each axis.
      return1d_pixID : bool, optional
          Returns 1d pixel index.

    :Returns:
      x, y, z : float or array
          Mid-point of the pixel points.
      dx, dy, dz : float
          Pixel width.
      xpixID, ypixID, zpixID : int or array
          1D pixel index.
