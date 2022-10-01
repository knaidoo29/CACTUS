
magpie.pixels.get_polarEA_shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.get_polarEA_shape(pixID, nr, rmax=1., base_nphi=3, proj='cart', center=[0., 0.], steps=12, returnpoly=False)

    Returns a polygon for the shape of a single polar coordinate pixel in
    polar or cartesian 2D coordinates.

    :Parameters:
      pixID : int
          Pixel index.
      nr : int
          Number of divisions along the r-axis.
      rmax : float, optional
          Maximum radial value, default=1.
      base_nphi : int, optional
          Number of pixels around r=0, default=3.
      proj : str, optional
          Output coordinate projection. Either 'cart' for cartesian or 'polar'
          for polar projection.
      center : list, optional
          Center point of the polar grid, only relevant for cartesian
          projection, default at the origin.
      steps : int, optional
          Number of steps in the polygon, default=12.
      returnpoly : bool, optional
          Return matplotlib polygon object, default=False.

    :Returns:
      x, y or p, r : array
          Cartesian or polar coordinate elements for the pixel polygon.
