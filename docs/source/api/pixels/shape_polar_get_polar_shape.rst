
magpie.pixels.get_polar_shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.get_polar_shape(pixID, nphi, nr, rmin=0., rmax=1., phimin=0., phimax=2.*np.pi, proj='cart', center=[0., 0.], steps=12, returnpoly=False)

    Returns a polygon for the shape of a single polar coordinate pixel in
    polar or cartesian 2D coordinates.

    :Parameters:
      pixID : int
          Pixel index in the polar grid.
      nphi, nr : int
          Grid dimensions along the angular and radial axis.
      rmin, rmax : float, optional
          Radial ranges for the polar grid, default rmin=0 and rmax=1.
      phimin, phimax : float, optional
          Angular ranges for the polar grid, default phimin=0 and phimax=2pi.
      proj : str, optional
          Output coordinate projection. Either 'cart' for cartesian or 'polar'
          for polar projection.
      center : list, optional
          Center point of the polar grid, only relevant for cartesian
          projection, default at the origin.
      steps : int
          Number of steps in the polygon, default=12.
      returnpoly : bool, optional
          Return matplotlib polygon object, default=False.

    :Returns:
      x, y or p, r : array
          Cartesian or polar coordinate elements for the pixel polygon.
