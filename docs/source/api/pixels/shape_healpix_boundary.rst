
magpie.pixels.healpix_boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.healpix_boundary(p, nside, steps=40, reverse=False)

    Returns the boundary of a healpix pixel in healpix x and y coordinates,
    default given in clockwise directions.

    :Parameters:
      p : int
          Healpix pixel index.
      nside : int
          Healpix Nside.
      steps : int, optional
          Number of steps.
      reverse : bool, optional
          Reverse the order so the output is anti-clockwise.

    :Returns:
      x : array
          X-coordinates of the boundaries of the healpix pixel.
      y : array
          Y-coordinates of the boundaries of the healpix pixel.
