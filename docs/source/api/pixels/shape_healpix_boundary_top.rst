
magpie.pixels.healpix_boundary_top
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.healpix_boundary_top(p, nside, steps=20, reverse=False)

    Returns the top side of a healpix pixel in healpix x and y coordinates.

    :Parameters:
      p : int
          Healpix pixel index.
      nside : int
          Healpix Nside.
      steps : int, optional
          Number of steps for the top function.
      reverse : bool, optional
          Reverse the order so the output has descending x.

    :Returns:
      x : array
          X-coordinates of the top side of the healpix pixel.
      y : array
          Y-coordinates of the top side of the healpix pixel.
