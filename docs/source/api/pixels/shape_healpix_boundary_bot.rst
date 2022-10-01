
magpie.pixels.healpix_boundary_bot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.healpix_boundary_bot(p, nside, steps=20, reverse=False)

    Returns the bottom side of a healpix pixel in healpix x and y coordinates.

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
          X-coordinates of the bottom side of the healpix pixel.
      y : array
          Y-coordinates of the bottom side of the healpix pixel.
