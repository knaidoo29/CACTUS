
magpie.pixels.pos2pix_polarEA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.pos2pix_polarEA(phi, r, nphi, nr, base_nphi=3, rmax=1.)

    Angular and radial coordinates to polar equal area grid.

    :Parameters:
      phi, r : float or array
          Angular and radial coordinates.
      nr : int
          Grid size along the radial axes.
      base_nphi : int, optional
          Number of divisions in the angular axes across the first ring.
      rmax : float, optional
          Maximum radius, default=1.

    :Returns:
      pixID : int or array
          The pixel index the points are located in the polar coordinate grid.
