
magpie.pixels.pix2pos_polarEA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.pix2pos_polarEA(pixID, nr, rmax=1., base_nphi=3)

    Returns the pixel coordinate on an equal area polar grid.

    :Parameters:
      pixID : int or array
          Pixel index.
      nr : int
          Number of divisions along the r-axis.
      rmax : float, optional
          Maximum radial value, default=1.
      base_nphi : int, optional
          Number of pixels in the first ring, default=3.

    :Returns:
      p, r : float or array
          Angular and radial mid points for each pixel in the equal area polar grid.
      dp, dr : float or array
          Pixel width.
