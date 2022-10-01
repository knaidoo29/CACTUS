
magpie.grids.polarEA_area
^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.grids.polarEA_area(nr, rmax=1., base_nphi=3)

    Returns the area for an equal area polar grid.

    :Parameters:
      nr : int
          Number of divisions along the r-axis.
      rmax : float, optional
          Maximum radial value, default=1.
      base_nphi : int, optional
          Number of pixels around r=0, default=3.

    :Returns:
      area : float
          The area of pixels in the equal area polar grid.
