
magpie.pixels.pix2pos_polar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.pix2pos_polar(pixID, nphi, nr, rmin=0., rmax=1., phimin=0., phimax=2.*np.pi, return1d_pixID=False)

    Returns the 2D pixel coordinate from a pixel index.

    :Parameters:
      pixID : int or array
          2D pixel index.
      nphi : int
          Number of divisions along the p-axis.
      nr : int
          Number of divisions along the r-axis.
      rmin : float, optional
          Minimum radial value, default=0.
      rmax : float, optional
          Maximum radial value, default=1.
      phimin : float, optional
          Minimum phi value, default=0.
      phimax : float, optional
          Maximum phi value, default=2pi.
      return1d_pixID : bool, optional
          Returns 1d pixel index.

    :Returns:
      phi, r : float or array
          Mid-point of the pixel points.
      dphi, dr : float
          Pixel width.
      phipixID, rpixID : int or array
          1D pixel index.
