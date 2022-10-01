
magpie.pixels.get_disc
^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.get_disc(rmin=0., rmax=1., phimin=0., phimax=2.*np.pi, center=[0., 0.], steps=40)

    Returns the coordinates of a disc segment.

    :Parameters:
      rmin : float, optional
          Minimum radial value, default=0.
      rmax : float, optional
          Maximum radial value, default=1.
      phimin : float, optional
          Minimum phi value, default=0.
      phimax : float, optional
          Maximum phi value, default=2pi.
      center : list, optional
          Central x and y coordinates for the arc.
      steps : int, optional
          The number of points along the arc curve.


    :Returns:
      x, y : array
          Coordinates along the arc of a circle.
