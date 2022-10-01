
magpie.pixels.get_arc
^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.get_arc(radius, phimin=0., phimax=2.*np.pi, center=[0., 0.], steps=10)

    Returns the coordinates of the arc of a circle, default settings will
    return the coordinates along a complete circle.


    :Parameters:
      radius : float
          Radius of the circle.
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
