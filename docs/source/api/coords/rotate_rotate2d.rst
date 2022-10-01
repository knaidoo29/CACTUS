
magpie.coords.rotate2d
^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.rotate2d(x, y, dphi, center=[0., 0.])

    Rotates points in 2D cartesian coordinates counter clockwise by dphi in
    radians.

    :Parameters:
      x, y : float or array
          Cartesian coordinates.
      dphi : float
          Counter-clockwise rotation.
      center : list[float], optional
          Center of rotation, default=[0., 0.].

    :Returns:
      xrot, yrot : float or array
          Rotated x and y coordinates.
