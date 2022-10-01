
magpie.coords.ortho2cart
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.ortho2cart(x, y, r= 1., fill_value=np.nan)

    Orthographic projection to cartesian coordinates.

    :Parameters:
      x : array
          X value in the orthographic projection.
      y : array
          Y value in the orthographic projection.
      r : float
          Radius of the sphere.
      fill_value : float
          Fill values outside the sphere with this value.

    :Returns:
      z : array
          Returns the z value of the cartesian coordinates.
