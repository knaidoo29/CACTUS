
magpie.pixels.get_box
^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.get_box(xmin, xmax, ymin, ymax, zmin, zmax, steps=120, return_nearest=False, center=[0., 0., 0.])

    Returns 3D coordinates for a box.

    :Parameters:
      xmin : float
          Minimum x value.
      xmax : float
          Maximum x value.
      ymin : float
          Minimum y value.
      ymax : float
          Maximum y value.
      zmin : float
          Minimum z value.
      zmax : float
          Maximum z value.
      steps : int, optional
          Number of divisions in each line element.
      return_nearest : bool, optional
          If True only the lines for the nearest or visible faces from a defined
          center will be outputed.
      center : list, optional
          Coordinate center used for return_nearest=True

    :Returns:
      x, y, z : array
          Coordinates of the box edges.
