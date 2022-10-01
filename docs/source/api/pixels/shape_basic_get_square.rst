
magpie.pixels.get_square
^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.get_square(xmin, xmax, ymin, ymax, steps=40)

    Returns the coordinates of the perimeter of a rectangle.


    :Parameters:
      xmin : float
          Minimum x-value.
      xmax : float
          Maximum x-value.
      ymin : float
          Minimum y-value.
      ymax : float
          Maximum y-value.
      steps : int, optional
          Number of divisions across each line segment, default = 4.


    :Returns:
      x, y : array
          Perimeter coordinates of a rectangle.
