
magpie.coords.usphere_area
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.usphere_area(phi_min, phi_max, theta_min, theta_max)

    Returns the area for a 'square' segment of a unit sphere given in
    spherical coordinates phi, theta.

    :Parameters:
      phi_min : float
          Minimum latitudinal coordinate in radians (where phi lies [0, 2pi]).
      phi_max : float
          Maximum latitudinal coordinate in radians (where phi lies [0, 2pi]).
      theta_min : float
          Minimum longitudinal coordinate in radians (where theta lies [0, pi]).
      theta_max : float
          Maximum longitudinal coordinate in radians (where theta lies [0, pi]).

    :Returns:
      Area : float
          Area in square radians.
