
magpie.randoms.stochastic_binary_weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.randoms.stochastic_binary_weights(weights)

    Returns stochastic binary integer weights for an input weight. This is
    useful for point processes that require binary integer weights, where a
    non-integer weight can be achieved by superposition of many realisations.

    :Parameters:
      weights : array
          Input weights.

    :Returns:
      weights_SB : array
          Stochastic binary weights.
