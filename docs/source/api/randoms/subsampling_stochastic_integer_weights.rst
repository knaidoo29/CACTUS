
magpie.randoms.stochastic_integer_weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.randoms.stochastic_integer_weights(weights)

    Returns stochastic integer weights for an input weight. This is useful for
    point processes that require integer weights, where a non-integer weight can
    be achieved by superposition of many realisations.

    :Parameters:
      weights : array
          Input weights.

    :Returns:
      weights_SI : array
          Stochastic integer weights.
