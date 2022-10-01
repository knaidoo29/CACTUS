
magpie.randoms.randoms_cdf
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.randoms.randoms_cdf(x, cdf, size, kind='cubic')

    Generates randoms from a given cumulative distribution function.

    :Parameters:
      x : array
          X-coordinates.
      cdf : array
          Cumulative distribution function, extreme points must be 0 and 1 i.e.
          cdf[0] = 0 and cdf[-1] = 1.
      size : int
          Size of the random sample.
      kind : str, optional
          Scipy CDF interpolation kind.

    :Returns:
      rands : array
          Randoms drawn from sample CDF.
