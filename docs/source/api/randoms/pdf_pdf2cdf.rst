
magpie.randoms.pdf2cdf
^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.randoms.pdf2cdf(xmid, pdf, return_normpdf=True)

    Calculates the CDF from a given PDF.

    :Parameters:
      xmid : array
          Linearly spaced x-values given at the middle of a bin of length dx.
      pdf : array
          Probabilty distribution function.
      return_normpdf : bool, optional
          Normalise PDF is also outputed.

    :Returns:
      x : array
          X-coordinates.
      cdf : array
          Cumulative distribution function with extreme points set 0 and 1.
      normpdf : array, optional
          Normalised PDF.
