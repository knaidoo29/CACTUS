
magpie.randoms.randoms_pdf
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.randoms.randoms_pdf(x, pdf, size, kind='cubic')

    Generates randoms from a given probability distribution function by first
    calculating a CDF.

    :Parameters:
      xmid : array
          Linearly spaced x-values given at the middle of a bin of length dx.
      pdf : array
          Probabilty distribution function.
      size : int
          Size of the random sample.
      kind : str, optional
          Scipy CDF interpolation kind.

    :Returns:
      rands : array
          Randoms drawn from sample PDF.
