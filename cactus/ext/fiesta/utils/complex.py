import numpy as np


def complex_mult(complex_array, factors):
    """Complex array multiplication.

    Parameters
    ----------
    complex_array : array
        Complex array.
    factors : float/array
        Factors to be multiplied by.
    """
    complex_array.real *= factors
    complex_array.imag *= factors
    return complex_array


def complex_div(complex_array, factors):
    """Complex array division.

    Parameters
    ----------
    complex_array : array
        Complex array.
    factors : float/array
        Factors to be dividied by.
    """
    complex_array.real /= factors
    complex_array.imag /= factors
    return complex_array
