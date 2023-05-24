

def is_pos_monotonic(arr):
    """Returns True if an array is positively monotonic.

    Parameters
    ----------
    arr : array
        Array.
    """
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
