

def flatten_list(list_):
    """Flattens input list

    Parameters
    ----------
    list_ : list
        Unflattened list.

    Returns
    -------
    flat_list : list
        Flattened list.
    """
    flat_list = [item for sublist in list_ for item in sublist]
    return flat_list
