import numpy as np


def shift_array(
    arr: np.ndarray,
    num: np.ndarray,
    fill_value: float = 0.
) -> np.ndarray:
    """
    The equivalent of pd.shift, but for numpy arrays.
    """
    result = arr.copy()
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]

    return result


def round_to_nearest(num, base=None):
    if base is None:
        return np.round(num)
    else:
        return base * np.round(num/base)


def floor_to_nearest(num, base=None):
    if base is None:
        return np.floor(num)
    else:
        return base * np.floor(num/base)


def ceil_to_nearest(num, base=None):
    if base is None:
        return np.ceil(num)
    else:
        return base * np.ceil(num/base)
