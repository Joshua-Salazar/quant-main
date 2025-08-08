import pandas as pd
import datetime
import math


def uniq(x):
    if all([math.isnan(_x) for _x in list(x)]):
        return list(x)[0]
    else:
        _x = list(set(list(x)))
        assert len(_x) == 1, _x
        return _x[0]


def make_tuple(x, size=None, pad=None):
    # extend to cover list type, i.e. cast to tuple
    t = x if type(x) is tuple else (tuple(x) if type(x) is list else (x,))
    if size is None or len(t) == size:
        return t
    elif len(t) < size:
        if pad is None:
            pad = t[-1]
        for i in range(size - len(t)):
            t = t + (pad,)
        return t
    else:
        raise RuntimeError('The size of the input is larger than the specified size')


def format_number(value, precision):
    return ('{:.' + str(precision) + 'f}').format(value)


def format_percentage(value, precision):
    return ('{:.' + str(precision) + 'f}%').format(value * 100)


def lower_case_and_underscore(input: str):
    return input.lower().replace(" ", "_")


def format_isodate(dt):
    if isinstance(dt, datetime.date) or isinstance(dt, datetime.datetime):
        dt = dt.strftime("%Y-%m-%d")
    return dt

def merge_dict(a, b, path=None, in_place=False):
    "merges b into a"
    if not in_place:
        return merge_dict(dict(a), b, path=path, in_place=True)
    else:
        if path is None: path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    merge_dict(a[key], b[key], path + [str(key)], in_place=True)
                # elif isinstance(a[key], pd.DataFrame) and isinstance(b[key], pd.DataFrame):
                #     a[key] = pd.concat([a[key], b[key]])
                elif isinstance(a[key], list) and isinstance(b[key], list):
                    a[key] = a[key] + b[key]
                elif a[key] == b[key]:
                    pass # same leaf value
                else:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]
        return a


def dict_of_dict_to_dataframe(dictionary, key_column_name=None):
    df = pd.DataFrame(data=list(dictionary.values()), index=list(dictionary.keys()))
    if key_column_name is None:
        return df
    else:
        return df.reset_index().rename(columns={'index': key_column_name})


def dataframe_to_records(df):
    cols = list(df.columns)
    v = df.values
    records = []
    for row in range(df.shape[0]):
        item = dict(zip(cols, v[row]))
        records.append(item)
    return records
