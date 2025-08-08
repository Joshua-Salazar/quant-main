

def assert_index_equal(x, y=None):
    ''''
    Confirm index of multiple dataframes are the same

    :param x:    pd.DataFrame or list of pd.DataFrame
    :param y:    pd.DataFrame (if x is not a list)
    '''

    # convert to list if two dataframes supplied
    if type(x) is not list:
        x = [x, y]

    # loop through list and confirm all indices match
    all_equal=True
    for i in range(1,len(x)):
        all_equal = (list(x[0].index) == list(x[i].index)) & all_equal

    return all_equal


def assert_columns_equal(x, y=None):
    ''''
    Confirm columns of multiple dataframes are the same

    :param x:    pd.DataFrame or list of pd.DataFrame
    :param y:    pd.DataFrame (if x is not a list)
    '''

    # convert to list if two dataframes supplied
    if type(x) is not list:
        x = [x, y]

    # loop through list and confirm all columns match
    all_equal=True
    for i in range(1,len(x)):
        all_equal = (list(x[0].columns) == list(x[i].columns)) & all_equal

    return all_equal


def assert_index_and_columns_equal(x, y=None):
    ''''
    Confirm index and columns of multiple dataframes are the same

    :param x:    pd.DataFrame or list of pd.DataFrame
    :param y:    pd.DataFrame (if x is not a list)
    '''

    # convert to list if two dataframes supplied
    if type(x) is not list:
        x = [x, y]

    # loop through list and confirm all columns match
    all_equal=True
    for i in range(1,len(x)):
        all_equal = (list(x[0].index) == list(x[i].index)) & \
                    (list(x[0].columns) == list(x[i].columns)) & \
                    all_equal

    return all_equal



