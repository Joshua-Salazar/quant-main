

def pretty_print_dict(d, sort_keys=True, indent=''):
    '''
    Function to 'pretty print' a dictionary
    @param d: dictionary to print
    @param sort_keys: True or False
    @param indent: whether to indent the keys in the string
    @return: string representing the dictionary
    '''

    out = ''
    keys = d.keys()
    if sort_keys:
        keys = sorted(keys)
    keys_that_are_dicts = []
    keys_that_arent_dicts = []
    for k in keys:
        if type(d[k]) == dict:
            keys_that_are_dicts.append(k)
        else:
            keys_that_arent_dicts.append(k)
    for k in keys_that_arent_dicts+keys_that_are_dicts:
        if type(d[k]) == dict:
            out += indent + k + ':\n'
            out += pretty_print_dict(d[k], sort_keys=sort_keys, indent=indent+'  ')
        elif d[k] is None:
            out += indent + k + ': null\n'
        elif type(d[k]) == tuple:
            out += indent + k + ': ' + str(d[k]).replace('(', '[').replace(')', ']') + '\n'
        else:
            out += indent + k + ': ' + str(d[k]) + '\n'
    return out
