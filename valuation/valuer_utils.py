def return_results_based_on_dictionary(calc_types, results, throw_if_not_calculated=False):
    if not isinstance(calc_types, list):
        calc_types = [calc_types]
    output = tuple()
    for item in calc_types:
        if item not in results and throw_if_not_calculated:
            raise RuntimeError(f'calc type {item} is not available')
        output = output + (results.get(item, None),)
    if len(output) == 1:
        output = output[0]
    return output
