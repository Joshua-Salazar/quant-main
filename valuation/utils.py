def find_fx_for_tradable(market, tradable, currency):
    if currency is None:
        fx = 1.0
    else:
        if tradable.currency == currency:
            fx = 1.0
        else:
            fx_pair_name = f'{tradable.currency}{currency}'
            fx = market.get_fx_spot(fx_pair_name)
    return fx


def return_valuer_res(calc_types, values, exclude_scaling_cols, contract_size, fx):
    calc_types_list = calc_types if isinstance(calc_types, list) else [calc_types]
    res = []
    calc_res_map = dict(zip(calc_types_list, values))
    for calc_type, x in calc_res_map.items():
        if calc_type in exclude_scaling_cols:
            res.append(x)
        elif "numerical#" in calc_type:
            res.append({k: v if any([skip_greek in k for skip_greek in ["SpotRef#", "VolRef#"]]) else v * contract_size * fx for k, v in calc_res_map[calc_type].items()})
        else:
            res.append(x * contract_size * fx)
    return res if isinstance(calc_types, list) else res[0]
