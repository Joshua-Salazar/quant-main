import unittest
from datetime import datetime
from ...tools import test_utils
from ...risk.risk_engine import RiskEngine, RiskEngineConfig
from ...dates.utils import get_ny_timezone
from ...tradable.condvarianceswap import CondVarianceSwap
from ...tradable.varianceswap import VarianceSwap
from ...valuation.eq_condvarswap_nx_valuer import EQCondVarSwapNxValuer
from ...valuation.eq_varswap_nx_valuer import EQVarSwapNxValuer
from ...valuation.varianceswap_vola_valuer import VarianceSwapVolaValuer
import os
import pandas as pd


class TestRE(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRE, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_re")
        self.rebase = False

    def test_var_upvar(self):
        test_name = "test_var_upvar"
        base_dt = datetime(2025, 5, 23, 17, tzinfo=get_ny_timezone())
        inst_ids = [74664375, 74664374]
        greeks_list = ["price", "numerical#delta"]
        one_side = True
        model_param_overrides_sol = dict(
            sim_path=30000,
            calib_time_steps=500,
            pricing_time_steps=500,
            x_steps=200,
            exercise_probability_threshold=0.01,
            use_ctp_config=False,
        )
        valuer_map_override = {
            CondVarianceSwap: EQCondVarSwapNxValuer(override_model_params=model_param_overrides_sol),
            VarianceSwap: EQVarSwapNxValuer(override_model_params=model_param_overrides_sol),
        }

        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side, valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_expiried(self):
        test_name = "test_expiried"
        base_dt = datetime(2025, 6, 20, 17, tzinfo=get_ny_timezone())
        inst_ids = [64653967]
        greeks_list = ["price"]
        one_side = True
        model_param_overrides_sol = dict(
            sim_path=30000,
            calib_time_steps=500,
            pricing_time_steps=500,
            x_steps=200,
            exercise_probability_threshold=0.01,
            use_ctp_config=False,
        )
        valuer_map_override = {
            CondVarianceSwap: EQCondVarSwapNxValuer(override_model_params=model_param_overrides_sol),
        }

        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_var_greeks_before(self):
        test_name = "test_var_greeks_before"
        base_dt = datetime(2025, 6, 18, 17, tzinfo=get_ny_timezone())

        one_side = False
        same_vol_grid_in_delta = False
        sticky_strike_in_vega = False
        use_abs_strike = False
        inst_ids = [75137198]
        greeks_list = ["price", "numerical#delta", "numerical#vega", "numerical#gamma"]
        model_param_overrides_sol = dict(
            sim_path=30000,
            calib_time_steps=500,
            pricing_time_steps=500,
            x_steps=200,
            exercise_probability_threshold=0.01,
            use_ctp_config=False,
            use_abs_strike=use_abs_strike,
        )
        valuer_map_override = {
            VarianceSwap: EQVarSwapNxValuer(override_model_params=model_param_overrides_sol),
        }

        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  sticky_strike_in_vega=sticky_strike_in_vega, same_vol_grid_in_delta=same_vol_grid_in_delta,
                                  valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res_nx = re.run()
        actual_res_nx = actual_res_nx.transpose().drop_duplicates().transpose()

        # volar numerical greeks
        inst_ids = [76000227]
        greeks_list = ["price", "numerical#delta", "numerical#vega", "delta", "deltausd", "vega", "gamma", "gammausd", "numerical#gamma", ]
        valuer_map_override = {
            VarianceSwap: VarianceSwapVolaValuer(),
        }
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  sticky_strike_in_vega=sticky_strike_in_vega, same_vol_grid_in_delta=same_vol_grid_in_delta,
                                  valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res_volar = re.run()
        actual_res_volar = actual_res_volar.transpose().drop_duplicates().transpose()
        actual_res = pd.concat([actual_res_nx, actual_res_volar])
        actual_res = actual_res[["Delta#SPX Index", "DeltaUSD#SPX Index", "delta", "deltausd", "Vega#SPX Index", "vega", "Gamma#SPX Index", "gamma", "GammaUSD#SPX Index", "gammausd"]].transpose()
        print(actual_res)

        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_var_greeks(self):
        test_name = "test_var_greeks"
        base_dt = datetime(2025, 6, 18, 17, tzinfo=get_ny_timezone())

        one_side = False
        sticky_strike_in_vega = True
        same_vol_grid_in_delta = True
        use_abs_strike = True
        inst_ids = [75137198]
        greeks_list = ["price", "numerical#delta", "numerical#vega", "numerical#gamma"]
        model_param_overrides_sol = dict(
            sim_path=60000,
            calib_time_steps=500,
            pricing_time_steps=500,
            x_steps=200,
            exercise_probability_threshold=0.01,
            use_ctp_config=False,
            use_abs_strike=use_abs_strike,
        )
        valuer_map_override = {
            VarianceSwap: EQVarSwapNxValuer(override_model_params=model_param_overrides_sol),
        }

        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  sticky_strike_in_vega=sticky_strike_in_vega, same_vol_grid_in_delta=same_vol_grid_in_delta,
                                  valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res_nx = re.run()
        actual_res_nx = actual_res_nx.transpose().drop_duplicates().transpose()

        # volar numerical greeks
        inst_ids = [76000227]
        greeks_list = ["price", "numerical#delta", "numerical#vega", "delta", "deltausd", "vega", "gamma", "gammausd", "numerical#gamma", ]
        valuer_map_override = {
            VarianceSwap: VarianceSwapVolaValuer(),
        }
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  sticky_strike_in_vega=sticky_strike_in_vega, same_vol_grid_in_delta=same_vol_grid_in_delta,
                                  valuer_map_override=valuer_map_override)
        re = RiskEngine(config=config)
        actual_res_volar = re.run()
        actual_res_volar = actual_res_volar.transpose().drop_duplicates().transpose()
        actual_res = pd.concat([actual_res_nx, actual_res_volar])
        actual_res = actual_res[["Delta#SPX Index", "DeltaUSD#SPX Index", "delta", "deltausd", "Vega#SPX Index", "vega", "Gamma#SPX Index", "gamma", "GammaUSD#SPX Index", "gammausd"]].transpose()
        print(actual_res)

        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_duplicate_columns(self):
        test_name = "test_duplicate_columns"
        base_dt = datetime(2025, 6, 18, 17, tzinfo=get_ny_timezone())
        inst_ids = [74574110, 75932332]
        greeks_list = ["price", "numerical#delta", "numerical#gamma", "numerical#vega"]
        one_side = False
        model_param_overrides_sol = dict(
            sim_path=60000,
            calib_time_steps=500,
            pricing_time_steps=500,
            x_steps=200,
            exercise_probability_threshold=0.01,
            use_ctp_config=False,
            use_abs_strike=True,
        )
        valuer_map_override = {
            VarianceSwap: EQVarSwapNxValuer(override_model_params=model_param_overrides_sol),
        }
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side,
                                  valuer_map_override=valuer_map_override, sticky_strike_in_vega=True,
                                  same_vol_grid_in_delta=True)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_corr_matrix(self):
        test_name = "test_corr_matrix"
        inst_ids = [76095560]
        base_dt = datetime(2025, 6, 4, 17, 0, tzinfo=get_ny_timezone())
        greeks_list = ["price"]
        unds = ["SPX Index", "NDX Index", "RTY Index"]
        flat_corr = pd.DataFrame(0.96, index=unds, columns=unds)
        for und in unds:
            flat_corr.loc[und, und] = 1
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, flat_corr=flat_corr)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)
