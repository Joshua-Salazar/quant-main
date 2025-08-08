import copy
import pandas as pd
from ..backtest_nb.common_bt import CommonBT
from ..backtest.strategies.fx_options_daily import FXOptionsDaily, FXOptionsDailyState
from ..infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateCitiDataSource
from ..infrastructure.fx_sol_sabr_vol_data_container import FXSOLSABRVolDataSourceOnDemand, FXSOLSABRVolRequest
from ..tradable.portfolio import Portfolio


class CommonBTFXOptions(CommonBT):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False, cache_market_data=False, data_cache_path=None, logging=False, log_file=""):
        super().__init__(start_date, end_date, currency, parameters, force_run=force_run, cache_market_data=cache_market_data, data_cache_path=data_cache_path, logging=logging, log_file=log_file)
        self.valid()
        self.init()

    def valid(self):
        name_list = ["asset", "legs"]
        for name in name_list:
            if name not in self.parameters:
                raise Exception(f"Not found {name} in parameters")

    def init(self):
        asset = self.parameters["asset"]
        base_ccy = asset[:3]
        term_ccy = asset[3:]
        calendar = [base_ccy, term_ccy]
        curve_name = "SWAP_SOFR" if term_ccy == "USD" else ( "SWAP_EUROSTR" if term_ccy == "EUR" else "SWAP_OIS")
        data_requests = {
            "vol": (
                FXSOLSABRVolRequest(asset, self.start_date, self.end_date),
                FXSOLSABRVolDataSourceOnDemand()
            ),
            "rate_curve": (
                SpotRateRequest(self.start_date, self.end_date, term_ccy, curve_name),
                SpotRateCitiDataSource()
            ),
        }
        parameters = dict()
        supported_strike_types = ["delta"]
        parameters["legs"] = copy.deepcopy(self.parameters["legs"])
        for leg_name, leg in parameters["legs"].items():
            if leg["strike_type"] not in supported_strike_types:
                raise Exception(f"Found unsupported strike type {leg['strike_type']}. Only support: {','.join(supported_strike_types)}")
            leg["asset"] = asset
            leg["tranche"] = CommonBT.create_tranche(leg["tranche"])
        parameters["trade_first_day"] = self.parameters.get("trade_first_day", True)

        if "tc_factor" in self.parameters and self.parameters["tc_factor"] != 0:
            parameters["flat_costs"] = {
                "tc_delta": self.parameters["tc_factor"],
                "tc_vega": self.parameters["tc_factor"]
            }
        self.strategy = FXOptionsDaily(start_date=self.start_date, end_date=self.end_date, calendar=calendar, currency=self.currency,
                                       parameters=parameters, data_requests=data_requests, force_run=self.force_run, cache_market_data=self.cache_market_data,
                                       data_cache_path=self.data_cache_path, logging=self.logging, log_file=self.log_file)
        self.initial_state = FXOptionsDailyState(self.start_date, Portfolio([]), 0.0, 0.0)

    def get_pfo_df(self):
        if self.pfo_df is None:
            res = []
            for state in self.results:
                for leg_name, leg_pfo in state.portfolio.root.items():
                    if isinstance(leg_pfo, Portfolio):
                        for pfo_name, pfo in leg_pfo.root.items():
                            if isinstance(pfo, Portfolio):
                                for pos_name, pos in pfo.root.items():
                                    if pos_name == "delta_hedge":
                                        for hedge_pos_name, hedge_pos in pos.root.items():
                                            tmp = [state.time_stamp, leg_name, pos_name, hedge_pos.tradable.name(), hedge_pos.quantity, hedge_pos.price, hedge_pos.quantity * hedge_pos.price, pos.forward_delta, pos.vega, pos.theta]
                                            res.append(tmp)
                                    else:
                                        tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(), pos.quantity, pos.price, pos.quantity * pos.price, pos.forward_delta, pos.vega, pos.theta]
                                        res.append(tmp)
                            else:
                                tmp = [state.time_stamp, leg_name, pfo_name, pfo.tradable.name(), pfo.quantity, pfo.price, pfo.quantity * pfo.price, pos.forward_delta, pos.vega, pos.theta]
                                res.append(tmp)

                    else:
                        assert "Cash" in leg_name
                        pfo_name = "Cash"
                        pos = leg_pfo
                        forward_delta = 0
                        vega = 0
                        theta = 0
                        tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(), pos.quantity, pos.price, pos.quantity * pos.price, forward_delta, vega, theta]
                        res.append(tmp)

            self.pfo_df = pd.DataFrame(res, columns=["dt", "leg", "pfo", "trd", "qty", "px", "price", "forward_delta", "vega", "theta"])
        return self.pfo_df


if __name__ == "__main__":
    from datetime import datetime

    from ..backtest_nb.common_bt_factory import CommonBTFactory
    from ..tools.test_utils import get_temp_folder
    from shared.backtest.indicator import Indicator
    import os

    st = datetime(2022, 1, 4)
    et = datetime(2024, 12, 31)
    asset = "USDJPY"
    ccy = "JPY"
    asset = "AUDUSD"
    ccy = "USD"
    vega_size = -1
    params = dict(asset=asset)
    params["legs"] = {
        "call": {"type": "C", "strike_type": "delta", "strike": 0.25, "expiry": "1M",
                 "sizing_measure": "vega", "sizing_target": vega_size, "hedge": True,
                 "tranche": "RollingAtExpiryTranche",
                 "unwind": Indicator(name="delta", one_range=(0, 0.02)),
                 }
    }

    data_cache_path = os.path.join(get_temp_folder(), "common_bt_fx_option")
    # shutil.rmtree(data_cache_path)
    bt_base = CommonBTFactory().create(leg_type="fx_option", start_date=st, end_date=et, currency=ccy,
                                       parameters=params, force_run=False, cache_market_data=False,
                                       data_cache_path=data_cache_path).run()