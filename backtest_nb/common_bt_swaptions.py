import copy
import pandas as pd
from dateutil.parser import parse
from ..backtest_nb.common_bt import CommonBT
from ..backtest_nb.common_bt_future_options_config import *
from ..backtest.indicator import Indicator
from ..infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateCitiDataSource
from ..infrastructure.forward_rate_data_container import ForwardRateRequest, ForwardRateCitiDataSource
from ..infrastructure.rate_vol_data_container import RateVolRequest, RateVolCitiDataSource
from ..backtest.strategies.swaptions_daily import SwaptionsDailyState, SwaptionsDaily
from ..backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ..tradable.forwardstartswap import ForwardStartSwap
from ..tradable.portfolio import Portfolio
from ..tradable.swaption import Swaption


class CommonBTSwaption(CommonBT):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False, cache_market_data=False, data_cache_path=None, logging=False, log_file=""):
        super().__init__(start_date, end_date, currency, parameters, force_run=force_run, cache_market_data=cache_market_data, data_cache_path=data_cache_path, logging=logging, log_file=log_file)
        self.valid()
        self.init()

    def valid(self):
        name_list = ["currency", "asset", "legs"]
        for name in name_list:
            if name not in self.parameters:
                raise Exception(f"Not found {name} in parameters")

    def init(self):
        currency = self.parameters["currency"]
        if "calendar"in self.parameters:
            calendar = self.parameters["calendar"] if isinstance(self.parameters["calendar"], list) else [self.parameters["calendar"]]
        else:
            exchange_code_map = {"USD": "XCME"}
            if currency == "USD":
                calendar = ["NYC", "LON"]
                if currency in exchange_code_map:
                    calendar.append(exchange_code_map[currency])
            else:
                # just add currency into calendar as strategy recognize holidays from currency
                calendar = [currency]

        curve_type, tenor = self.parameters["asset"].upper().split(".")
        missing_vol_dates = ['2018-12-05', '2020-07-03', '2021-12-24']
        missing_dates = [parse(x) for x in missing_vol_dates]
        calendar += missing_dates
        LIBOR_CUTOFF_DATE = datetime(2019, 6, 27)
        if self.start_date < LIBOR_CUTOFF_DATE < self.end_date and curve_type == "SWAP_SOFR":
            fwd_rates_request = [
                [ForwardRateRequest(self.start_date, self.end_date, currency, "SWAP_LIBOR"), ForwardRateRequest(self.start_date, self.end_date, currency, curve_type)],
                [LIBOR_CUTOFF_DATE, LIBOR_CUTOFF_DATE]
            ]
        elif self.start_date < LIBOR_CUTOFF_DATE < self.end_date and curve_type == "SWAP_EUROSTR":
            fwd_rates_request = [
                [ForwardRateRequest(self.start_date, self.end_date, currency, "SWAP_EONIA"), ForwardRateRequest(self.start_date, self.end_date, currency, curve_type)],
                [LIBOR_CUTOFF_DATE, LIBOR_CUTOFF_DATE]
            ]
        else:
            fwd_rates_request = ForwardRateRequest(self.start_date, self.end_date, currency, curve_type)
        data_requests = {
            "vol_cube": (
                RateVolRequest(self.start_date, self.end_date, currency),
                RateVolCitiDataSource(),
            ),
            "fwd_rates": (
                fwd_rates_request,
                ForwardRateCitiDataSource(),
            ),
            "spot_rates": (
                SpotRateRequest(self.start_date, self.end_date, currency, curve_type),
                SpotRateCitiDataSource(),
            ),
        }

        parameters = dict()
        supported_strike_types = ["atmf", "atms", "delta"]
        parameters["legs"] = copy.deepcopy(self.parameters["legs"])
        for leg_name, leg in parameters["legs"].items():
            if leg["strike_type"] not in supported_strike_types:
                raise Exception(f"Found unsupported strike type {leg['strike_type']}. Only support: {','.join(supported_strike_types)}")
            leg["strike_type"] = "forward" if leg["strike_type"] == "atmf" else ("spot" if leg["strike_type"] == "atms" else leg["strike_type"])
            leg["currency"] = currency
            # payer defined as call on yield, receiver defined as put on yield
            leg["style"] = "Payer" if leg["type"].upper() == "C" else ("Receiver" if leg["type"].upper() == "P" else leg["type"])
            leg["curve_type"] = curve_type
            leg["tenor"] = tenor
            leg["delta_hedge"] = "Daily" if leg["hedge"] else "Unhedged"
            leg["tranche"] = CommonBT.create_tranche(leg["tranche"])
        parameters["df_rates_type"] = curve_type
        parameters["trade_first_day"] = self.parameters.get("trade_first_day", True)
        parameters["mtm"] = True
        parameters["use_delta_hedge_path"] = True
        # optional parameter
        optional_params = ["tc_rate", "cost_info_types"]
        for name in optional_params:
            if name in self.parameters:
                parameters[name] = self.parameters[name]

        if "tc_factor" in self.parameters and self.parameters["tc_factor"] != 0:
            parameters["tc_rate"] = {
                Swaption: {'vega': self.parameters["tc_factor"]},
                ForwardStartSwap: {'delta': self.parameters["tc_factor"]},
            }
        self.strategy = SwaptionsDaily(start_date=self.start_date, end_date=self.end_date, calendar=calendar, currency=self.currency,
                                       parameters=parameters, data_requests=data_requests, force_run=self.force_run, cache_market_data=self.cache_market_data,
                                       data_cache_path=self.data_cache_path, logging=self.logging, log_file=self.log_file)
        self.initial_state = SwaptionsDailyState(self.start_date, Portfolio([]), 0.0, 0.0, None)

    def get_pfo_df(self):
        if self.pfo_df is None:
            res = []
            for state in self.results:
                for leg_name, leg_pfo in state.portfolio.root.items():
                    if isinstance(leg_pfo, Portfolio):
                        for pfo_name, pfo in leg_pfo.root.items():
                            assert isinstance(leg_pfo, Portfolio)
                            for pos_name, pos in pfo.root.items():
                                if pos_name == "delta_hedge":
                                    for hedge_pos_name, hedge_pos in pos.root.items():
                                        tmp = [state.time_stamp, leg_name, pos_name, hedge_pos.tradable.name(), hedge_pos.quantity, hedge_pos.price, hedge_pos.quantity * hedge_pos.price, hedge_pos.delta, hedge_pos.vega, hedge_pos.theta]
                                        res.append(tmp)
                                else:
                                    tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(), pos.quantity, pos.price, pos.quantity * pos.price, pos.delta, pos.vega, pos.theta]
                                    res.append(tmp)
                    else:
                        assert "Cash" in leg_name
                        pfo_name = "Cash"
                        pos = leg_pfo
                        delta = 0
                        vega = 0
                        theta = 0
                        tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(), pos.quantity, pos.price, pos.quantity * pos.price, delta, vega, theta]
                        res.append(tmp)

            self.pfo_df = pd.DataFrame(res, columns=["dt", "leg", "pfo", "trd", "qty", "px", "price", "delta", "vega", "theta"])
        return self.pfo_df


if __name__ == "__main__":
    from ..tools.test_utils import get_temp_folder
    import os

    st = datetime(2024, 11, 4)
    # st = datetime(2017, 9, 28)
    et = datetime(2024, 11, 15)
    ccy_list = ["EUR"]
    for ccy in ccy_list:
        currency = ccy
        asset = "SWAP_EUROSTR.1Y"
        # asset = "SWAP_LIBOR.1Y"
        print(f"running: {currency}.{asset}")
        vega_size = -1
        params = dict(currency=currency, asset=asset)
        params["legs"] = {
            "call": {
                "type": "Payer", "strike_type": "atmf", "strike": 0, "expiry": "1M",
                "sizing_measure": "vega", "sizing_target": vega_size, "hedge": True,
                "tranche": RollingAtExpiryTranche(),
                "unwind": Indicator(name="deltapct", one_range=(0, 0.02)),
            },
        }
        data_cache_path = os.path.join(get_temp_folder(), "common_bt_swaption")
        # shutil.rmtree(data_cache_path)
        bt_base = CommonBTSwaption(start_date=st, end_date=et, currency=currency, parameters=params, force_run=False,
                                   cache_market_data=False, data_cache_path=data_cache_path)
        try:
            bt_base.run()
            print(bt_base.get_nav())
        except:
            raise