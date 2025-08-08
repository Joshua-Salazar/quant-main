import copy
from datetime import datetime
import pandas as pd
from ..analytics.symbology import option_calendar_from_ticker
from ..backtest_nb.common_bt import CommonBT
from ..backtest_nb.common_bt_stock_options_daily_config import *
from ..backtest.indicator import Indicator
from ..backtest.functions.stock_options_daily import stock_options_daily
from ..backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ..tradable.portfolio import Portfolio


class CommonBTStockOptionsDaily(CommonBT):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False, cache_market_data=False,
                 data_cache_path=None, logging=False, log_file=""):
        super().__init__(start_date, end_date, currency, parameters, force_run=force_run,
                         cache_market_data=cache_market_data, data_cache_path=data_cache_path, logging=logging,
                         log_file=log_file)
        self.valid()
        self.init()

    def handle_legacy_setting(self):
        asset = None
        if "legs" in self.parameters:
            for leg_name, leg in self.parameters["legs"].items():
                if "underlying" in leg:
                    if asset is None:
                        asset = leg["underlying"]
                    else:
                        if asset != leg["underlying"]:
                            raise Exception(f"Found multiple underlying: {asset}, {leg['underlying']}")
                else:
                    raise Exception(f"Not found underlying in leg {leg_name} for legacy setting")

            # setup asset
            self.parameters["asset"] = asset
        else:
            raise Exception(f"Not found legs in parameters")

    def handle_common_bt_setting(self):

        if "legs" in self.parameters:
            for leg_name, leg in self.parameters["legs"].items():
                if "tenor" not in leg:
                    if "expiry" in leg:
                        leg["tenor"] = leg["expiry"]
                    else:
                        raise Exception(f"Not found expiry in leg {leg_name} for legacy setting")
                if "selection_by" not in leg:
                    if "strike_type" in leg:
                        if leg["strike_type"] == "delta":
                            leg["selection_by"] = "delta"
                            if "strike" in leg:
                                leg["delta"] = leg["strike"]
                            else:
                                raise Exception(f"Not found delta or strike in leg {leg_name} for delta legacy setting")
                        elif leg["strike_type"] == "strike":
                            leg["selection_by"] = "strike"
                        else:
                            raise Exception(f"Unsupported selection_by: {leg['selection_by']}")
                # handle tranche setting later
                if "tranche" not in leg:
                    if "roll_style" in self.parameters:
                        if self.parameters["roll_style"] == "expiry":
                            leg["tranche"] = "RollingAtExpiryTranche"
                        else:
                            raise Exception(f"Only support roll_style: expiry, but found {self.parameters['roll_style']}")
                    else:
                        raise Exception(f"Not found roll_style in parameters for legacy setting")
                if 'underlying' not in leg:
                    leg["underlying"] = self.parameters["asset"]
        else:
            raise Exception(f"Not found legs in parameters")

    def valid(self):
        if "asset" in self.parameters:
            self.handle_common_bt_setting()
        else:
            self.handle_legacy_setting()

    def init(self):

        defaults = STOCK_OPTIONS_DAILY_DEFAULTS
        for leg_name, leg in self.parameters["legs"].items():
            leg["tranche"] = CommonBT.create_tranche(leg["tranche"])
        if "tc_factor" in self.parameters and self.parameters["tc_factor"] != 0:
            self.parameters["flat_vega_charge"] = {
                "tc_delta": self.parameters["tc_factor"],
                "tc_vega": self.parameters["tc_factor"]
            }
        self.strategy, self.initial_state = stock_options_daily(
            start_date=self.start_date,
            end_date=self.end_date,
            calendar=option_calendar_from_ticker(self.parameters["asset"]),
            legs=copy.deepcopy(self.parameters["legs"]),
            hedged=self.parameters.get("hedged", defaults["hedged"]),
            max_option_expiry_days=self.parameters.get("max_option_expiry_days", defaults["max_option_expiry_days"]),
            option_data_source=self.parameters.get("option_data_source", defaults["option_data_source"]),
            extra_data_requests=self.parameters.get("extra_data_requests", defaults["extra_data_requests"]),
            allow_fill_forward_missing_data=self.parameters.get("allow_fill_forward_missing_data",
                                                                defaults["allow_fill_forward_missing_data"]),
            prev_state=[],
            use_listed=self.parameters.get("use_listed", defaults["use_listed"]),
            cost_params=self.parameters.get("cost_params", defaults["cost_params"]),
            expiration_rules=self.parameters.get("expiration_rules", defaults["expiration_rules"]),
            keep_hedges_in_tranche_portfolio=self.parameters.get("keep_hedges_in_tranche_portfolio",
                                                                 defaults['keep_hedges_in_tranche_portfolio']),
            hedge_future_expiry_at_option_expiry=self.parameters.get("hedge_future_expiry_at_option_expiry",
                                                                     defaults['hedge_future_expiry_at_option_expiry']),
            number_of_futures_to_load=self.parameters.get("number_of_futures_to_load",
                                                          defaults["number_of_futures_to_load"]),
            trade_first_day=self.parameters.get("trade_first_day", defaults["trade_first_day"]),
            greeks_to_include=self.parameters.get("greeks_to_include", defaults['greeks_to_include']),
            inc_greeks=self.parameters.get("inc_greeks", defaults['inc_greeks']),
            scale_by_nav=self.parameters.get("scale_by_nav", defaults['scale_by_nav']),
            data_start_date_shift=self.parameters.get("data_start_date_shift", defaults['data_start_date_shift']),
            allow_fix_option_price_from_settlement=self.parameters.get("allow_fix_option_price_from_settlement",
                                                                       defaults['allow_fix_option_price_from_settlement']),
            allow_reprice=self.parameters.get("allow_reprice", defaults['allow_reprice']),
            inc_trd_dts=self.parameters.get("inc_trd_dts", defaults['inc_trd_dts']),
            return_strategy_and_initial_state=True
        )

    def get_pfo_df(self):
        if self.pfo_df is None:
            greeks_included = self.strategy.parameters.get('greeks_to_include')
            res = []
            for state in self.results:
                for leg_name, leg_pfo in state.portfolio.root.items():
                    if isinstance(leg_pfo, Portfolio):
                        for pfo_name, pfo in leg_pfo.root.items():
                            if isinstance(pfo, Portfolio):
                                for pos_name, pos in pfo.root.items():
                                    tmp_greeks = [getattr(pos, greek) for greek in greeks_included]
                                    tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(),
                                           pos.quantity, pos.price, pos.quantity * pos.price, *tmp_greeks]
                                    res.append(tmp)
                            else:
                                # ESH22, USDConstant
                                tmp = [state.time_stamp, leg_name, pfo_name, pfo.tradable.name(),
                                       pfo.quantity, pfo.price, pfo.quantity * pfo.price, *tmp_greeks]
                                res.append(tmp)
                    else:
                        assert "Cash" in leg_name
                        pfo_name = "Cash"
                        pos = leg_pfo
                        tmp_zero_greeks = [0] * len(greeks_included)
                        tmp = [state.time_stamp, leg_name, pfo_name, pos.tradable.name(),
                               pos.quantity, pos.price, pos.quantity * pos.price, *tmp_zero_greeks]
                        res.append(tmp)

            self.pfo_df = pd.DataFrame(res, columns=["dt", "leg", "pfo", "trd", "qty", "px", "price", *greeks_included])
        return self.pfo_df


if __name__ == "__main__":
    # Shaun: I should change this at some point
    st = datetime(2025, 1, 4)
    et = min(datetime.today(), datetime(2025, 4, 28))

    asset = "TY"
    ccy = "USD"
    # "sizing_measure": "vega", "sizing_target": vega_size,
    theta_size = -0.15
    vega_size = -.5
    hedge_size = 1 * .5

    vega_size_ = vega_size

    params = dict(asset=asset)
    params["legs"] = {

        "funding": {"type": "C", "strike_type": "delta", "strike": 0.40, "expiry": "1W", "fut_tgt_tenor": "45D", "fut_min": "45D",
                    "sizing_measure": "units", "sizing_target": -1, "hedge": True,
                    "tranche": RollingAtExpiryDailyTranche(),
                    "unwind": Indicator(name="delta", one_range=(1, 1)),
                    },

    }

    from ..backtest_nb.common_bt_factory import CommonBTFactory
    bt_v9 = CommonBTFactory().create(leg_type="future_option", start_date=st, end_date=et, currency=ccy, parameters=params, force_run=False)
    bt_v9.run()
    nav_v9 = bt_v9.get_nav()
    print(nav_v9)
