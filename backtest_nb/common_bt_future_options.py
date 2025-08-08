import copy
import pandas as pd
from ..backtest_nb.common_bt import CommonBT
from ..backtest_nb.common_bt_future_options_config import *

from ..backtest.indicator import Indicator
from ..backtest.strategies.future_option_daily import FutureOptionState, FutureOptionDaily
from ..backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ..infrastructure.future_data_container import FutureDataRequest, IVOLFutureDataSource
from ..infrastructure.future_option_data_container import IVOLOptionDataRequest, IVOLOptionDataSource
from ..tradable.future import Future
from ..tradable.option import Option
from ..tradable.portfolio import Portfolio
from ..valuation.future_data_valuer import FutureDataValuer
from ..valuation.option_data_valuer import OptionDataValuer_Zero_Px


class CommonBTFutureOption(CommonBT):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False, cache_market_data=False, data_cache_path=None, logging=False, log_file=""):
        super().__init__(start_date, end_date, currency, parameters, force_run=force_run, cache_market_data=cache_market_data, data_cache_path=data_cache_path, logging=logging, log_file=log_file)
        self.valid()
        self.init()

    def handle_legacy_setting(self):
        asset = None
        if "legs" in self.parameters:
            for leg_name, leg in self.parameters["legs"].items():
                if "root" in leg:
                    if asset is None:
                        asset = leg["root"]
                    else:
                        if asset != leg["root"]:
                            raise Exception(f"Found multiple roots: {asset}, {leg['root']}")
                else:
                    raise Exception(f"Not found root in leg {leg_name} for legacy setting")
                if "expiry" not in leg:
                    if "target_option_tenor" in leg:
                        leg["expiry"] = leg["target_option_tenor"]
                    else:
                        raise Exception(f"Not found target_option_tenor in leg {leg_name} for legacy setting")
                if "strike_type" not in leg:
                    if "delta" in leg:
                        leg["strike_type"] = "delta"
                        leg["strike"] = leg["delta"]
                    else:
                        raise Exception(f"Not found delta in leg {leg_name} for legacy setting")
                if "tranche" not in leg:
                    if "roll_style" in self.parameters:
                        if self.parameters["roll_style"] == "expiry":
                            leg["tranche"] = "RollingAtExpiryTranche"
                        else:
                            raise Exception(f"Only support roll_style: expiry, but found {self.parameters['roll_style']}")
                    else:
                        raise Exception(f"Not found roll_style in parameters for legacy setting")

            # setup asset
            self.parameters["asset"] = asset
        else:
            raise Exception(f"Not found legs in parameters")

    def valid(self):
        name_list = ["asset", "legs"]
        for name in name_list:
            if name not in self.parameters:
                if name == "asset":
                    # backwards compatible
                    self.handle_legacy_setting()
                    continue
                raise Exception(f"Not found {name} in parameters")

    def init(self):
        root = self.parameters["asset"]
        calendar = CALENDAR_DICT.get(root, ["XCBO"])
        skip_months = SKIP_MONTHS_DICT.get(root, [])
        weekly_tickers = ["FV", "TU", "TY", "US"]
        include_weekly = root in weekly_tickers
        weekly_expiry_filter = self.parameters.get("weekly_expiry_filter", None)
        skip_future = self.parameters.get("skip_future", False)
        data_requests = {
            "Options": (IVOLOptionDataRequest(root, self.start_date, self.end_date, calendar=calendar, include_weekly=include_weekly, weekly_expiry_filter=weekly_expiry_filter, skip_future=skip_future),
                        IVOLOptionDataSource()),
            "Futures": (FutureDataRequest(self.start_date, self.end_date, calendar, root, "Comdty", skip_months=skip_months),
                        IVOLFutureDataSource()),
        }

        def get_fut_tgt_tenor(opt_tgt, leg_params, leg_name, weekly_tickers):
            if "fut_tgt_tenor" in leg_params:
                return leg_params["fut_tgt_tenor"]
            if root in TENOR_DICT and TENOR_DICT[root]["opt_tgt"] == opt_tgt:
                return TENOR_DICT[root]["fut_tgt"]
            if leg["root"] in weekly_tickers:
                return None
            raise Exception(f"Not found target future tenor: fut_tgt_tenor in leg {leg_name} parameters")

        def get_future_min_tenor(opt_tgt, leg_params, leg_name, weekly_tickers):
            if "fut_min" in leg_params:
                return leg_params["fut_min"]
            if root in TENOR_DICT and TENOR_DICT[root]["opt_tgt"] == opt_tgt:
                return TENOR_DICT[root]["fut_min"]
            if leg["root"] in weekly_tickers:
                return None
            raise Exception(f"Not found minimum future tenor: fut_min in leg {leg_name} parameters")

        parameters = dict()
        parameters["legs"] = copy.deepcopy(self.parameters["legs"])
        supported_strike_types = ["delta", "atmf"]
        for leg_name, leg in parameters["legs"].items():
            leg["target_option_tenor"] = leg["expiry"]
            if leg["strike_type"] not in supported_strike_types:
                raise Exception(f"Found unsupported strike type {leg['strike_type']}. Only support: {','.join(supported_strike_types)}")
            if leg["strike_type"] == "delta":
                leg["delta"] = leg["strike"]
                del leg["strike"]
            elif leg["strike_type"] == "atmf":
                leg["strike"] = leg["strike"]
            leg["root"] = root
            leg["tenor"] = get_fut_tgt_tenor(leg["target_option_tenor"], leg, leg_name, weekly_tickers)
            leg["min_tenor"] = get_future_min_tenor(leg["target_option_tenor"], leg, leg_name, weekly_tickers)
            leg["tranche"] = CommonBT.create_tranche(leg["tranche"])
        parameters["valuer_map"] = {
            Option: OptionDataValuer_Zero_Px(otm_threshold=-0.0, itm_threshold=0.125, raise_if_zero_px=True,
                                             underlying_valuer=FutureDataValuer(price_name="price", imply_delta_from_spot=False), verbose=False),
            Future: FutureDataValuer(price_name="price", imply_delta_from_spot=False, overrides={}),
        }
        parameters["allow_fill_forward_missing_data"] = 0
        parameters["trade_first_day"] = self.parameters.get("trade_first_day", True)

        if "tc_factor" in self.parameters and self.parameters["tc_factor"] != 0:
            parameters["flat_vega_charge"] = {
                "tc_delta": self.parameters["tc_factor"],
                "tc_vega": self.parameters["tc_factor"]
            }
        self.strategy = FutureOptionDaily(start_date=self.start_date, end_date=self.end_date, calendar=calendar, currency=self.currency,
                                          parameters=parameters, data_requests=data_requests, force_run=self.force_run, cache_market_data=self.cache_market_data,
                                          data_cache_path=self.data_cache_path, logging=self.logging, log_file=self.log_file)
        self.initial_state = FutureOptionState(self.start_date, Portfolio([]), 0.0, 0.0)

    def get_pfo_df(self):
        if self.pfo_df is None:
            res = []
            for state in self.results:
                for leg_name, leg_pfo in state.portfolio.root.items():
                    if isinstance(leg_pfo, Portfolio):
                        for pfo_name, pfo in leg_pfo.root.items():
                            assert isinstance(leg_pfo, Portfolio)
                            for pos_name, pos in pfo.root.items():
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
