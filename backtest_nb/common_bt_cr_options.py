import copy
import pandas as pd
from dateutil.parser import parse

from ..backtest.strategies.simple_cr_options import SimpleCROptions, SimpleCROptionsState
from ..infrastructure.cr_spot_data_container import CRSpotDataRequest, CRSpotsDataSource
from ..backtest_nb.common_bt import CommonBT
from ..backtest_nb.common_bt_future_options_config import *
from ..backtest.indicator import Indicator
from ..infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateInternalDataSource
from ..infrastructure.forward_rate_data_container import ForwardRateRequest, ForwardRateCitiDataSource
from ..backtest.tranche import RollingAtExpiryTranche, RollingAtExpiryDailyTranche
from ..tradable.forwardstartswap import ForwardStartSwap
from ..tradable.portfolio import Portfolio
from ..tradable.swaption import Swaption
from ..backtest_nb.common_bt_cr_options_config import *
from ..infrastructure.cr_vol_data_container import CRVolDataRequest, QuotedCRVolDataSource


class CommonBTCROptions(CommonBT):
    def __init__(self, start_date, end_date, currency, parameters, force_run=False):
        super().__init__(start_date, end_date, currency, parameters, force_run=force_run)
        self.valid()
        self.init()

    def valid(self):
        name_list = ["asset", "legs"]
        for name in name_list:
            if name not in self.parameters:
                raise Exception(f"Not found {name} in parameters")

    def init(self):
        """
        Test case setup to support the original notebook & roll at exp tranche too
        """
        calendar = [CDS_CALENDAR_MAP[self.parameters["asset"]]]
        missing_vol_dates = []
        # format e.g. missing_vol_dates = ['2021-12-24']
        missing_dates = [parse(x) for x in missing_vol_dates]
        calendar += missing_dates

        data_requests = {
            'vols': (
                CRVolDataRequest(
                    self.start_date,
                    self.end_date,
                    calendar,
                    self.parameters["asset"]
                ),
                QuotedCRVolDataSource()
            ),
            'spots': (
                CRSpotDataRequest(
                    self.start_date,
                    self.end_date,
                    calendar,
                    self.parameters["asset"]
                ),
                CRSpotsDataSource()
            ),
            'spot_rates': (
                SpotRateRequest(self.start_date, self.end_date, self.currency, "SOFRRATE"),
                SpotRateInternalDataSource(),
            ),
            'fwd_rates': (
                ForwardRateRequest(self.start_date, self.end_date, self.currency, "SWAP_SOFR"),
                ForwardRateCitiDataSource(),
                # fwd_rates_ds,
            ),
        }

        if "tc_factor" in self.parameters and self.parameters["tc_factor"] != 0:
            self.parameters["tc_rate"] = {
                # implement tc
            }

        self.strategy = SimpleCROptions(
            start_date=self.start_date,
            end_date=self.end_date,
            calendar=calendar,
            currency=self.currency,
            parameters=self.parameters,
            data_requests=data_requests
        )

        self.initial_state = SimpleCROptionsState(self.start_date, Portfolio([]), 0.0, 0.0)

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