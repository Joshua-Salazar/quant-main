from ..constants.market_item_type import MarketItemType
from ..infrastructure.correlation_matrix_container import CorrelationMatrixDataRequest, ConstantCorrelationMatrixDataSource, FlatCorrelationMatrixDataSource
from ..infrastructure.df_curve_data_container import DFCurveRequest, DFCurveBBGDataSource
from ..infrastructure.eq_vol_data_container import EqVolRequest, VolaEqVolDataSource, VolaEqVolDataSourceOnDemand
from ..infrastructure.fx_sabr_vol_data_container import DatalakeFXSABRVolDataSource, FXSABRVolDataRequest
from ..infrastructure.market import Market
from ..infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateInternalDataSource, SpotRateCitiDataSource
from ..infrastructure.fixing_requirement import FixingRequirement
from ..infrastructure.fixing_table import FixingTable
from ..infrastructure.holiday_center import HolidayCenter
from ..infrastructure.holiday_requirement import HolidayRequirement
from ..reporting.trade_reporter import TradeReporter
from ..dates.utils import date_to_datetime, get_holidays
from ..infrastructure import market_utils
from ..data import datalake
import pandas as pd
from datetime import datetime


class MarketBuilder:

    @staticmethod
    def build_market(tradables, dt, bpipe_get_bbg_history=None, valuer_map_override={}, load_shared_file_eq_vol=False, flat_corr=None, datalake_credentials=None):
        tradable_list = tradables if isinstance(tradables, list) else [tradables]
        # 1) ask market keys
        keys = []
        for tradable in tradable_list:
            keys += tradable.ask_keys(valuer=valuer_map_override.get(type(tradable), None))

        # 2) build data request
        data_requests = MarketBuilder.build_data_request(keys, dt, dt, load_shared_file_eq_vol, flat_corr=flat_corr, datalake_credentials=datalake_credentials)
        # 3) create market with data requests
        market = MarketBuilder.create_market_with_data_requests(dt, data_requests)
        # 4) add fixing and holidays
        market = MarketBuilder.add_fixings_and_holidays(tradable_list, market, bpipe_get_bbg_history, valuer_map_override)
        return market

    @staticmethod
    def build_data_request(keys, start_date, end_date, load_shared_file_eq_vol=False, flat_corr=None, datalake_credentials=None):
        data_requests = dict()
        for key in set(keys):
            tokens = key.split(".")
            key_type = tokens.pop(0)
            if key_type == MarketItemType.VOLATILITY.value:
                und = tokens.pop(0)
                data_requests[key] = (
                    EqVolRequest(und, start_date, end_date, num_regular=252),
                    VolaEqVolDataSourceOnDemand(skip_pattern_search=True, load_shared_file=load_shared_file_eq_vol)
                )
            elif key_type == MarketItemType.FXVOLATILITY.value:
                und = tokens.pop(0)
                is_intraday = (start_date.date() == end_date.date()) and (start_date.date() == datetime.today().date())
                data_requests[key] = (
                    FXSABRVolDataRequest(start_date, end_date, calendar=["NYC", "LON"], base_currency=und[:3], term_currency=und[3:]),
                    DatalakeFXSABRVolDataSource(datalake_credentials, live=is_intraday)
                )
            elif key_type == MarketItemType.SPOTRATECURVE.value:
                currency = tokens.pop(0)
                curve_name = tokens.pop(0)
                if curve_name == "BBG_ZERO_RATES":
                    data_requests[key] = (
                        DFCurveRequest(start_date, end_date, currency=currency, name=curve_name),
                        DFCurveBBGDataSource()
                    )
                elif curve_name == "SWAP_SOFR":
                    data_requests[key] = (
                        SpotRateRequest(start_date, end_date, currency, curve_name),
                        SpotRateCitiDataSource(),
                    )
                else:
                    data_requests[key] = (
                        SpotRateRequest(start_date, end_date, currency=currency, curve_name=curve_name),
                        SpotRateInternalDataSource()
                    )
            elif key_type == MarketItemType.CORRELATIONMATRIX.value:
                und_list = tokens
                if flat_corr is None:
                    correlation_shift = 0.05
                    data_requests[key] = (
                        CorrelationMatrixDataRequest(start_date, end_date, und_list, correlation_shift=correlation_shift),
                        ConstantCorrelationMatrixDataSource()
                    )
                else:
                    correlation_shift = 0.0
                    data_requests[key] = (
                        CorrelationMatrixDataRequest(start_date, end_date, und_list, correlation_shift=correlation_shift),
                        FlatCorrelationMatrixDataSource(corr=flat_corr)
                    )
            else:
                raise Exception(f"Found unsupported key {key}")
        return data_requests

    @staticmethod
    def create_market_with_data_requests(dt, data_requests, market=None):
        if market is not None:
            assert market.base_datetime == dt
        else:
            market = Market(base_datetime=dt)

        for key, (data_request, data_source) in data_requests.items():
            data = data_source.initialize(data_request)
            if market_utils.is_fx_vol_surface_key(key):
                dt_used = date_to_datetime(dt.date())
                item = data.get_market_item(dt_used)
                item.base_date = dt
            else:
                item = data.get_market_item(dt)
            market.add_item(key, item)

        return market

    @staticmethod
    def add_fixings_and_holidays(tradables, market, bpipe_get_bbg_history=None, valuer_map_override={}):
        tradable_list = tradables if isinstance(tradables, list) else [tradables]
        bbg_ticker_map = {"SOFRRATE": "SOFRRATE Index", "SWAP_SOFR": "SOFRRATE Index"}
        rate_ticker = ["SOFRRATE", "SWAP_SOFR"]
        fixing_reqs = []
        hol_reqs = []
        for tradable in tradable_list:
            fixing_reqs += TradeReporter(tradable).get_fixing_requirement(market)
            hol_reqs += TradeReporter(tradable).get_holiday_requirement(market, valuer=valuer_map_override.get(type(tradable), None))
        if len(fixing_reqs) > 0:
            # aggregate fixing requirement
            und_fixings = dict()
            for fixing_req in fixing_reqs:
                if fixing_req.underlying in und_fixings:
                    und_fixings[fixing_req.underlying][0] = min(fixing_req.start_date, und_fixings[fixing_req.underlying][0])
                    und_fixings[fixing_req.underlying][1] = max(fixing_req.end_date, und_fixings[fixing_req.underlying][1])
                else:
                    und_fixings[fixing_req.underlying] = [fixing_req.start_date, fixing_req.end_date]
            agg_fixing_reqs = [FixingRequirement(k, v[0], v[1]) for k, v in und_fixings.items()]
            # check if it is live
            is_live = market.base_datetime.date() == datetime.now().date()
            if is_live and bpipe_get_bbg_history is None:
                raise Exception(f"Not found bpipe for live risk on {market.base_datetime.strftime('%Y-%m-%d')}")
            # create fixing table from bbg
            fixings = []
            for fixing_req in agg_fixing_reqs:
                und = bbg_ticker_map.get(fixing_req.underlying, fixing_req.underlying)
                if bpipe_get_bbg_history is None:
                    fixing = datalake.get_bbg_history([und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                else:
                    fixing = bpipe_get_bbg_history([und], "PX_LAST", fixing_req.start_date, fixing_req.end_date)
                fixing["date"] = pd.to_datetime(fixing["date"]).dt.date
                fixing["ticker"] = fixing["ticker"].apply(lambda ticker: fixing_req.underlying)
                fixing = fixing.rename(columns={"PX_LAST": "fixing", "ticker": "underlying"}).set_index(
                    ["date", "underlying"])
                fixings.append(fixing)

            fixings = pd.concat(fixings).reset_index()
            fixings.loc[fixings.underlying.isin(rate_ticker), "fixing"] /= 100
            fixing_table = FixingTable(fixings)
            market.add_item(market_utils.create_fixing_table_key(), fixing_table)

        # check if holiday is required
        if len(hol_reqs) > 0:
            hol_map = dict()
            for hol_req in hol_reqs:
                if hol_req.code in hol_map:
                    hol_map[hol_req.code][0] = min(hol_req.start_date, hol_map[hol_req.code][0])
                    hol_map[hol_req.code][1] = max(hol_req.end_date, hol_map[hol_req.code][1])
                else:
                    hol_map[hol_req.code] = [hol_req.start_date, hol_req.end_date]
            agg_hol_reqs = [HolidayRequirement(k, v[0], v[1]) for k, v in hol_map.items()]
            hol_center = dict()
            for hol_req in agg_hol_reqs:
                holidays = get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
                code_hols = hol_center.get(hol_req.code, []) + holidays
                hol_center[hol_req.code] = sorted(set(code_hols))

            if len(hol_center) > 0:
                key = market_utils.create_holiday_center_key()
                market.add_item(key, HolidayCenter(hol_center))

        return market


