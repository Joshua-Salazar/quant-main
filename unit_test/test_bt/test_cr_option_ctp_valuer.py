import os
import unittest
import warnings
from datetime import datetime, timedelta
import pandas as pd
from ...valuation.cr_option_ctp_valuer import CROptionCTPValuer
from ...infrastructure.market import Market
from ...tools import test_utils
from ...tradable.option import Option
from ...infrastructure.cr_vol_data_container import QuotedCRVolDataSource, \
    CRVolDataRequest
from ...infrastructure.spot_rate_data_container import SpotRateRequest, SpotRateCitiDataSource, \
    SpotRateInternalDataSource
from ...infrastructure.cr_spot_data_container import CRSpotsDataSource, CRSpotDataRequest
from ...dates.utils import add_business_days


class TestCRCTPValuer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCRCTPValuer, self).__init__(*args, **kwargs)
        self.test_folder = test_utils.get_test_data_folder("test_cr_option_valuer")
        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_sept_24_roll(self):

        test_name = "test_sept_24_roll"
        st = datetime(2024, 9, 1)
        et = datetime(2024, 10, 15)

        underlying = "CDX_NA_IG"
        currency = "USD"
        expiration = datetime(2024, 10, 16)
        strike = 52
        is_call = False
        is_american = False
        contract_size = 1.0
        tz_name = "America/New_York"
        listed_ticker = None
        expiration_rule = None
        series_version_tenor = '42_1_5'

        call_option = Option(
            root=underlying, underlying=underlying, currency=currency, expiration=expiration, strike=strike,
            is_call=is_call, is_american=is_american, contract_size=contract_size, tz_name=tz_name,
            listed_ticker=listed_ticker, expiration_rule=expiration_rule, specialisation=series_version_tenor)

        calendar = ['NYC']
        holiday_days = []
        for cal in calendar:
            if isinstance(cal, str):
                from ...dates.holidays import get_holidays
                holiday_days = holiday_days + get_holidays(cal, st, et)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')

        cr_vol_data_request = CRVolDataRequest(start_date=st,
                                               end_date=et,
                                               calendar=calendar,
                                               underlier=underlying)
        cr_vol_data_source = QuotedCRVolDataSource()
        cr_vol_data = cr_vol_data_source.initialize(cr_vol_data_request)

        cr_spot_data_request = CRSpotDataRequest(start_date=st,
                                                 end_date=et,
                                                 calendar=calendar,
                                                 underlier=underlying)
        cr_spot_data_source = CRSpotsDataSource()
        cr_spot_data = cr_spot_data_source.initialize(cr_spot_data_request)

        spot_rate_data_request = SpotRateRequest(start_date=st,
                                                 end_date=et,
                                                 currency=currency,
                                                 curve_name="SOFRRATE")
        spot_rate_data_source = SpotRateInternalDataSource()
        spot_rate_data = spot_rate_data_source.initialize(spot_rate_data_request)

        roll_test_series = {
            'dt': [],
            'price': [],
        }

        dt = st
        while dt <= min(et, datetime.today() + timedelta(days=-1)):
            market = Market(base_datetime=dt)
            try:
                market.add_item(cr_vol_data.get_market_key(), cr_vol_data.get_market_item(dt))
                market.add_item(cr_spot_data.get_market_key(), cr_spot_data.get_market_item(dt))
                market.add_item(spot_rate_data.get_market_key(), spot_rate_data.get_market_item(dt))
            except Exception as e:
                print(f"Skipping date {dt} due to error: {e}")
                dt = add_business_days(dt, 1, holiday_days)
                continue
            pv = CROptionCTPValuer(discount_curve_name="SOFRRATE").price(option=call_option, market=market)
            print(dt, pv)
            roll_test_series['dt'].append(dt)
            roll_test_series['price'].append(pv)
            dt = add_business_days(dt, 1, holiday_days)

        roll_test_df = pd.DataFrame(roll_test_series)
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(roll_test_df, target_file, self.rebase)

    def test_mar_25_roll(self):

        test_name = "test_mar_25_roll"
        st = datetime(2025, 3, 1)
        et = datetime(2025, 4, 15)

        underlying = "CDX_NA_IG"
        currency = "USD"
        expiration = datetime(2025, 4, 16)
        strike = 52
        is_call = False
        is_american = False
        contract_size = 1.0
        tz_name = "America/New_York"
        listed_ticker = None
        expiration_rule = None
        series_version_tenor = '42_1_5'

        call_option = Option(
            root=underlying, underlying=underlying, currency=currency, expiration=expiration, strike=strike,
            is_call=is_call, is_american=is_american, contract_size=contract_size, tz_name=tz_name,
            listed_ticker=listed_ticker, expiration_rule=expiration_rule, specialisation=series_version_tenor)

        calendar = ['NYC']
        holiday_days = []
        for cal in calendar:
            if isinstance(cal, str):
                from ...dates.holidays import get_holidays
                holiday_days = holiday_days + get_holidays(cal, st, et)
            elif isinstance(cal, datetime):
                holiday_days.append(cal)
            else:
                raise RuntimeError(f'Unknown type of calendar {str(type(cal))}')

        cr_vol_data_request = CRVolDataRequest(start_date=st,
                                               end_date=et,
                                               calendar=calendar,
                                               underlier=underlying)
        cr_vol_data_source = QuotedCRVolDataSource()
        cr_vol_data = cr_vol_data_source.initialize(cr_vol_data_request)

        cr_spot_data_request = CRSpotDataRequest(start_date=st,
                                                 end_date=et,
                                                 calendar=calendar,
                                                 underlier=underlying)
        cr_spot_data_source = CRSpotsDataSource()
        cr_spot_data = cr_spot_data_source.initialize(cr_spot_data_request)

        spot_rate_data_request = SpotRateRequest(start_date=st,
                                                 end_date=et,
                                                 currency=currency,
                                                 curve_name="SOFRRATE")
        spot_rate_data_source = SpotRateInternalDataSource()
        spot_rate_data = spot_rate_data_source.initialize(spot_rate_data_request)

        roll_test_series = {
            'dt': [],
            'price': [],
        }

        dt = st
        while dt <= min(et, datetime.today() + timedelta(days=-1)):
            market = Market(base_datetime=dt)
            try:
                market.add_item(cr_vol_data.get_market_key(), cr_vol_data.get_market_item(dt))
                market.add_item(cr_spot_data.get_market_key(), cr_spot_data.get_market_item(dt))
                market.add_item(spot_rate_data.get_market_key(), spot_rate_data.get_market_item(dt))
            except Exception as e:
                print(f"Skipping date {dt} due to error: {e}")
                dt = add_business_days(dt, 1, holiday_days)
                continue
            pv = CROptionCTPValuer(discount_curve_name="SOFRRATE").price(option=call_option, market=market)
            print(dt, pv)
            roll_test_series['dt'].append(dt)
            roll_test_series['price'].append(pv)
            dt = add_business_days(dt, 1, holiday_days)

        roll_test_df = pd.DataFrame(roll_test_series)
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(roll_test_df, target_file, self.rebase)
