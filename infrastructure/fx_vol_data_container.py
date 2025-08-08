from datetime import datetime, date

from ..analytics.fx_vol_surface import FXVolSurfaceFromQuotedVols
from ..data.datalake import DATALAKE
from ..data.market import get_expiry_in_year_from_string
from ..dates.utils import get_business_days
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
import numpy as np
import re

import pandas as pd


class FXVolDataContainer(DataContainer):
    def __init__(self, pair: str):
        self.market_key = market_utils.create_fx_vol_surface_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_surface(self, dt=None):
        return self._get_fx_surface(dt)

    def get_market_item(self, dt):
        return self.get_fx_surface(dt)

    def get_option_data(self, dt, option, dummy_arg ):
        return self._get_option_data(dt, option )

class FXVolDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, base_currency, term_currency):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.base_currency = base_currency
        self.term_currency = term_currency


class DatalakeSABRFXVolDataSource(IDataSource):
    def __init__(self):
        self.data_container = pd.DataFrame()

    def initialize(self, data_request):
        und_to_CTPid = {'AUDUSD': '1127503', 'EURUSD': '1127704', 'GBPUSD': '1127737', 'USDCAD': '1128035',
                        'USDCHF': '1128031', 'USDJPY': '1128060'}
        if data_request.base_currency in ['AUD', 'EUR', 'GBP']:
            data_pair_ticker = data_request.base_currency + '.' + data_request.term_currency
            fx_pair = data_request.base_currency + data_request.term_currency
        else:
            data_pair_ticker = data_request.term_currency + '.' + data_request.base_currency
            fx_pair = data_request.term_currency + data_request.base_currency
        # see https://gitea.capstoneco.com/dcirmirakis/ctp_py_examples/src/branch/master/data_provider.py
        fields = 'source,location,under_id,under_pricing_id,type,dimension,term,actual_date,capture_date,fit_source,' \
                 'spot,forward,time_to_expiry,alpha,beta,rho,nu'
        ex_vals = 'CLOSE|LDN|SURFACE'
        CTP_id = und_to_CTPid[data_pair_ticker.replace('.', '')]
        all_data = DATALAKE.getData('CTP_DAILY_VOL_SABR', CTP_id, fields, data_request.start_date,
                              data_request.end_date, extra_fields='source|location|dimension', extra_values=ex_vals)

        #
        all_dates = all_data.tstamp.unique()
        data = []
        for dt in all_dates:
            data_dt = all_data[all_data.tstamp == dt]
            if len(np.unique(np.array([x[11:] for x in data_dt.capture_date]))) > 1:
                final_capture = data_dt.capture_date.values[-1]
                data_dt = data_dt[data_dt.capture_date == final_capture]
            data.append(data_dt)
        data_container = pd.concat(data)
        data_container = data_container[['tstamp', 'term', 'actual_date', 'capture_date', 'spot', 'forward',
                                         'time_to_expiry', 'alpha', 'beta','rho', 'nu']]
        # TODO: add vol computation from SABR model in BS pricer
        return data_container


class DatalakeCitiFXVolDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    @staticmethod
    def delta_strike_str_to_delta(strike_str):
        if strike_str == 'ATM':
            # TODO: this is only a shortcut -- need to solve for the delta that makes straddle delta neutral
            return 0.5
        else:
            strike_elements = re.match(r'STRIKE_(C|P)(\d{2})', strike_str).groups()
            type_str = strike_elements[0]
            strike = strike_elements[1]
            if type_str == 'C':
                return float(strike) / 100.0
            elif type_str == 'P':
                return -float(strike) / 100.0
            else:
                raise RuntimeError(f'Unknown strike quote type found in {strike_str}')

    def initialize(self, data_request):
        fixed_tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y']
        call_opts = ['C35', 'C25', 'C10']
        put_opts = ['P35', 'P25', 'P10']
        if data_request.base_currency in ['AUD', 'EUR', 'GBP'] \
                or (data_request.base_currency == "USD" and data_request.term_currency in ["JPY", 'CAD', 'CHF']):
            data_pair_ticker = data_request.base_currency + '.' + data_request.term_currency
            fx_underlying = data_request.base_currency + data_request.term_currency
        else:
            data_pair_ticker = data_request.term_currency + '.' + data_request.base_currency
            fx_underlying = data_request.term_currency + data_request.base_currency

        # load vol data
        tickers = ''
        for tenor in fixed_tenors:
            for opt in ['ATM'] + call_opts + put_opts:
                if opt == 'ATM':
                    tickers += 'FX.IMPLIED_VOL.%s.%s.%s.CITI,' % (data_pair_ticker, opt, tenor)
                else:
                    tickers += 'FX.IMPLIED_VOL.%s.STRIKE_%s.%s.CITI,' % (data_pair_ticker, opt, tenor)
        vol_data = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date,
                                    None).rename(columns={'tstamp': 'date'})
        vol_data['delta_strike'] = vol_data['ticker'].str.split('.').apply(lambda x: x[4])
        vol_data['tenor'] = vol_data['ticker'].str.split('.').apply(lambda x: x[5])
        vol_data['date'] = vol_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())
        # spot
        spots = DATALAKE.getData('CITI_VELOCITY', f'FX.SPOT.{data_pair_ticker}.CITI', 'VALUE', data_request.start_date, data_request.end_date, None).rename(columns={'tstamp': 'date'})
        spots['date'] = spots['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())
        # load forwards
        tickers = ''
        for tenor in fixed_tenors:
            tickers += 'FX.FORWARD.FWD_OUTRIGHT.%s.%s.CITI,' % (data_pair_ticker, tenor)
        forward_data = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date,
                                    None).rename(columns={'tstamp': 'date'})
        forward_data['tenor'] = forward_data['ticker'].str.split('.').apply(lambda x: x[5])
        forward_data['date'] = forward_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())

        for dt in get_business_days(data_request.start_date, data_request.end_date):
            if len( spots[spots['date'] == dt.isoformat()]['VALUE'] ) == 0:
                continue
            spot = spots[spots['date'] == dt.isoformat()]['VALUE'].values[0]

            forwards = {}
            volatilities = {}
            for tenor in fixed_tenors:
                data = vol_data[(vol_data['date'] == dt.isoformat()) & (vol_data['tenor'] == tenor)]
                if data.empty:
                    print(f'No FX vol data found at {dt} for tenor {tenor}')
                else:
                    delta_strikes = data['delta_strike'].values
                    vols = data['VALUE'].values / 100.0
                    volatilities[get_expiry_in_year_from_string(tenor)] = vols
                    data = forward_data[(forward_data['date'] == dt.isoformat()) & (forward_data['tenor'] == tenor)]
                    # retry with different formatting for fwd data ??
                    if data.empty:
                        data = forward_data[(forward_data['date'] == dt.isoformat())]
                        data = data[data['tenor'] == tenor]

                    if data.empty:
                        if dt.date() > date( 2012, 11, 19 ) :
                            print(f'No FX forward data found at {dt} for tenor {tenor} but vol data is available. NB No 2W fwd data pre 19Nov22.')
                    else:
                        forward = data['VALUE'].values[0]
                        forwards[get_expiry_in_year_from_string(tenor)] = forward
            if len(volatilities) == 0:
                print(f"Empty vol surface on {dt}")
                continue

            self.data_dict.setdefault(dt, {})[fx_underlying] = \
                FXVolSurfaceFromQuotedVols(fx_underlying, dt,
                                           spot, [DatalakeCitiFXVolDataSource.delta_strike_str_to_delta(x) for x in delta_strikes],
                                           forwards, volatilities)

        container = FXVolDataContainer(fx_underlying)

        def _get_fx_surface(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_fx_surface = _get_fx_surface
        return container

class DatalakeBVolFXDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    @staticmethod
    def delta_strike_str_to_delta(strike_str):
        if strike_str == 'ATM':
            # TODO: this is only a shortcut -- need to solve for the delta that makes straddle delta neutral
            return 0.5
        else:
            type_str = strike_str[-1]
            strike = strike_str[:2]
            if type_str == 'C':
                return float(strike) / 100.0
            elif type_str == 'P':
                return -float(strike) / 100.0
            else:
                raise RuntimeError(f'Unknown strike quote type found in {strike_str}')

    def initialize(self, data_request):
        fixed_tenors = [ '1D', '1W', '2W', '1M', '2M', '3M', '6M', '9M']
        call_opts = [ '25DC', '10DC' ]
        put_opts = [ '25DP', '10DP' ]
        data_pair_ticker = data_request.base_currency + '.' + data_request.term_currency
        fx_underlying = data_request.base_currency + data_request.term_currency

        # load vol data
        tickers = ''
        for tenor in fixed_tenors:
            for opt in ['ATM'] + call_opts + put_opts:
                tickers += '%s %s %s VOL BVOL Curncy,' % (data_pair_ticker, tenor, opt )
        vol_data = DATALAKE.getData( 'BBG_PRICE', tickers.replace( '.', ''), 'PX_LAST', data_request.start_date, data_request.end_date, None).rename(columns={'tstamp': 'date'} )
        vol_data['delta_strike'] = vol_data['ticker'].str.split( ' ' ).apply( lambda x: x[2] )
        vol_data['tenor'] = vol_data['ticker'].str.split( ' ' ).apply( lambda x: x[1] )
        vol_data['date'] = vol_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())
        # spot
        spots = DATALAKE.getData('CITI_VELOCITY', f'FX.SPOT.{data_pair_ticker}.CITI', 'VALUE', data_request.start_date, data_request.end_date, None).rename(columns={'tstamp': 'date'})
        spots['date'] = spots['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())
        # load forwards
        tickers = ''
        for tenor in fixed_tenors:
            tickers += 'FX.FORWARD.FWD_OUTRIGHT.%s.%s.CITI,' % ( data_pair_ticker, tenor.replace( '1D', 'ON' ) )
        forward_data = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date, None).rename(columns={'tstamp': 'date'})
        forward_data['tenor'] = forward_data['ticker'].str.split('.').apply(lambda x: x[5]).replace( 'ON', '1D' )
        forward_data['date'] = forward_data['date'].apply(lambda x: datetime.fromisoformat(x).isoformat())

        biz_days = list( set(get_business_days( data_request.start_date, data_request.end_date ) ) - set( data_request.calendar ) )
        biz_days.sort()
        for dt in biz_days:
            spot = spots[spots['date'] == dt.isoformat()]['VALUE'].values[0]

            forwards = {}
            volatilities = {}
            for tenor in fixed_tenors:
                data = vol_data[(vol_data['date'] == dt.isoformat()) & (vol_data['tenor'] == tenor)]
                if data.empty:
                    print(f'No FX vol data found at {dt} for tenor {tenor}')
                else:
                    delta_strikes = data['delta_strike'].values
                    vols = data['PX_LAST'].values / 100.0
                    volatilities[get_expiry_in_year_from_string(tenor)] = vols
                    data = forward_data[(forward_data['date'] == dt.isoformat()) & (forward_data['tenor'] == tenor)]
                    if data.empty:
                        if dt.date() > date( 2012, 11, 19 ) :
                            print(f'No FX forward data found at {dt} for tenor {tenor} but vol data is available. NB No 2W fwd data pre 19Nov22.')
                    else:
                        forward = data['VALUE'].values[0]
                        forwards[get_expiry_in_year_from_string(tenor)] = forward

            self.data_dict.setdefault(dt, {})[fx_underlying] = \
                FXVolSurfaceFromQuotedVols(fx_underlying, dt,
                                           spot, [self.delta_strike_str_to_delta(x) for x in delta_strikes],
                                           forwards, volatilities)

        container = FXVolDataContainer(fx_underlying)

        def _get_fx_surface(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        container._get_fx_surface = _get_fx_surface
        return container

