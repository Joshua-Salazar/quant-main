from datetime import datetime
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..data.datalake import DATALAKE


class FXDataContainer(DataContainer):
    def __init__(self, pair: str):
        self.market_key = market_utils.create_fx_spot_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_fx_data(self, dt=None):
        return self._get_fx_data(dt)

    def get_data_dates(self):
        return self._get_data_dates()

    def get_market_item(self, dt):
        return self.get_fx_data(dt)


class FXDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, currency, denominator_currency, fixing_type=None):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.currency = currency
        self.denominator_currency = denominator_currency
        self.fixing_type = fixing_type

    def get_bbg_ticker(self):
        if self.fixing_type is None:
            return f'{self.currency}{self.denominator_currency} Curncy'
        else:
            return f'{self.currency}{self.denominator_currency} {self.fixing_type} Curncy'

    def get_inverse_bbg_ticker(self):
        if self.fixing_type is None:
            return f'{self.denominator_currency}{self.currency} Curncy'
        else:
            return f'{self.denominator_currency}{self.currency} {self.fixing_type} Curncy'

    def get_fx_pair_name(self):
        return f'{self.currency}{self.denominator_currency}'


class DatalakeCitiFXDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        fx_pair = data_request.get_fx_pair_name()
        tickers = f"FX.SPOT.{data_request.currency}.{data_request.denominator_currency}.CITI"
        fx_df = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
        if fx_df.empty:
            inverse_tickers = f"FX.SPOT.{data_request.denominator_currency}.{data_request.currency}.CITI"
            fx_df = DATALAKE.getData('CITI_VELOCITY', inverse_tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
            if fx_df.empty:
                raise RuntimeError(f"cannot find fx spot history for {fx_pair} and source {data_request.fixing_type}")
            else:
                fx_df['VALUE'] = 1 / fx_df['VALUE']

        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in fx_df['tstamp'].values],
                                  [{fx_pair: x} for x in fx_df['VALUE'].values]))

        def _get_fx_data(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict[dt]

        def _get_data_dates():
            return self.data_dict.keys()

        container = FXDataContainer(fx_pair)
        container._get_fx_data = _get_fx_data
        container._get_data_dates = _get_data_dates
        return container


class DatalakeBBGFXDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        fx_pair = data_request.get_fx_pair_name()
        fx_bbg_ticker = data_request.get_bbg_ticker()
        fx_df = get_bbg_history([fx_bbg_ticker], 'PX_LAST', data_request.start_date, data_request.end_date, throw_if_missing=False)
        if fx_df.empty:
            inverse_fx_bbg_ticker = data_request.get_inverse_bbg_ticker()
            fx_df = get_bbg_history([inverse_fx_bbg_ticker], 'PX_LAST', data_request.start_date, data_request.end_date, throw_if_missing=False)
            if fx_df.empty:
                raise RuntimeError(f"cannot find fx spot history for {fx_pair} and source {data_request.fixing_type}")
            else:
                fx_df['PX_LAST'] = 1 / fx_df['PX_LAST']
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in fx_df['date'].values],
                                  [{fx_pair: x} for x in fx_df['PX_LAST'].values]))

        container = FXDataContainer(fx_pair)
        container._get_fx_data = self._get_fx_data
        container._get_data_dates = self._get_data_dates
        return container

    def _get_fx_data(self, dt):
        if dt is None:
            return self.data_dict
        else:
            return self.data_dict[dt]

    def _get_data_dates(self):
        return self.data_dict.keys()
