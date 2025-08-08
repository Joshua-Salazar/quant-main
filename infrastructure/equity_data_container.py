from datetime import datetime, time
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..infrastructure.eq_spot import EQSpot
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..interface.market_items.ispot import ISpot
from ..data.datalake_cassandra import DatalakeCassandra
from ..analytics.symbology import option_root_from_ticker
from ..dates.utils import minus_tenor


class EquityDataContainer(DataContainer, ISpot):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_spot_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_equity_prices(self, dt):
        return self._get_equity_prices(dt)

    def get_spot(self, base_date: datetime) -> float:
        """
        :param dt: current date for spot value
        :return: scalar spot value
        """
        data = self.get_equity_prices(base_date)
        if data is None:
            return None
        else:
            return data[self.ticker]

    def get_market_item(self, dt):
        spot = self.get_spot(dt)
        return None if spot is None else EQSpot(self.ticker, spot)

    def get_data_dts(self):
        return self._get_data_dts()


class EquityDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker,
                 inc_prev_day = False ):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker
        self.inc_prev_day = inc_prev_day


class EquityPricesDataSource(IDataSource):
    def __init__(self, dates, prices):
        self.prices = prices
        self.dates = dates
        self.data_dict = dict(zip(dates, prices))

    def initialize(self, data_request):
        ticker = data_request.ticker
        for k in self.data_dict.keys():
            self.data_dict[k] = {ticker: self.data_dict[k]}

        def _get_equity_prices(dt):
            return self.data_dict if dt is None else self.data_dict.get(dt, None)

        def _get_data_dts():
            return list(self.data_dict.keys())

        container = EquityDataContainer(ticker)
        container._get_equity_prices = _get_equity_prices
        container._get_data_dts = _get_data_dts

        return container


class DatalakeBBGEquityDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def get_equity_prices(self, dt):
        return self.data_dict if dt is None else self.data_dict.get(dt, None)

    def get_data_dts(self):
        return list(self.data_dict.keys())

    def initialize(self, data_request):
        ticker = data_request.ticker
        data_df = get_bbg_history([ticker], 'PX_LAST', data_request.start_date, data_request.end_date)
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in data_df['date'].values],
                                  [{ticker: x}for x in data_df['PX_LAST'].values]))

        container = EquityDataContainer(ticker)
        container._get_equity_prices = self.get_equity_prices
        container._get_data_dts = self.get_data_dts

        return container


class IVolEquityDataSource(IDataSource):
    def __init__(self, spot_field='price_close_opt'):
        self.cassandra = DatalakeCassandra()
        self.data_dict = {}
        # TODO: this should not be other than 'price_close_opt'
        # remove after reconsicilation
        self.spot_field = spot_field

    def initialize(self, data_request):
        ticker = data_request.ticker
        self.data_dict = {}

        if data_request.inc_prev_day:
            # include data before start date for e.g. t-1 based sizing
            data_df = self.cassandra.get_stock_data(option_root_from_ticker(ticker),
                                                    minus_tenor( data_request.start_date, '1W'),
                                                    data_request.end_date)
        else:
            data_df = self.cassandra.get_stock_data(option_root_from_ticker(ticker), data_request.start_date, data_request.end_date)
        if not data_df.empty:
            self.data_dict = dict(zip([datetime.combine(x.date(), datetime.min.time()) for x in list(data_df['tstamp'].dt.to_pydatetime())],
                                      [{ticker: x}for x in data_df[self.spot_field].values]))

        def _get_equity_prices(dt):
            try:
                return self.data_dict[dt]
            except:
                # hack to make id market data work
                # TODO: generalise
                return self.data_dict.get(datetime.combine(dt.date(), time(0, 0)), None)

        def _get_data_dts():
            return list( self.data_dict.keys() )

        container = EquityDataContainer(ticker)
        container._get_equity_prices = _get_equity_prices
        container._get_data_dts = _get_data_dts
        return container
    

class EquityDataExtraFieldContainer(DataContainer, ISpot):
    def __init__(self, ticker):
        self.ticker = ticker
        self.market_key = market_utils.create_spot_key(ticker)

    def get_market_key(self):
        return self.market_key

    def get_equity_prices(self, dt):
        return self._get_equity_prices(dt)

    def get_spot(self, base_date: datetime) -> float:
        """
        :param dt: current date for spot value
        :return: scalar spot value
        """
        data = self.get_equity_prices(base_date)
        if data is None:
            return None
        else:
            return data[self.ticker]
    
    def get_equity_fields(self, dt):
        return self._get_equity_fields(dt)

    def get_market_item(self, dt):
        spot = self.get_spot(dt)
        return None if spot is None else EQSpot(self.ticker, spot)

    def get_data_dts(self):
        return self._get_data_dts()


class EquityDataExtraFieldRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, ticker, fields,
                 inc_prev_day = False ):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.ticker = ticker
        self.fields = fields
        self.inc_prev_day = inc_prev_day

class DatalakeBBGEquityExtraFieldDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def get_equity_prices(self, dt):
        return {self.ticker:self.data_dict if dt is None else self.data_dict.get(dt, None)['PX_LAST']}
    
    def get_equity_fields(self, dt):
        return {self.ticker:self.data_dict if dt is None else self.data_dict.get(dt, None)}
        
        

    def get_data_dts(self):
        return list(self.data_dict.keys())

    def initialize(self, data_request):
        ticker = data_request.ticker
        self.ticker = ticker
        data_df = get_bbg_history([ticker], data_request.fields, data_request.start_date, data_request.end_date)

        data_df['date']=[datetime.fromisoformat(x) for x in data_df['date'].values]

        data_df=data_df[['date']+data_request.fields]

        self.data_dict=data_df.set_index('date').to_dict('index')

        container = EquityDataExtraFieldContainer(ticker)
        container._get_equity_prices = self.get_equity_prices
        container._get_data_dts = self.get_data_dts
        container._get_equity_fields = self.get_equity_fields

        return container
