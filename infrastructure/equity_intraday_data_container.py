from datetime import datetime, time
from ..data.datalake import get_bbg_history
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
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
        return self.get_spot(dt)

    def get_data_dts(self):
        return self._get_data_dts()


class EquityIntradayDataRequest(IDataRequest):
    def __init__(self, look_times, ticker,
                 inc_prev_day = False ):
        self.look_times = look_times
        self.ticker = ticker

class IVolEquityIntradayDataSource(IDataSource):
    def __init__(self, spot_field='price_last'):
        self.cassandra = DatalakeCassandra()
        self.data_dict = {}
        # TODO: this should not be other than 'price_close_opt'
        # remove after reconsicilation
        self.spot_field = spot_field

    def initialize(self, data_request):
        ticker = data_request.ticker
        self.data_dict = {}

        stock_ids = self.cassandra.find_stock_ids(self.cassandra.get_session(), option_root_from_ticker(ticker)).astype(str).to_list()

        data_df = self.cassandra.get_intraday_stock_data(stock_ids, data_request.look_times)
        data_df['tstamp']=data_df['tstamp'].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        if not data_df.empty:
            self.data_dict = dict(zip([x for x in list(data_df['tstamp'].dt.to_pydatetime())],
                                      [{ticker: x}for x in data_df[self.spot_field].values]))
            

        def _get_equity_prices(dt):
            
            if dt in self.data_dict:
                return self.data_dict[dt]
            else:
                for key in self.data_dict.keys():
                    if key>dt:
                        break
                    most_recent_dt=key
                return self.data_dict[most_recent_dt]


        def _get_data_dts():
            return list( self.data_dict.keys() )

        container = EquityDataContainer(ticker)
        container._get_equity_prices = _get_equity_prices
        container._get_data_dts = _get_data_dts
        return container
    
if __name__ == '__main__':
    import pandas as pd
    start_date = datetime(2024, 1, 5, 0)
    end_date = datetime(2024, 2, 1, 0)
    schedule=pd.date_range(start_date,end_date, freq='1T')
    schedule=schedule[schedule.day_of_week < 5]
    data_request = EquityIntradayDataRequest(ticker='SPX Index', look_times=schedule[schedule.indexer_between_time('9:00:00','16:00:00')].strftime("%Y-%m-%d %H:%M:%S").to_list())
    data_source = IVolEquityIntradayDataSource()
    container = data_source.initialize(data_request)
    print(container.get_spot(datetime.now()))