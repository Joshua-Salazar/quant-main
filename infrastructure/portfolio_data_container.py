import pickle
from datetime import datetime
from dateutil.parser import parse

import pandas as pd
from ..data.instruments import Instrument

from ..data.vola import find_underlying
from ..constants.underlying_type import UnderlyingType
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
from ..tradable.future import Future
from ..tradable.portfolio import Portfolio
from ..tradable.position import Position
from ..tradable.option import Option
from ..tradable.stock import Stock


class PortfolioDataContainer(DataContainer):
    def __init__(self, name):
        self.name = name
        self.market_key = market_utils.create_portfolio_key(name)

    def get_market_key(self):
        return self.market_key

    def get_portfolio_data(self, dt):
        return self._get_portfolio_data(dt)

    def get_portfolio_spec_data(self, dt):
        return self._get_portfolio_spec_data(dt)

    def get_market_item(self, dt):
        return self.get_portfolio_data(dt)


class PortfolioDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, calendar, name):
        self.start_date = start_date
        self.end_date = end_date
        self.calendar = calendar
        self.name = name


class PortfolioFileDataSource(IDataSource):
    def __init__(self, file_name=None, spec_file_name=None, inst_file_name=None):
        self.file_name = file_name
        self.spec_file_name = spec_file_name
        self.inst_file_name = inst_file_name
        self.data_dict = {}
        self.spec_data_dict = {}

        if not (self.file_name is None or self.inst_file_name is None):
            raise Exception("Missing file_name or inst_file_name")
        if self.file_name is not None and self.inst_file_name is not None:
            raise Exception(f"Cannot have both file_name {self.file_name} and inst_file_name {self.inst_file_name}")
        if self.inst_file_name is not None and self.spec_file_name is not None:
            raise Exception(f"Not yet support spec_file_name {self.spec_file_name} "
                            f"with inst_file_name {self.inst_file_name}")

        self.use_inst = self.file_name is None

    def get_portfolio_data(self, dt):
        return self.data_dict[dt] if dt in self.data_dict else None

    def get_portfolio_spec_data(self, dt):
        return None if self.use_inst else self.spec_data_dict[dt]

    def initialize(self, data_request):
        name = data_request.name

        if self.use_inst:
            all_data = pd.read_pickle(self.inst_file_name)
            all_data = all_data[~all_data["units"].isna()]
            all_data = all_data[all_data["error"] == ""]
            all_data = all_data[~all_data["underlying"].isin(["AEX Index"])]
            all_data["underlying"] = all_data["underlying"].apply(lambda und: und.replace("SPY", "SPX"))
            all_data["root"] = all_data["root"].apply(lambda und: und.replace("SPY", "SPX"))
            all_data = all_data[~all_data["root"].isin(["CRUDE"])]
            all_data = all_data[all_data["units"] != 0]
            all_data["date"] = pd.to_datetime(all_data["date"]).dt.strftime('%Y-%m-%d')
        else:
            all_data = pd.read_csv(self.file_name)
        all_data = all_data[(all_data['date'] >= data_request.start_date.strftime('%Y-%m-%d')) & (all_data['date'] <= data_request.end_date.strftime('%Y-%m-%d'))]
        all_dates = sorted(list(set(all_data['date'].values)))

        if self.spec_file_name is not None:
            all_spec_data = pd.read_csv(self.spec_file_name)
            all_spec_data = all_spec_data[(all_spec_data['date'] >= data_request.start_date.strftime('%Y-%m-%d')) & (all_spec_data['date'] <= data_request.end_date.strftime('%Y-%m-%d'))]
            all_spec_dates = sorted(list(set(all_spec_data['date'].values)))
            assert all_dates == all_spec_dates

        self.data_dict = {}
        self.spec_data_dict = {}
        has_pfo_name = "pfo" in all_data.columns
        for d in all_dates:
            positions_data = all_data[all_data['date'] == d].to_records()
            pfo = Portfolio([])
            for row in positions_data:
                use_inst = self.use_inst and isinstance(row["tradable_obj"], Instrument)
                if row['tradable'] in ['Option', "OTCOption"]:
                    expiration = row["tradable_obj"].local_expiration_time() if use_inst else (parse(row['expiration']) if isinstance(row['expiration'], str) else row['expiration'])
                    tz_name = row["tradable_obj"].time_zone() if use_inst else "America/New_York"
                    tradable = Option(
                        root=row['underlying'], underlying=row['underlying'], currency=row['currency'],
                        expiration=expiration, strike=row['strike'],
                        is_call=True if row['call_put'] == 'C' else False, is_american=True if row['style'] == 'A' else False,
                        contract_size=row['contract_size'], tz_name=tz_name
                    )
                elif row['tradable'] == 'OptionOnFuture':
                    expiration = row["tradable_obj"].local_expiration_time() if use_inst else (
                        parse(row['expiration']) if isinstance(row['expiration'], str) else row['expiration'])
                    tz_name = row["tradable_obj"].time_zone() if use_inst else "America/New_York"
                    currency = row['currency']
                    root = row['root'] + " Index"
                    underlying_in_option = find_underlying(UnderlyingType.FUTURES, root, expiration, tz_name,
                                                           currency)
                    tradable = Option(root=root, underlying=underlying_in_option, currency=currency,
                                      expiration=expiration, strike=row['strike'],
                                      is_call=True if row['call_put'] == 'C' else False,
                                      is_american=True if row['style'] == 'A' else False,
                                      contract_size=row['contract_size'], tz_name="America/New_York")
                elif row["tradable"] in ["ETF", "Equity", "Stock"]:
                    tradable = Stock(ticker=row["underlying"], currency=row['currency'])
                elif row["tradable"] == "Future":
                    expiration = row["tradable_obj"].local_expiration_time() if use_inst else (
                        parse(row['expiration']) if isinstance(row['expiration'], str) else row['expiration'])
                    tz_name = row["tradable_obj"].time_zone() if use_inst else "America/New_York"
                    tradable = Future(row['underlying'], row['currency'], expiration,
                                      exchange="", listed_ticker="", tz_name=tz_name,
                                      contract_size=row['contract_size'])
                else:
                    raise RuntimeError(f"unknown tradable type in data {row['tradable']}")
                units = row['units']
                position_path = (row["pfo"],) if has_pfo_name else ()
                pfo.add_position(tradable, units, position_path=position_path)

            self.data_dict[datetime.strptime(d, '%Y-%m-%d')] = pfo

            if self.spec_file_name is not None:
                positions_spec_data = all_spec_data[all_spec_data['date'] == d].to_records()
                positions_spec = []
                for row in positions_spec_data:
                    if row['tradable'] == 'Option':
                        tradable = Option(
                            root=row['underlying'], underlying=row['underlying'], currency=row['currency'],
                            expiration=row['expiration'], strike=row['strike'],
                            is_call=True if row['call_put'] == 'C' else False,
                            is_american=True if row['style'] == 'A' else False,
                            contract_size=row['contract_size'], tz_name="America/New_York"
                        )
                    else:
                        raise RuntimeError(f"unknown tradable type in data {row['tradable']}")
                    units = row['units']
                    positions_spec.append(Position(tradable, units))

                self.spec_data_dict[datetime.strptime(d, '%Y-%m-%d')] = positions_spec

        container = PortfolioDataContainer(name)
        container._get_portfolio_data = self.get_portfolio_data
        container._get_portfolio_spec_data = self.get_portfolio_spec_data
        return container


class BacktestResultsDataSource(IDataSource):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_dict = {}

    def initialize(self, data_request):
        name = data_request.name

        with open(self.file_name, 'rb') as f:
            backtest_results = pickle.load(f)

        self.data_dict = {}
        for state in backtest_results:
            if data_request.start_date <= state.time_stamp <= data_request.end_date:
                self.data_dict[state.time_stamp] = state.portfolio

        def _get_portfolio_data(dt):
            return self.data_dict[dt]

        container = PortfolioDataContainer(name)
        container._get_portfolio_data = _get_portfolio_data
        return container
