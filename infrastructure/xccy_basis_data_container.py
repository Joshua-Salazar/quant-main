from datetime import datetime
from ..data.datalake import DATALAKE
from ..data.datalake import get_bbg_history
from ..constants.ccy import Ccy
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource


class XccyBasisDataContainer(DataContainer):
    def __init__(self, ccy: Ccy, swap_term: str):
        self.market_key = market_utils.create_xccy_basis_key(ccy, swap_term)

    def get_market_key(self):
        return self.market_key

    def get_xccy_basis_data(self, dt=None):
        return self._get_xccy_basis_data(dt)

    def get_data_dates(self):
        return self._get_data_dates()

    def get_market_item(self, dt):
        return self.get_xccy_basis_data(dt)


class XccyBasisDataRequest(IDataRequest):
    def __init__(self, start_date, end_date, ccy: Ccy, swap_term: str = "3M", forward_starting_tenor: str = ""):
        self.swap_term = swap_term.upper()
        self.forward_starting_tenor = forward_starting_tenor.upper()
        self.support_ccy_list = [Ccy.AUD, Ccy.EUR, Ccy.JPY]
        if ccy not in self.support_ccy_list:
            raise Exception(f"Not support ccy {ccy}, on support: {','.join(self.support_ccy_list)}")
        self.ccy = ccy
        self.start_date = start_date
        self.end_date = end_date
        self.bbg_ticker_map = {
            "3M": {Ccy.AUD: "ADBSC IIRS Curncy", Ccy.EUR: "EUBSC IIRS Curncy", Ccy.JPY: "JYBSC IIRS Curncy"}
        }

    def get_bbg_ticker(self):
        if self.swap_term != "3M":
            raise Exception(f"Only support 3M swap for bbg source but found {self.swap_term}")
        return self.bbg_ticker_map[self.swap_term][self.ccy]

    def get_citi_tickers(self):
        pair_str = f"{self.ccy.value}.{Ccy.USD.value}"
        prefix = f"RATES.XCCY_SWAP.{pair_str}.PAR" if self.forward_starting_tenor == "" else \
            f"RATES.XCCY_SWAP.{pair_str}.FWD.{self.forward_starting_tenor}"
        supported_tenors = ["3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "11Y",
                            "12Y", "15Y", "20Y", "25Y", "30Y"]
        if self.swap_term not in supported_tenors:
            raise Exception(f"Missing {self.swap_term} in citi source")
        ticker = f"{prefix}.{self.swap_term}.BASIS_SPREAD"
        return ticker


class DatalakeBBGXccyBasisDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        ccy = data_request.ccy
        xccy_basis_bbg_ticker = data_request.get_bbg_ticker()
        xccy_basis_df = get_bbg_history([xccy_basis_bbg_ticker], 'PX_LAST', data_request.start_date, data_request.end_date)
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in xccy_basis_df['date'].values],
                                  [{ccy: x} for x in xccy_basis_df['PX_LAST'].values]))

        def _get_xccy_basis_data(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        def _get_data_dates():
            return self.data_dict.keys()

        container = XccyBasisDataContainer(ccy, data_request.swap_term)
        container._get_xccy_basis_data = _get_xccy_basis_data
        container._get_data_dates = _get_data_dates
        return container


class DatalakeCitiXccyBasisDataSource(IDataSource):
    def __init__(self):
        self.data_dict = {}

    def initialize(self, data_request):
        ccy = data_request.ccy

        tickers = data_request.get_citi_tickers()
        df = DATALAKE.getData('CITI_VELOCITY', tickers, 'VALUE', data_request.start_date, data_request.end_date, None)
        self.data_dict = dict(zip([datetime.fromisoformat(x) for x in df['tstamp'].values],
                                  [{ccy: x} for x in df['VALUE'].values]))

        def _get_xccy_basis_data(dt):
            if dt is None:
                return self.data_dict
            else:
                return self.data_dict.get(dt, None)

        def _get_data_dates():
            return self.data_dict.keys()

        container = XccyBasisDataContainer(ccy, data_request.swap_term)
        container._get_xccy_basis_data = _get_xccy_basis_data
        container._get_data_dates = _get_data_dates
        return container
