from datetime import datetime
from ..infrastructure import market_utils
from ..infrastructure.data_container import DataContainer
from ..infrastructure.fx_sol_sabr_vol_surface import FXSOLSABRVolSurface
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource


class FXSOLSABRVolDataContainer(DataContainer):
    def __init__(self, pair: str):
        self.market_key = market_utils.create_fx_vol_surface_key(pair)

    def get_market_key(self):
        return self.market_key

    def get_vol_surface(self, dt=None):
        return self._get_vol_surface(dt)

    def get_market_item(self, dt):
        return self.get_vol_surface(dt)


class FXSOLSABRVolRequest(IDataRequest):
    def __init__(self, pair: str, start_date: datetime, end_date: datetime, requested_dates=None):
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        self.requested_dates = requested_dates


class FXSOLSABRVolDataSourceOnDemand(IDataSource):
    def __init__(self):
        self.data = None
        self.date = None
        self.pair = None

    def clone(self):
        return FXSOLSABRVolDataSourceOnDemand()

    def get_vol_surface(self, dt):
        if dt is None:
            raise Exception("Not support")

        if dt == self.date:
            return self.data
        else:
            self.data = FXSOLSABRVolSurface(self.pair[:3], self.pair[3:], dt)
            self.dt = dt
            return self.data

    def initialize(self, data_request):
        self.pair = data_request.pair

        container = FXSOLSABRVolDataContainer(self.pair)
        container._get_vol_surface = self.get_vol_surface
        return container