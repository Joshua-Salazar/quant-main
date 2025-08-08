from .data_container import DataContainer
from ..interface.idatarequest import IDataRequest
from ..interface.idatasource import IDataSource
import pandas as pd


def dataframe_to_generic_dataframe_container(data_request, df):
    container = GenericDataContainer()

    def _get_data():
        return df

    container._get_data = _get_data
    return container


class GenericDataContainer(DataContainer):
    def get_market_key(self):
        pass

    def get_market_item(self, dt):
        pass

    def get_data(self):
        return self._get_data()


class GenericDataRequest(IDataRequest):
    def __init__(self):
        pass


class GenericDataFrameFlatFileDataSource(IDataSource):
    def __init__(self, file_name=None, dataframe_to_container_func=dataframe_to_generic_dataframe_container):
        self.file_name = file_name
        self.dataframe_to_container_func = dataframe_to_container_func

    def initialize(self, data_request):
        df = pd.read_csv(self.file_name)
        container = self.dataframe_to_container_func(data_request, df)
        return container
