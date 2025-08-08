import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from ..infrastructure.generic_data_container import GenericDataRequest, GenericDataContainer, GenericDataFrameFlatFileDataSource


class TestGenericDataRequest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGenericDataRequest, self).__init__(*args, **kwargs)

    def test(self):
        data_request = GenericDataRequest()
        file_name = "../test_data/source/index_returns.csv"
        file_name = Path(__file__).parent.resolve().joinpath(file_name).absolute()
        data_container = GenericDataFrameFlatFileDataSource(file_name=file_name).initialize(data_request)
        data = data_container.get_data()
        source_data = pd.read_csv(file_name)
        assert_frame_equal(data, source_data)

