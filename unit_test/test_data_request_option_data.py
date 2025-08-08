import unittest
from datetime import datetime
from ..data.datalake_cassandra import ExpiryFilterByDateOffset
from ..infrastructure.option_data_container import OptionDataRequest, CassandraDSOnDemand


class TestOptionDataRequest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestOptionDataRequest, self).__init__(*args, **kwargs)

    def test(self):
        dt = datetime(2023, 9, 29)
        root = "SPX"
        option_data_request = OptionDataRequest(
            dt, dt, calendar='XCBO', root=root, frequency="daily", expiry_filter=ExpiryFilterByDateOffset(30))

        option_data_container = CassandraDSOnDemand(adjust_expiration=True).initialize(option_data_request)
        option_chain = option_data_container.get_option_universe(dt)
        ccy = option_chain["currency"].iloc[0]
        self.assertEqual(ccy, "USD")

