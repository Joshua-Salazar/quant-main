import unittest
import warnings
from datetime import datetime
from ...tools import test_utils
from ...data.instruments import InstrumentService
from ...risk.risk_engine import RiskEngine, RiskEngineConfig
from ...dates.utils import get_ny_timezone
import os


class TestAutoCallableNXValuerQA(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutoCallableNXValuerQA, self).__init__(*args, **kwargs)
        self.src_folder = test_utils.get_test_src_folder("test_autocallable_qa")
        self.test_folder = test_utils.get_test_data_folder("test_autocallable_qa")
        self.svc = InstrumentService()
        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_jpm(self):
        test_name = "test_jpm"
        inst_ids = [76095560]
        base_dt = datetime(2025, 6, 4, 17, 0, tzinfo=get_ny_timezone())
        greeks_list = ["price", "numerical#delta", "numerical#gamma", "numerical#vega"]
        one_side = False
        flat_corr = 0.96
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side, flat_corr=flat_corr)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res.transpose(), target_file, self.rebase)

