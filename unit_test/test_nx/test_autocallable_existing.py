import unittest
import warnings
from datetime import datetime
from ...tools import test_utils
from ...valuation.eq_autocallable_nx_valuer import EQAutoCallableNXValuer
from ...data.instruments import InstrumentService
from ...risk.risk_engine import RiskEngine, RiskEngineConfig
from ...dates.utils import get_ny_timezone
from ...infrastructure.market_builder import MarketBuilder
from ...risk.greeks import flatten_numerical_greeks
import os


class TestAutoCallableExisting(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAutoCallableExisting, self).__init__(*args, **kwargs)
        self.src_folder = test_utils.get_test_src_folder("test_autocallable_existing")
        self.test_folder = test_utils.get_test_data_folder("test_autocallable_existing")
        self.svc = InstrumentService()
        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test(self):
        svc = InstrumentService()
        inst_ids = [75051666]
        inst = svc.get_instruments(inst_ids=inst_ids)[0]
        dt = datetime(2025, 5, 8)
        autocallable = inst.tradable
        market = MarketBuilder.build_market(autocallable, dt)
        pv = EQAutoCallableNXValuer().price(autocallable, market, calc_types="price")
        print(pv)
        self.assertAlmostEqual(pv, -0.01428480043989514)

    def test_numerical_greeks(self):
        test_name = "test_numerical_greeks"
        svc = InstrumentService()
        inst_ids = [75051666]
        inst = svc.get_instruments(inst_ids=inst_ids)[0]
        dt = datetime(2025, 5, 8)
        autocallable = inst.tradable
        market = MarketBuilder.build_market(autocallable, dt)
        calc_types = ["price", "numerical#delta", "numerical#vega"]
        greeks = EQAutoCallableNXValuer().price(autocallable, market, calc_types=calc_types, one_side=False)
        actual_res = flatten_numerical_greeks(greeks, calc_types)
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_numerical_greeks_single_name(self):
        test_name = "test_numerical_greeks_single_name"
        svc = InstrumentService()
        inst_ids = [75570225]
        inst = svc.get_instruments(inst_ids=inst_ids)[0]
        dt = datetime(2025, 5, 8)
        autocallable = inst.tradable
        market = MarketBuilder.build_market(autocallable, dt)
        calc_types = ["price", "numerical#delta", "numerical#gamma", "numerical#vega", "numerical#volga", "numerical#vanna", "numerical#rho", "numerical#repo", "numerical#div"]
        greeks = EQAutoCallableNXValuer().price(autocallable, market, calc_types=calc_types, one_side=True)
        actual_res = flatten_numerical_greeks(greeks, calc_types)
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)

    def test_risk_engine(self):
        test_name = "test_risk_engine"
        inst_ids = [75570225, 75570225]
        base_dt = datetime(2025, 5, 8, 17, 0, tzinfo=get_ny_timezone())
        greeks_list = ["price", "numerical#delta"]
        one_side = True
        config = RiskEngineConfig(base_dt=base_dt, inst_ids=inst_ids, greeks_list=greeks_list, one_side=one_side)
        re = RiskEngine(config=config)
        actual_res = re.run()
        target_file = os.path.join(self.test_folder, f"{test_name}.csv")
        test_utils.assert_dataframe(actual_res, target_file, self.rebase)
