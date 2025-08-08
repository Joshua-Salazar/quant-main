import unittest
import warnings
from datetime import datetime
from ...constants.asset_class import AssetClass
from ...tradable.volswap import VolSwap
from ...valuation.eq_volswap_nx_valuer import EQVolSwapNxValuer

from ...data.refdata import get_underlyings_map
from ...constants.underlying_type import UnderlyingType
from ...infrastructure import market_utils
from ...infrastructure.market import Market
from ...infrastructure.volatility_surface import VolatilitySurface


class TestEQNXValuer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEQNXValuer, self).__init__(*args, **kwargs)

        self.rebase = False

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_vol_wap(self):
        # 1) create trade
        dt = datetime(2024, 10, 22)
        und = "SPX Index"
        ccy = "USD"
        inception = datetime(2024, 10, 22)
        expiration = datetime(2025, 10, 22)
        notional = 1
        strike_in_vol = 17.69

        vol_swap = VolSwap(
            underlying=und, inception=inception, expiration=expiration, strike_in_vol=strike_in_vol,
            notional=notional, currency=ccy, asset_class=AssetClass.EQUITY)

        # 2) create market
        ref_time = 'NYC|CLOSE|{}'.format(dt.strftime('%Y%m%d'))
        und_map = get_underlyings_map(return_df=False)
        und_id = und_map[und]
        surface = VolatilitySurface.load(underlying_id=und_id, ref_time=ref_time, underlying_type=UnderlyingType.EQUITY, underlying_full_name=und)
        surface = surface.override_num_regular(num_regular=252)
        market = Market(base_datetime=dt)
        market.add_item(market_utils.create_vol_surface_key(und), surface)

        # 3) calculate fair strike using default DUPIRE model
        model_param_overrides = dict(
            sim_path=3000,
            calib_time_steps=300,
            pricing_time_steps=300,
            # model_name="DUPIRE",
        )
        pv = EQVolSwapNxValuer(override_model_params=model_param_overrides).price(vol_swap, market)
        print(pv)
        self.assertAlmostEqual(pv, -0.5630071747458658, 8)
