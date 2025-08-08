from ..interface.itradable import ITradable
from ..interface.ivaluer import IValuer

from ..tradable.cash import Cash
from ..valuation.cash_valuer import CashValuer

from ..tradable.constant import Constant
from ..valuation.constant_valuer import ConstantValuer

from ..tradable.forwardstartswap import ForwardStartSwap
from ..valuation.forwardstartswap_valuer import ForwardStartSwapValuer

from ..tradable.future import Future
from ..valuation.future_valuer import FutureValuer

from ..tradable.option import Option
from ..tradable.FXforward import FXforward
from ..valuation.fx_forward_outright_data_valuer import FXForwardOutrightDataValuer

from ..tradable.stock import Stock
from ..valuation.stock_data_valuer import StockDataValuer

from ..tradable.fxspot import FXSpot
from ..valuation.fx_data_valuer import FXDataValuer
from ..valuation.fx_sabr_vol_surface_valuer import FXSABRVolSurfaceValuer

from ..tradable.swaption import Swaption
from ..valuation.swaption_norm_valuer import SwaptionNormValuer

from ..tradable.varianceswap import VarianceSwap
from ..valuation.varianceswap_replication_valuer import VarianceSwapReplicationValuer

from ..tradable.condvarianceswap import CondVarianceSwap
from ..tradable.volswap import VolSwap
from ..tradable.voloption import VolOption

from ..tradable.xccyswap import XCcySwap
from ..valuation.xccyswap_valuer import XCcySwapValuer

from ..constants.ccy import Ccy


class ValuerFactory:
    def __init__(self):
        pass

    def get_valuer(self, tradable: ITradable, **kwargs) -> IValuer:
        if isinstance(tradable, Cash):
            return CashValuer()
        elif isinstance(tradable, Constant):
            return ConstantValuer()
        elif isinstance(tradable, ForwardStartSwap):
            return ForwardStartSwapValuer()
        elif isinstance(tradable, Future):
            return FutureValuer()
        elif isinstance(tradable, Option):
            from ..valuation.fx_cr_option_bs_valuer import FXOptionBSValuer
            from ..valuation.option_vola_valuer import OptionVolaValuer
            return FXOptionBSValuer() if Ccy.contains(tradable.underlying) else OptionVolaValuer()
        elif isinstance(tradable, Stock):
            return StockDataValuer()
        elif isinstance(tradable, FXSpot):
            return FXSABRVolSurfaceValuer()
        elif isinstance(tradable, Swaption):
            return SwaptionNormValuer()
        elif isinstance(tradable, VarianceSwap):
            return VarianceSwapReplicationValuer()
        elif isinstance(tradable, CondVarianceSwap):
            from ..valuation.eq_condvarswap_nx_valuer import EQCondVarSwapNxValuer
            return EQCondVarSwapNxValuer()
        elif isinstance(tradable, VolSwap):
            from ..valuation.fx_volswap_nx_valuer import FXVolSwapNxValuer
            return FXVolSwapNxValuer()
        elif isinstance(tradable, VolOption):
            from ..valuation.fx_voloption_nx_valuer import FXVolOptionNxValuer
            return FXVolOptionNxValuer()
        elif isinstance(tradable, XCcySwap):
            return XCcySwapValuer()
        elif isinstance(tradable, FXforward):
            return FXForwardOutrightDataValuer()
        else:
            from ..tradable.autocallable import AutoCallable
            from ..valuation.eq_autocallable_nx_valuer import EQAutoCallableNXValuer
            if isinstance(tradable, AutoCallable):
                return EQAutoCallableNXValuer()

        raise Exception(f"Missing valuer for {tradable}")





