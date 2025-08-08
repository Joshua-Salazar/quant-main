from ..interface.imarket import IMarket
from ..interface.itradable import ITradable
from ..interface.ivaluer import IValuer
from ..tradable.autocallable import AutoCallable
from ..reporting.autocallable_reporter import AutoCallableReporter
from ..tradable.cash import Cash
from ..reporting.cash_reporter import CashReporter
from ..tradable.condvarianceswap import CondVarianceSwap
from ..reporting.condvarianceswap_reporter import CondVarianceSwapReporter
from ..tradable.constant import Constant
from ..reporting.constant_reporter import ConstantReporter
from ..tradable.crosscondvarianceswap import CrossCondVarianceSwap
from ..reporting.crosscondvarianceswap_reporter import CrossCondVarianceSwapReporter
from ..tradable.crosscondvolswap import CrossCondVolSwap
from ..reporting.crosscondvolswap_reporter import CrossCondVolSwapReporter
from ..tradable.forwardstartoption import ForwardStartOption
from ..reporting.forwardstartoption_reporter import ForwardStartOptionReporter
from ..tradable.forwardstartswap import ForwardStartSwap
from ..reporting.forwardstartswap_reporter import ForwardStartSwapReporter
from ..tradable.future import Future
from ..reporting.future_reporter import FutureReporter
from ..tradable.fva import FVA
from ..reporting.fva_reporter import FVAReporter
from ..tradable.FXforward import FXforward
from ..reporting.FXforward_reporter import FXforwardReporter
from ..tradable.fxspot import FXSpot
from ..reporting.FXspot_reporter import FXSpotReporter
from ..tradable.geodispvarswap import GeoDispVarSwap
from ..reporting.geodispvarswap_reporter import GeoDispVarSwapReporter
from ..tradable.geodispvolswap import GeoDispVolSwap
from ..reporting.geodispvolswap_reporter import GeoDispVolSwapReporter
from ..tradable.option import Option
from ..reporting.option_reporter import OptionReporter
from ..tradable.stock import Stock
from ..reporting.stock_reporter import StockReporter
from ..tradable.swaption import Swaption
from ..reporting.swaption_reporter import SwaptionReporter
from ..tradable.vanillaswap import VanillaSwap
from ..reporting.vanillaswap_reporter import VanillaSwapReporter
from ..tradable.varianceswap import VarianceSwap
from ..reporting.varianceswap_reporter import VarianceSwapReporter
from ..tradable.volswap import VolSwap
from ..reporting.volswap_reporter import VolSwapReporter
from ..tradable.voloption import VolOption
from ..reporting.voloption_reporter import VolOptionReporter
from ..tradable.xccyswap import XCcySwap
from ..reporting.xccyswap_reporter import XCcySwapReporter


class TradeReporter:
    def __init__(self, tradable: ITradable):
        if isinstance(tradable, AutoCallable):
            self.reporter = AutoCallableReporter(tradable)
        elif isinstance(tradable, Cash):
            self.reporter = CashReporter(tradable)
        elif isinstance(tradable, CondVarianceSwap):
            self.reporter = CondVarianceSwapReporter(tradable)
        elif isinstance(tradable, Constant):
            self.reporter = ConstantReporter(tradable)
        elif isinstance(tradable, CrossCondVarianceSwap):
            self.reporter = CrossCondVarianceSwapReporter(tradable)
        elif isinstance(tradable, CrossCondVolSwap):
            self.reporter = CrossCondVolSwapReporter(tradable)
        elif isinstance(tradable, ForwardStartOption):
            self.reporter = ForwardStartOptionReporter(tradable)
        elif isinstance(tradable, ForwardStartSwap):
            self.reporter = ForwardStartSwapReporter(tradable)
        elif isinstance(tradable, Future):
            self.reporter = FutureReporter(tradable)
        elif isinstance(tradable, FVA):
            self.reporter = FVAReporter(tradable)
        elif isinstance(tradable, FXforward):
            self.reporter = FXforwardReporter(tradable)
        elif isinstance(tradable, FXSpot):
            self.reporter = FXSpotReporter(tradable)
        elif isinstance(tradable, GeoDispVarSwap):
            self.reporter = GeoDispVarSwapReporter(tradable)
        elif isinstance(tradable, GeoDispVolSwap):
            self.reporter = GeoDispVolSwapReporter(tradable)
        elif isinstance(tradable, Option):
            self.reporter = OptionReporter(tradable)
        elif isinstance(tradable, Stock):
            self.reporter = StockReporter(tradable)
        elif isinstance(tradable, Swaption):
            self.reporter = SwaptionReporter(tradable)
        elif isinstance(tradable, VanillaSwap):
            self.reporter = VanillaSwapReporter(tradable)
        elif isinstance(tradable, VarianceSwap):
            self.reporter = VarianceSwapReporter(tradable)
        elif isinstance(tradable, VolSwap):
            self.reporter = VolSwapReporter(tradable)
        elif isinstance(tradable, VolOption):
            self.reporter = VolOptionReporter(tradable)
        elif isinstance(tradable, XCcySwap):
            self.reporter = XCcySwapReporter(tradable)
        else:
            raise Exception(f"Missing reporter for {tradable}")

    def get_cash_flows(self, market: IMarket):
        return self.reporter.get_cash_flows(market)

    def get_fixing_requirement(self, market: IMarket):
        return self.reporter.get_fixing_requirement(market)

    def get_holiday_requirement(self, market: IMarket, valuer: IValuer=None):
        return self.reporter.get_holiday_requirement(market, valuer)

    def get_holidays(self, market: IMarket, valuer: IValuer=None):
        holiday_requirement = self.get_holiday_requirement(market, valuer)
        holidays = []
        for hol_req in holiday_requirement:
            holidays += market.get_holidays(hol_req.code, hol_req.start_date, hol_req.end_date)
        holidays = sorted(set(holidays))
        return holidays
