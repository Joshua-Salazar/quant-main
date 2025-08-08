from ..interface.itradereporter import ITradeReporter
from ..constants.asset_class import AssetClass
from ..interface.imarket import IMarket
from ..interface.ivaluer import IValuer
from ..infrastructure.fixing_requirement import FixingRequirement
from ..infrastructure.holiday_requirement import HolidayRequirement
from ..tradable.condvarianceswap import CondVarianceSwap
from ..valuation.eq_nx_valuer import EQNxValuer
from ..valuation.valuer_factory import ValuerFactory
from dateutil.relativedelta import relativedelta


class CondVarianceSwapReporter(ITradeReporter):
    def __init__(self, condvarianceswap: CondVarianceSwap):
        self.tradable = condvarianceswap

    def get_fixing_requirement(self, market: IMarket):
        dt = market.get_base_datetime()
        fixings = []
        # only request fixings after inception date
        if dt > self.tradable.inception:
            underlying = self.tradable.fixing_src if len(self.tradable.fixing_src) > 0 else self.tradable.underlying
            fixings.append(FixingRequirement(underlying, self.tradable.inception.date(), dt.date()))
        return fixings

    def get_holiday_requirement(self, market: IMarket, valuer: IValuer=None):
        dt = market.get_base_datetime()
        codes = []
        if len(self.tradable.cdr_code) > 0:
            codes.append(self.tradable.cdr_code)
        if self.tradable.asset_class == AssetClass.FX:
            codes += [self.tradable.underlying[:3], self.tradable.underlying[3:]]
        holidays = []
        st = min(dt.date(), self.tradable.inception.date())
        et = self.tradable.expiration.date()
        if valuer is None:
            valuer = ValuerFactory().get_valuer(self.tradable)
        if isinstance(valuer, EQNxValuer):
            max_dt = dt + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
            et = max(et, max_dt.date())
        for code in codes:
            holidays.append(HolidayRequirement(code, st, et))
        return holidays
