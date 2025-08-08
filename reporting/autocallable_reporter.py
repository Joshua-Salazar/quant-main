from ..interface.itradereporter import ITradeReporter
from ..interface.imarket import IMarket
from ..interface.ivaluer import IValuer
from ..infrastructure.fixing_requirement import FixingRequirement
from ..infrastructure.holiday_requirement import HolidayRequirement
from ..tradable.autocallable import AutoCallable
from ..valuation.eq_nx_valuer import EQNxValuer
from ..valuation.valuer_factory import ValuerFactory
from datetime import timedelta
from dateutil.relativedelta import relativedelta


class AutoCallableReporter(ITradeReporter):
    def __init__(self, autocallable: AutoCallable):
        self.tradable = autocallable

    def get_fixing_requirement(self, market: IMarket):
        dt = market.get_base_datetime()
        fixings = []
        # only request fixings after inception date
        # ac start date is only date no time/timezone
        if dt.date() >= self.tradable.start_date.date():
            for und in self.tradable.get_underlyings():
                fixings.append(FixingRequirement(und, self.tradable.start_date.date(), dt.date()))
            # skip today for sofr rate
            is_today = dt.date() == market.get_base_datetime().date()
            dt_sofr = dt - timedelta(days=1) if is_today else dt
            if dt_sofr.date() >= self.tradable.start_date.date():
                fixings.append(FixingRequirement(self.tradable.rate_index, self.tradable.start_date.date(), dt_sofr.date()))
        return fixings

    def get_holiday_requirement(self, market: IMarket, valuer: IValuer=None):
        dt = market.get_base_datetime()
        codes = [self.tradable.cdr_code, self.tradable.index_cdr_code]
        holidays = []
        st = min(dt.date(), self.tradable.start_date.date())
        et = self.tradable.expiration.date()
        if valuer is None:
            valuer = ValuerFactory().get_valuer(self.tradable)
        if isinstance(valuer, EQNxValuer):
            max_dt = dt + relativedelta(days=EQNxValuer.MAX_VOL_DAYS)
            et = max(et, max_dt.date())
        for code in codes:
            holidays.append(HolidayRequirement(code, st, et))
        return holidays
