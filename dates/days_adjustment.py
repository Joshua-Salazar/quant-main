from datetime import datetime, timedelta
from ..dates import utils
from ..constants.business_day_convention import BusinessDayConvention


class DaysAdjustment:
    def __init__(self, days: int, holidays: [int],
                 adjustment: BusinessDayConvention = BusinessDayConvention.NONE,
                 use_bds=False):
        self.days = days
        self.holidays = holidays
        self.adjustment = adjustment
        self.use_bds = use_bds

    def apply(self, dt: datetime):
        if self.use_bds:
            return utils.add_business_days(dt, self.days, self.holidays)
        else:
            shifted_dt = dt + timedelta(days=self.days)
            return utils.bdc_adjustment(shifted_dt, self.adjustment, self.holidays)
