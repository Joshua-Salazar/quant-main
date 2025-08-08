import unittest
import ctp.instruments.swaps as swaps
import ctp.specifications.defs as defs
from ctp.specifications.daycount import Actual360
from ctp.specifications.currency import Currency
from ctp.utils.time import Date, DayDuration, FrequencyAnnual, BusinessDayConvention, TimePoint, MicroDuration, \
    GregorianDate, HolidayCalendarCpp
from ctp.specifications.daycount import Actual365
from ctp.pricing.engines import ISDACDSIndexEngine, CreditData
from ctp.instruments.swaps import CDSIndex
from ctp.specifications.daycount import DayCountConvention
from ctp.termstructures.common import InterestRateTermStructure


class Test(unittest.TestCase):
    def setUp(self):

        # pricing date
        self.valuation_time_point = TimePoint(Date(2025, 3, 24), MicroDuration(16, 0, 0))
        self.ref_date = self.valuation_time_point.date

        # calendar
        name = "CDSCalendar"
        start_date = GregorianDate(1900, 1, 1)
        end_date = GregorianDate(2100, 12, 31)
        hols = []
        weekends = False
        self.cal = HolidayCalendarCpp(name, start_date, end_date, hols, weekends)

        # day counters
        self.day_counter_actual_360 = Actual360()
        self.day_counter_actual_365 = Actual365()

        # currency
        self.currency = Currency.USD

        # cds contracts
        self.cds_index_three_years = CDSIndex(
            name="CDX IG S44 V1 3Y"
            , first_accrual_date=GregorianDate(2025, 3, 20)
            , maturity_date=GregorianDate(2028, 12, 20)
            , coupon_rate=defs.AnnualRate(100.0 / 10000.0)
            , coupon_frequency=FrequencyAnnual.Quarterly
            , day_counter=self.day_counter_actual_360
            , currency=self.currency
            , notional=defs.Notional(1)
            , accrued_paid_on_default=True
            , start_date_protected=True
            , recovery_rate=defs.RecoveryRate(0.4)
            , pay_type=swaps.PayType.DefaultTime
            , trades_on_price=False
            , index_factor=1
            , pay_calendar=self.cal
            , pay_convention=BusinessDayConvention.FOLLOWING
            , settle_calendar=self.cal
            , settle_offset=DayDuration(3)
        )

        self.cds_index_five_years = CDSIndex(
            name="CDX IG S44 V1 5Y"
            , first_accrual_date=GregorianDate(2025, 3, 20)
            , maturity_date=GregorianDate(2030, 12, 20)
            , coupon_rate=defs.AnnualRate(100.0 / 10000.0)
            , coupon_frequency=FrequencyAnnual.Quarterly
            , day_counter=self.day_counter_actual_360
            , currency=self.currency
            , notional=defs.Notional(1)
            , accrued_paid_on_default=True
            , start_date_protected=True
            , recovery_rate=defs.RecoveryRate(0.4)
            , pay_type=swaps.PayType.DefaultTime
            , trades_on_price=False
            , index_factor=1
            , pay_calendar=self.cal
            , pay_convention=BusinessDayConvention.FOLLOWING
            , settle_calendar=self.cal
            , settle_offset=DayDuration(3)
        )

        self.cds_index_seven_years = CDSIndex(
            name="CDX IG S44 V1 7Y"
            , first_accrual_date=GregorianDate(2025, 3, 20)
            , maturity_date=GregorianDate(2032, 12, 20)
            , coupon_rate=defs.AnnualRate(100.0 / 10000.0)
            , coupon_frequency=FrequencyAnnual.Quarterly
            , day_counter=self.day_counter_actual_360
            , currency=self.currency
            , notional=defs.Notional(1)
            , accrued_paid_on_default=True
            , start_date_protected=True
            , recovery_rate=defs.RecoveryRate(0.4)
            , pay_type=swaps.PayType.DefaultTime
            , trades_on_price=False
            , index_factor=1
            , pay_calendar=self.cal
            , pay_convention=BusinessDayConvention.FOLLOWING
            , settle_calendar=self.cal
            , settle_offset=DayDuration(3)
        )

        self.cds_index_ten_years = CDSIndex(
            name="CDX IG S44 V1 10Y"
            , first_accrual_date=GregorianDate(2025, 3, 20)
            , maturity_date=GregorianDate(2035, 12, 20)
            , coupon_rate=defs.AnnualRate(100.0 / 10000.0)
            , coupon_frequency=FrequencyAnnual.Quarterly
            , day_counter=self.day_counter_actual_360
            , currency=self.currency
            , notional=defs.Notional(1)
            , accrued_paid_on_default=True
            , start_date_protected=True
            , recovery_rate=defs.RecoveryRate(0.4)
            , pay_type=swaps.PayType.DefaultTime
            , trades_on_price=False
            , index_factor=1
            , pay_calendar=self.cal
            , pay_convention=BusinessDayConvention.FOLLOWING
            , settle_calendar=self.cal
            , settle_offset=DayDuration(3)
        )

        self.discount_curve = self.build_discount_curve()

        self.stepInDay = DayDuration(1)

        print("setup ok")

    def build_discount_curve(self):
        day_count_convention = DayCountConvention.A365

        i_days = [1, 30, 60, 90, 180, 270, 360, 540, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3600, 4320, 5400,
                  7200, 9000, 10800, 14400, 18000]
        i_rates = [0.042999, 0.044469, 0.043873, 0.044043, 0.043203, 0.04191, 0.040886, 0.039607, 0.038692, 0.038086,
                   0.038075, 0.038181, 0.038358, 0.038529, 0.038762, 0.03896, 0.039166, 0.039581, 0.040063, 0.040263,
                   0.039793, 0.039065, 0.037339, 0.03568]

        rates = [(DayDuration(i_days[i]), defs.AnnualRate(i_rates[i])) for i in range(len(i_days))]

        discount_curve = InterestRateTermStructure.make(
            self.ref_date,
            day_count_convention,
            self.currency,
            rates
        )

        return discount_curve

    def test_settlement_value(self):
        print('=== START ===')

        step_in_date = self.ref_date + self.stepInDay

        settlement_date = self.ref_date + self.cds_index_five_years.settlement_offset

        i_contracts = [self.cds_index_three_years, self.cds_index_five_years, self.cds_index_seven_years, self.cds_index_ten_years]
        i_spreads = [42.9025, 63.6925, 83.5425, 101.303]

        credit_data = CreditData.make(i_contracts, i_spreads)

        engine = ISDACDSIndexEngine.make(
            self.cds_index_five_years,
            self.valuation_time_point,
            step_in_date,
            settlement_date,
            self.discount_curve,
            credit_data,
            True
        )

        self.cds_index_five_years.setPricingEngine(engine)
        r = self.cds_index_five_years.calculateAllResults()

        # all results in local currency
        theo = r.getResult('VALUE')
        discount_factor = r.getResult('DISCOUNT_FACTOR')
        cr01 = r.getResult('CR01')

        print('==== END ====')


if __name__ == '__main__':
    unittest.main()
