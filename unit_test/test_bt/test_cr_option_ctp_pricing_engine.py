import unittest
import datetime
import ctp.instruments.swaps as swaps
from ctp.instruments.specs import OptionType, BuySell
from ctp.specifications.daycount import Actual360
from ctp.specifications.currency import Currency
from ctp.utils.time import Date, DayDuration, FrequencyAnnual, \
    BusinessDayConvention, TimePoint, MicroDuration, GregorianDate, HolidayCalendarCpp
from ctp.specifications.daycount import DayCountConvention
from ctp.termstructures.common import InterestRateTermStructure
from ctp.specifications.defs import RecoveryRate, Strike, Volatility, BasePrice, AnnualRate, Notional
from ctp.instruments.options import CDSIndexSwaption
from ctp.models.vol import ExpiryStrikeVolPoint, CubicSplineVolatility
from ctp.specifications.daycount import Actual365
from ctp.pricing.engines import PedersenCDSIndexOptionEngine
from ctp.instruments.swaps import CDSIndex


class Test(unittest.TestCase):
    def setUp(self):
        # pricing date
        self.valuation_time_point = TimePoint(Date(2025, 3, 24), MicroDuration(16, 0, 0))
        self.ref_date = self.valuation_time_point.date

        # calendar
        name = "test"
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

        # cds index underlying
        self.cds_index_five_years = CDSIndex(
            name="CDX IG S44 V1 5Y"
            , first_accrual_date=GregorianDate(2025, 9, 20)
            , maturity_date=GregorianDate(2030, 12, 20)
            , coupon_rate=AnnualRate(100.0 / 10000.0)
            , coupon_frequency=FrequencyAnnual.Quarterly
            , day_counter=self.day_counter_actual_360
            , currency=self.currency
            , notional=Notional(1)
            , accrued_paid_on_default=True
            , start_date_protected=True
            , recovery_rate=RecoveryRate(0.4)
            , pay_type=swaps.PayType.DefaultTime
            , trades_on_price=False
            , index_factor=1
            , pay_calendar=self.cal
            , pay_convention=BusinessDayConvention.FOLLOWING
            , settle_calendar=self.cal
            , settle_offset=DayDuration(3)
        )

        # cds index option
        self.cds_index_option = CDSIndexSwaption.make(
            BuySell.BUY,
            Notional(1),
            OptionType.PAYER,
            Strike(60),
            TimePoint(Date(2025, 3, 24), MicroDuration(16, 0, 0)),
            TimePoint(Date(2025, 6, 18), MicroDuration(0, 0, 0)),
            False
        )

        self.discount_curve = self.build_discount_curve()

        self.vol_surface = self.build_vol_surface()

        # We build a flat survival curve using the underlying reference
        self.ref = BasePrice(76.46)

        print("setup ok")

    def build_discount_curve(self):
        day_count_convention = DayCountConvention.A365

        i_days = [1, 30, 60, 90, 180, 270, 360, 540, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3600, 4320, 5400,
                  7200, 9000, 10800, 14400, 18000]
        i_rates = [0.042999, 0.044469, 0.043873, 0.044043, 0.043203, 0.04191, 0.040886, 0.039607, 0.038692, 0.038086,
                   0.038075, 0.038181, 0.038358, 0.038529, 0.038762, 0.03896, 0.039166, 0.039581, 0.040063, 0.040263,
                   0.039793, 0.039065, 0.037339, 0.03568]

        rates = [(DayDuration(i_days[i]), AnnualRate(i_rates[i])) for i in range(len(i_days))]

        discount_curve = InterestRateTermStructure.make(
            self.ref_date,
            day_count_convention,
            self.currency,
            rates
        )

        return discount_curve

    def build_vol_surface(self):

        vol_points = []

        expiries = ['20250416', '20250416', '20250416', '20250416', '20250416', '20250416', '20250416', '20250416',
                    '20250416', '20250416', '20250416', '20250416', '20250521', '20250521', '20250521', '20250521',
                    '20250521', '20250521', '20250521', '20250521', '20250521', '20250521', '20250521', '20250521',
                    '20250521', '20250521', '20250521', '20250521', '20250618', '20250618', '20250618', '20250618',
                    '20250618', '20250618', '20250618', '20250618', '20250618', '20250618', '20250618', '20250618',
                    '20250618', '20250618', '20250618', '20250618', '20250618', '20250716', '20250716', '20250716',
                    '20250716', '20250716', '20250716', '20250716', '20250716', '20250716', '20250716', '20250716',
                    '20250716', '20250716', '20250716', '20250716', '20250716', '20250820', '20250820', '20250820',
                    '20250820', '20250820', '20250820', '20250820', '20250820', '20250820', '20250820', '20250820',
                    '20250820', '20250820', '20250820']

        strikes = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0,
                   45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                   105.0, 110.0, 115.0, 120.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
                   85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0, 50.0, 55.0, 60.0,
                   65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0,
                   130.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                   110.0, 120.0, 130.0]
        strikes = [strike / 10000 for strike in strikes]

        vols = [127.75646163627871, 104.89572272089667, 82.00956534220506, 75.84118799506336,
                83.03368284069028, 92.04958291566852, 100.42701446946239, 109.0581733660768,
                118.91076424800592, 145.81323346188168, 160.27444072253846, 188.1570519295221,
                81.96622230721687, 70.13345394023548, 58.780886092688775, 55.08865660600852,
                59.24470216667654, 64.10721423628806, 68.38697916555793, 73.00143624856504,
                76.22785032984444, 81.50792143375854, 84.25664547789502, 88.76341995064925,
                90.49105350189683, 95.01880515488952, 97.42657212089023, 98.46796802629898,
                66.73353844784089, 57.952826050880525, 50.154684544846, 49.621509347438774,
                54.225418585896584, 58.52940272044614, 62.48493243172052, 66.27257625601356,
                69.04440607869108, 73.18038609710621, 75.0792310660283, 79.13623868123372,
                79.25788016767886, 84.09431071634503, 84.27836370819398, 89.13214435339815,
                94.91310416490089, 54.97226379455735, 48.34753078928212, 47.116313968606406,
                51.01989582756906, 54.807539263617095, 58.326285713111496, 61.69322623942636,
                64.94158811811837, 67.64887570136389, 70.45522977568599, 72.70734695855519,
                73.78259615976165, 77.17699129157238, 77.84233974805628, 81.20799882668186,
                86.34814593895453, 50.14152843018784, 45.16353689862339, 45.67258679547408,
                49.32104905477692, 52.77589844382375, 56.08460848474328, 59.03765581417956,
                61.90730136892537, 64.62109816943162, 67.21905554045499, 69.6423799379366,
                73.57481726350119, 77.39009620455167, 80.86339202335213]
        vols = [vol / 100 for vol in vols]

        for i, expiry in enumerate(expiries):
            vol_date = datetime.datetime.strptime(expiry, '%Y%m%d')
            vol_points.append(ExpiryStrikeVolPoint(Date(vol_date.year, vol_date.month, vol_date.day), Strike(strikes[i]), Volatility(vols[i])))

        vol_surface = CubicSplineVolatility(self.day_counter_actual_360, self.valuation_time_point, vol_points)

        return vol_surface

    def test_settlement_value(self):
        print('=== START ===')

        engine = PedersenCDSIndexOptionEngine.make(
            self.valuation_time_point,
            self.cds_index_five_years,
            self.ref,
            self.discount_curve,
            self.vol_surface,
            True
        )

        self.cds_index_option.setPricingEngine(engine)
        r = self.cds_index_option.calculateAllResults()

        # all results in local currency
        val = r.getResult('VALUE')
        delta = r.getResult('DELTA')
        vega = r.getResult('VEGA')

        print('==== END ====')


if __name__ == '__main__':
    unittest.main()