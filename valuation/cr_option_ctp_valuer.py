import unittest
from datetime import datetime, timedelta
import ctp.instruments.swaps as swaps
from ctp.instruments.specs import OptionType, BuySell
from ctp.utils.time import Date, DayDuration, FrequencyAnnual, \
    BusinessDayConvention, TimePoint, MicroDuration, GregorianDate, HolidayCalendarCpp, date_to_gregorian_date
from ctp.specifications.daycount import DayCountConvention
from ctp.termstructures.common import InterestRateTermStructure
from ctp.specifications.defs import RecoveryRate, Strike, Volatility, BasePrice, AnnualRate, Notional
from ctp.pricing.engines import PedersenCDSIndexOptionEngine
from ctp.instruments.swaps import CDSIndex
from ctp.specifications.daycount import Actual360, Actual365
from ctp.specifications.currency import Currency
from ..infrastructure.market import Market
from ..infrastructure.market_utils import create_cr_vol_surface_key
from ..tradable.option import Option
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
from ctp.instruments.options import CDSIndexSwaption


IMM_DAY_MAP = {
    'CDX_NA_HY': 27,
    'CDX_NA_IG': 20,
}

ROLL_AND_MATURITY_MAP = {
    'CDX_NA_HY': {3: {'roll_day': 27, 'maturity_month': 6, 'maturity_day': 20},
                  9: {'roll_day': 27, 'maturity_month': 12, 'maturity_day': 20}},
    'CDX_NA_IG': {3: {'roll_day': 20, 'maturity_month': 6, 'maturity_day': 20},
                  9: {'roll_day': 20, 'maturity_month': 12, 'maturity_day': 20}},
}

COUPON_RATE_MAP = {
    'CDX_NA_HY': 500.0 / 10_000.0,
    'CDX_NA_IG': 100.0 / 10_000.0,
}


# def get_previous_imm_date_1(trade_date: datetime) -> datetime:
#     trade_year = trade_date.year
#     imm_dates = [datetime(trade_year - 1, 12, 20), datetime(trade_year, 3, 20),
#                  datetime(trade_year, 6, 20), datetime(trade_year, 9, 20),
#                  datetime(trade_year, 12, 20)]
#     return max([imm_date for imm_date in imm_dates if imm_date <= trade_date])

def get_previous_imm_date(trade_date: datetime, underlying: str) -> datetime:
    imm_day = IMM_DAY_MAP.get(underlying, 20)
    imm_months = [3, 6, 9, 12]
    trade_year = trade_date.year
    imm_dates = [datetime(trade_year, month, imm_day) for month in imm_months]
    for imm_date in reversed(imm_dates):
        if trade_date >= imm_date:
            return imm_date
    return datetime(trade_year - 1, 12, imm_day)

# def timing_functions():
#     import time
#     imm_dates = [datetime(2025, m, 20) for m in [3, 6, 9, 12]]
#     dates_to_test = [date for dt in imm_dates for date in [dt + timedelta(days=-1), dt, dt + timedelta(days=1)]]
#     exp_results = [datetime(2024, 12, 20), datetime(2025, 3, 20), datetime(2025, 3, 20), datetime(2025, 3, 20),
#                    datetime(2025, 6, 20), datetime(2025, 6, 20), datetime(2025, 6, 20), datetime(2025, 9, 20),
#                    datetime(2025, 9, 20), datetime(2025, 9, 20), datetime(2025, 12, 20), datetime(2025, 12, 20)]
#
#     start_time = time.time()
#     test_result = all([get_previous_imm_date_1(dt) == exp_results[i] for i, dt in enumerate(dates_to_test)])
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#
#     print("Test Result for 1:", test_result)
#     print("Elapsed Time (seconds):", elapsed_time)
#
#     start_time = time.time()
#     test_result = all([get_previous_imm_date_2(dt) == exp_results[i] for i, dt in enumerate(dates_to_test)])
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#
#     # Print results
#     print("Test Result for 2:", test_result)
#     print("Elapsed Time (seconds):", elapsed_time)


def test_get_previous_imm_date():
    imm_dates = [datetime(2025, m, 20) for m in [3, 6, 9, 12]]
    dates_to_test = [date for dt in imm_dates for date in [dt + timedelta(days=-1), dt, dt + timedelta(days=1)]]
    exp_results = [datetime(2024, 12, 20), datetime(2025, 3, 20), datetime(2025, 3, 20), datetime(2025, 3, 20),
                   datetime(2025, 6, 20), datetime(2025, 6, 20), datetime(2025, 6, 20), datetime(2025, 9, 20),
                   datetime(2025, 9, 20), datetime(2025, 9, 20), datetime(2025, 12, 20), datetime(2025, 12, 20)]
    test_result = all([get_previous_imm_date(dt, underlying="CDX_NA_IG") == exp_results[i] for i, dt in enumerate(dates_to_test)])
    print("Test Result for get_previous_imm_date:", test_result)


# def get_cds_maturity_date(trade_date: datetime, underlying: str, tenor: int = 5) -> datetime:
#
#     if underlying not in ROLL_MONTHS or underlying not in ROLL_DAYS or underlying not in MATURITY_MONTH_MAP:
#         raise ValueError(f"Unsupported underlying: {underlying}")
#     roll_months = ROLL_MONTHS[underlying]
#     roll_day = ROLL_DAYS[underlying]
#     roll_dates = [datetime(trade_date.year, month, roll_day) for month in roll_months]
#     next_roll_date = min((roll_date for roll_date in roll_dates if roll_date > trade_date), default=None)
#     if next_roll_date is None:
#         next_roll_date = datetime(trade_date.year + 1, roll_months[0], roll_day)
#     maturity_year = next_roll_date.year + tenor
#     maturity_month = MATURITY_MONTH_MAP[underlying].get(next_roll_date.month)
#     maturity_date = next_roll_date.replace(year=maturity_year, month=maturity_month)
#     return maturity_date

def get_cds_maturity_date(trade_date: datetime, underlying: str, tenor: int = 5) -> datetime:

    if underlying not in ROLL_AND_MATURITY_MAP:
        raise ValueError(f"Unsupported underlying: {underlying}")

    roll_map = ROLL_AND_MATURITY_MAP[underlying]
    roll_dates = [
        datetime(trade_date.year, month, roll_map[month]['roll_day']) for month in roll_map
    ]
    next_roll_date = min((roll_date for roll_date in roll_dates if roll_date > trade_date), default=None)

    if next_roll_date is None:
        next_year_roll_month = list(roll_map.keys())[0]
        next_roll_date = datetime(trade_date.year + 1, next_year_roll_month, roll_map[next_year_roll_month]['roll_day'])

    maturity_month = roll_map[next_roll_date.month]['maturity_month']
    maturity_day = roll_map[next_roll_date.month]['maturity_day']
    maturity_date = next_roll_date.replace(year=next_roll_date.year + tenor, month=maturity_month, day=maturity_day)
    return maturity_date


hy_test_cases = [
    datetime(2023, 9, 26),  # Before September roll
    datetime(2023, 9, 28),  # After September roll
    datetime(2023, 3, 27),  # On March roll
    datetime(2023, 3, 28),  # After March roll
    datetime(2023, 12, 31),  # After all rolls in the year
]

ig_test_cases = [
    datetime(2023, 9, 19),  # Before September roll
    datetime(2023, 9, 21),  # After September roll
    datetime(2023, 3, 20),  # On March roll
    datetime(2023, 3, 21),  # After March roll
    datetime(2023, 12, 31),  # After all rolls in the year
]

expected_results = [
    datetime(2028, 12, 20),
    datetime(2029, 6, 20),
    datetime(2028, 12, 20),
    datetime(2028, 12, 20),
    datetime(2029, 6, 20),
]


def test_get_cds_maturity_date(test_cases, underlying, expected_results):
    for i, trade_date in enumerate(test_cases):
        try:
            result = get_cds_maturity_date(trade_date, underlying)
            assert result == expected_results[i], f"Test failed for {trade_date}: expected {expected_results[i]}, got {result}"
            print(f"Test passed for {trade_date}: {result}")
        except ValueError as e:
            print(f"ValueError for {trade_date}: {e}")


class CROptionCTPValuer(IValuer):
    def __init__(self, discount_curve_name: str):
        self.ref = None
        self.dt = None
        self.discount_curve = None
        self.spot = None
        self.df_curve = None
        self.cds_index_option = None
        self.cds_index_five_years = None
        self.currency = None
        self.day_counter_actual_365 = None
        self.day_counter_actual_360 = None
        self.cal = None
        self.ref_date = None
        self.valuation_time_point = None
        self.option = None
        self.market = None
        self.discount_curve_name = discount_curve_name

    @staticmethod
    def _initialize_calendar() -> HolidayCalendarCpp:
        name = "test"
        start_date = GregorianDate(1900, 1, 1)
        end_date = GregorianDate(2100, 12, 31)
        hols = []
        weekends = False
        return HolidayCalendarCpp(name, start_date, end_date, hols, weekends)

    def build_discount_curve(self) -> InterestRateTermStructure:
        day_count_convention = DayCountConvention.A365
        df_curve = {self.dt: self.market.get_spot_rates(self.option.currency, self.discount_curve_name).data_dict}
        df_curve_sorted = dict(sorted(df_curve[self.dt].items()))
        if self.discount_curve_name != "SOFRRATE":
            i_days = [int(year * 365) for year in df_curve_sorted.keys()]
            i_rates = [rate / 100 for rate in df_curve_sorted.values()]
        else:
            i_days, i_rates = list(df_curve_sorted.keys()), list(df_curve_sorted.values())

        i_days.insert(0, 0)
        i_rates.insert(0, 0)
        rates = [(DayDuration(i_days[i]), AnnualRate(i_rates[i])) for i in range(len(i_days))]

        return InterestRateTermStructure.make(
            self.ref_date,
            day_count_convention,
            self.currency,
            rates
        )

    def _initialize_cds_index(self):
        self.cds_index_five_years = CDSIndex(
            name="CDX IG S44 V1 5Y",  # this is a dummy name, actual name is not used in pricing
            first_accrual_date=date_to_gregorian_date(get_previous_imm_date(self.dt, self.option.underlying)),
            maturity_date=get_cds_maturity_date(trade_date=self.dt, underlying=self.option.underlying, tenor=5),
            coupon_rate=AnnualRate(COUPON_RATE_MAP.get(self.option.underlying)),
            coupon_frequency=FrequencyAnnual.Quarterly,
            day_counter=self.day_counter_actual_360,
            currency=self.currency,
            notional=Notional(1),
            accrued_paid_on_default=True,
            start_date_protected=True,
            recovery_rate=RecoveryRate(0.4),
            pay_type=swaps.PayType.DefaultTime,
            trades_on_price=False,
            index_factor=1,
            pay_calendar=self.cal,
            pay_convention=BusinessDayConvention.FOLLOWING,
            settle_calendar=self.cal,
            settle_offset=DayDuration(3),
        )

    def _initialize_cds_index_option(self):
        option_type = OptionType.RECEIVER if self.option.is_call else OptionType.PAYER
        exp = self.option.expiration
        self.cds_index_option = CDSIndexSwaption.make(
            BuySell.BUY,
            Notional(1),
            option_type,
            Strike(self.option.strike),
            self.valuation_time_point,
            TimePoint(Date(exp.year, exp.month, exp.day), MicroDuration(0, 0, 0)),
            False
        )

    def price(self, option: Option, market: Market, calc_types='price', return_struc=False, **kwargs):
        self.market = market
        self.option = option

        self.dt = market.get_base_datetime()
        self.valuation_time_point = TimePoint(Date(self.dt.year, self.dt.month, self.dt.day),
                                              MicroDuration(self.dt.hour, 0, 0))  # datetime_to_timepoint
        self.ref_date = self.valuation_time_point.date

        self.cal = self._initialize_calendar()

        self.day_counter_actual_360 = Actual360()
        self.day_counter_actual_365 = Actual365()

        self.currency = Currency.names[self.option.currency]

        self._initialize_cds_index()
        self._initialize_cds_index_option()

        self.discount_curve = self.build_discount_curve()

        vol_surface = market.get_cr_vol_surface(option.underlying)[option.specialisation]
        spot = market.get_spot(option.underlying).spot[option.specialisation]
        ref = BasePrice(spot)

        engine = PedersenCDSIndexOptionEngine.make(
            self.valuation_time_point,
            self.cds_index_five_years,
            ref,
            self.discount_curve,
            vol_surface,
            True
        )

        self.cds_index_option.setPricingEngine(engine)
        calc_results = self.cds_index_option.calculateAllResults()

        results = {
            'price': calc_results.getResult('VALUE'),
            'delta': calc_results.getResult('DELTA'),
            'vega': calc_results.getResult('VEGA'),
        }

        if return_struc:
            return results
        return valuer_utils.return_results_based_on_dictionary(calc_types, results)


def test_import():
    test_get_cds_maturity_date(hy_test_cases, 'CDX_NA_HY', expected_results)
    pass


if __name__ == "__main__":
    # Run the test cases for get_cds_maturity_date
    print("Testing get_cds_maturity_date with HY underlyings:")
    test_get_cds_maturity_date(hy_test_cases, 'CDX_NA_HY', expected_results)

    print("\nTesting get_cds_maturity_date with IG underlyings:")
    test_get_cds_maturity_date(ig_test_cases, 'CDX_NA_IG', expected_results)

    # Run the test for get_previous_imm_date
    print("\nTesting get_previous_imm_date:")
    test_get_previous_imm_date()