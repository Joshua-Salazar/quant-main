from ..constants.day_count_convention import DayCountConvention
from ..infrastructure.fx_pair import FXPair
from ..interface.itradable import ITradable
from ..constants.business_day_convention import BusinessDayConvention
from ..dates import utils
from ..dates.days_adjustment import DaysAdjustment
from ..constants.ccy import Ccy
from ..constants.frequency_annual import FrequencyAnnual
from ..constants.pay_receive import PayReceive
from dataclasses import dataclass
from datetime import datetime
import calendar


@dataclass
class FXIndex:
    name: str
    fixing_days: int
    holidays: [int]
    ccy_pair: FXPair

    def get_fixing_date_from_effective(self, effective_date: datetime):
        fixing_date = utils.add_business_days(effective_date, -self.fixing_days, self.holidays)
        return fixing_date


@dataclass
class FXResetData:
    fx_index: FXIndex
    fixing_date: datetime
    maturity_date: datetime


class IborIndexDefinition:
    def __init__(self, name: str, fixing_days: int, holidays: [], tenor: str, ccy: Ccy,  dc: int,
                 bdc: BusinessDayConvention):
        """
        :param name: ibor index name
        :param fixing_days: fixing days, e.g. -2
        :param holidays: holidays
        :param tenor: tenor for ibor index
        :param ccy: currency
        :param dc: day counter
        :param bdc: business day convention, e.g. MODIFIEDFOLLOWING
        """
        self.name = name
        self.fixing_days = fixing_days
        self.holidays = holidays
        self.tenor = tenor
        self.ccy = ccy
        self.dc = dc
        self.bdc = bdc

    def get_fixing_date_from_effective(self, effective_date: datetime):
        fixing_date = utils.add_business_days(effective_date, -self.fixing_days, self.holidays)
        return fixing_date

    def get_effective_date_from_fixing(self, fixing_date: datetime):
        effective_date = utils.add_business_days(fixing_date, self.fixing_days, self.holidays)
        return effective_date

    def get_maturity_from_effective(self, effective_date: datetime):
        maturity = utils.add_tenor(effective_date, self.tenor)
        maturity = utils.bdc_adjustment(maturity, self.bdc, self.holidays)
        return maturity


class IborRateData:
    def __init__(self, index: IborIndexDefinition, start_date: datetime):
        self.index = index
        self.fixing_date = index.get_fixing_date_from_effective(start_date)
        self.reset_start = index.get_effective_date_from_fixing(self.fixing_date)
        self.reset_end = index.get_maturity_from_effective(self.reset_start)
        self.reset_year_fraction = (self.reset_end - self.reset_start).days / self.index.dc.value


@dataclass
class RateAccrualPeriod:
    start_date: datetime
    end_date: datetime
    year_fraction: float
    spread: float
    ibor_rate_data: IborRateData


class I_Notional:
    def __init__(self, ccy: Ccy, notional: float, fx_reset_data: FXResetData):
        self.ccy = ccy
        self.notional = notional
        self.fx_reset_data = fx_reset_data

    def get_notional(self, market):
        if self.fx_reset_data is None:
            adj = 1.
        else:
            pair = self.fx_reset_data.fx_index.ccy_pair
            assert pair.contains(self.ccy)
            notional_ccy = pair.term_ccy if self.ccy == pair.base_ccy else pair.base_ccy
            base_dt = market.get_base_datetime()
            if self.fx_reset_data.fixing_date < base_dt:
                adj = market.get_fx_fixing(pair, self.fx_reset_data.fixing_date)
            else:
                adj = market.get_fx_fwd(FXPair(notional_ccy, self.ccy), self.fx_reset_data.fixing_date)
        return adj * self.notional


class RatePaymentPeriod(I_Notional):
    def __init__(self, rate_accrual_period: [RateAccrualPeriod], payment_date: datetime, ccy: Ccy, notional: float,
                 fx_reset_data: FXResetData):
        super().__init__(ccy, notional, fx_reset_data)
        self.rate_accrual_period = rate_accrual_period
        self.payment_date = payment_date


class NotionalExchange(I_Notional):
    def __init__(self, ccy: Ccy, notional: float, payment_date: datetime, fx_reset_data: FXResetData):
        super().__init__(ccy, notional, fx_reset_data)
        self.payment_date = payment_date


@dataclass
class RateCalculationLeg:
    pay_receive: PayReceive
    interest_payments: [RatePaymentPeriod]
    notional_payments: [NotionalExchange]


@dataclass
class IborLeg:
    rate_calculation_leg: RateCalculationLeg
    notional: float
    proj_curve_name: str = "SWAP"
    disc_curve_name: str = "OIS"

    def get_ccy(self):
        return self.rate_calculation_leg.interest_payments[0].ccy

    def get_pair(self):
        fx_reset_data = self.rate_calculation_leg.interest_payments[0].fx_reset_data
        return None if fx_reset_data is None else fx_reset_data.fx_index.ccy_pair


class XCcySwap(ITradable):
    def __init__(self, spread_leg: IborLeg, flat_leg: IborLeg, start_date: datetime = None):
        self.spread_leg = spread_leg
        self.flat_leg = flat_leg
        self.start_date = start_date
        self.expiration = self.spread_leg.rate_calculation_leg.interest_payments[-1].payment_date

    def has_expiration(self):
        return True

    def clone(self):
        return XCcySwap(self.spread_leg, self.flat_leg, self.start_date)

    def name(self):
        tenor = self.spread_leg.rate_calculation_leg.interest_payments[0].rate_accrual_period[0].ibor_rate_data.index.\
            tenor
        pair = self.flat_leg.get_pair() if self.spread_leg.get_pair() is None else self.spread_leg.get_pair()
        if self.start_date is None:
            return pair.to_string() + tenor + self.expiration.strftime('%Y-%m-%d')
        else:
            return pair.to_string() + self.start_date.strftime('%Y-%m-%d') + tenor + self.expiration.strftime('%Y-%m-%d')


class IborLegConvention:
    def __init__(self, accrual_adjustment: DaysAdjustment, payment_offset: DaysAdjustment,
                 index: IborIndexDefinition, payment_frequency: FrequencyAnnual, dc: DayCountConvention,
                 proj_curve_name: str, disc_curve_name: str):
        self.accrual_adjustment = accrual_adjustment
        self.payment_offset = payment_offset
        self.index = index
        self.payment_frequency = payment_frequency
        self.dc = dc
        self.ccy = index.ccy
        self.proj_curve_name = proj_curve_name
        self.disc_curve_name = disc_curve_name

    def roll_backward(self, date_in):
        date_out = utils.add_tenor(date_in, f"-{self.index.tenor}")
        if not (self.index.tenor.endswith('d') or self.index.tenor.endswith('D')):
            # cap month end day
            date_out = date_out.replace(day=min(calendar.monthrange(date_out.year, date_out.month)[1], date_in.day))
        return date_out

    def create_schedule(self, start_date: datetime, end_date: datetime):
        """
        generate schedule rolling backwards
        """""
        dates = [end_date]
        temp = self.roll_backward(end_date)
        temp_adjusted = utils.bdc_adjustment(temp, self.accrual_adjustment.adjustment, self.accrual_adjustment.holidays)
        while temp > start_date:
            dates.append(temp_adjusted)
            temp = self.roll_backward(temp)
            temp_adjusted = utils.bdc_adjustment(temp, self.accrual_adjustment.adjustment, self.accrual_adjustment.holidays)
        # check start date with first roll date
        days_tolerance = 0.1
        days = abs((dates[-1] - start_date).days) / utils.tenor_to_days(self.index.tenor)
        if days < days_tolerance:
            dates[-1] = start_date
        elif days > 1 - days_tolerance:
            dates.append(start_date)
        else:
            raise Exception(f"Not support non-standard swap schedule")
        return dates[::-1]

    def create_leg(self, start_date: datetime, end_date: datetime, pay_receive: PayReceive, notional: float,
                   notional_ccy: Ccy, fx_index: FXIndex, spread: float):
        accrual_schedule = self.create_schedule(start_date, end_date)
        rate_accrual_periods = []
        for idx, dt in enumerate(accrual_schedule[:-1]):
            start_date = dt
            end_date = accrual_schedule[idx + 1]
            year_fraction = (end_date - start_date).days / self.dc.value
            ibor_rate_data = IborRateData(self.index, start_date)
            period = RateAccrualPeriod(start_date, end_date, year_fraction, spread, ibor_rate_data)
            rate_accrual_periods.append(period)

        interest_payments = []
        # Assume interest payment schedule is the same as accrual schedule
        notional_factor = -1 if pay_receive == PayReceive.PAY else 1
        for idx, period in enumerate(rate_accrual_periods):
            fx_fixing_date = fx_index.get_fixing_date_from_effective(period.start_date)
            if idx == 0:
                fx_reset_data = FXResetData(fx_index, fx_fixing_date, period.start_date) if notional_ccy != self.ccy else None

            interest_payments.append(RatePaymentPeriod([period], period.end_date, self.ccy, notional_factor * notional,
                                                       fx_reset_data))

        # Assume notional payments on both initial date and final date
        notional_payments = [NotionalExchange(ccy=self.ccy, notional=-interest_payments[0].notional,
                                              payment_date=interest_payments[0].rate_accrual_period[0].start_date,
                                              fx_reset_data=interest_payments[0].fx_reset_data),
                             NotionalExchange(ccy=self.ccy, notional=interest_payments[0].notional,
                                              payment_date=interest_payments[-1].rate_accrual_period[-1].end_date,
                                              fx_reset_data=interest_payments[0].fx_reset_data)
                             ]
        rate_calculation_leg = RateCalculationLeg(pay_receive, interest_payments, notional_payments)
        ibor_leg = IborLeg(rate_calculation_leg, notional, self.proj_curve_name, self.disc_curve_name)
        return ibor_leg


class XCcySwapConvention:
    def __init__(self, name: str, spot_offset: DaysAdjustment, spread_leg: IborLegConvention,
                 flat_leg: IborLegConvention, fx_index: FXIndex, notional_ccy: Ccy):
        self.name = name
        self.spot_offset = spot_offset
        self.spread_leg = spread_leg
        self.flat_leg = flat_leg
        self.fx_index = fx_index
        self.notional_ccy = notional_ccy

        if not self.fx_index.ccy_pair.contains(self.notional_ccy):
            raise Exception(f"Notion ccy {self.notional_ccy.value} must be in swap pair {self.fx_index.ccy_pair.to_string()}")

    def create_swap(self, trade_date: datetime, tenor: str, spread: float, abs_notional: float,
                    spread_leg_pay_receive: PayReceive, swap_start_tenor: str = "0D"):
        is_fwd_starting = swap_start_tenor.upper() != "0D"
        swap_start_date = utils.add_tenor(trade_date, swap_start_tenor)
        start_date = self.spot_offset.apply(swap_start_date) if is_fwd_starting else self.spot_offset.apply(trade_date)
        unadjusted_end_date = utils.add_tenor(start_date, tenor)
        end_date = self.flat_leg.payment_offset.apply(unadjusted_end_date)
        spread_leg = self.spread_leg.create_leg(start_date, end_date, spread_leg_pay_receive, abs(abs_notional),
                                                self.notional_ccy, self.fx_index, spread)
        flat_leg = self.flat_leg.create_leg(start_date, end_date, spread_leg_pay_receive.inverse(), abs(abs_notional),
                                            self.notional_ccy, self.fx_index, spread=0)
        return XCcySwap(spread_leg, flat_leg, start_date) if is_fwd_starting else XCcySwap(spread_leg, flat_leg)




