from ..dates.days_adjustment import DaysAdjustment
from ..constants.ccy import Ccy
from ..constants.pay_receive import PayReceive
from ..interface.itradable import ITradable
from ..tradable.xccyswap import IborLegConvention, IborLeg, RateCalculationLeg, RateAccrualPeriod
from dataclasses import dataclass
from ..constants.frequency_annual import FrequencyAnnual
from ..constants.day_count_convention import DayCountConvention
from ..dates import utils
from datetime import datetime


@dataclass
class FixedLeg:
    rate_calculation_leg: RateCalculationLeg
    notional: float


@dataclass
class FixedRateAccrualPeriod:
    start_date: datetime
    end_date: datetime
    year_fraction: float
    spread: float
    rate: float


@dataclass
class FixedRatePaymentPeriod:
    rate_accrual_period: [RateAccrualPeriod]
    payment_date: datetime
    currency: Ccy
    notional: float
    rate: float


class FixedLegConvention:
    def __init__(self, accrual_adjustment: DaysAdjustment, payment_offset: DaysAdjustment,
                 rate: float, tenor: str, payment_frequency: FrequencyAnnual, dc: DayCountConvention):
        self.accrual_adjustment = accrual_adjustment
        self.payment_offset = payment_offset
        self.rate = rate
        self.tenor = tenor
        self.payment_frequency = payment_frequency
        self.dc = dc

    def roll_backward(self, date_in):
        date_out = utils.add_tenor(date_in, f"-{self.tenor}")
        if not (self.tenor.endswith('d') or self.tenor.endswith('D')):
            date_out = date_out.replace(day=date_in.day)
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
        dates.append(start_date)
        return dates[::-1]

    def create_leg(self, start_date: datetime, end_date: datetime, pay_receive: PayReceive, notional: float,
                   notional_ccy: Ccy):
        accrual_schedule = self.create_schedule(start_date, end_date)
        rate_accrual_periods = []
        for idx, dt in enumerate(accrual_schedule[:-1]):
            start_date = dt
            end_date = accrual_schedule[idx + 1]
            year_fraction = (end_date - start_date).days / self.dc.value
            period = FixedRateAccrualPeriod(start_date, end_date, year_fraction, 0, self.rate)
            rate_accrual_periods.append(period)

        interest_payments = []
        # Assume interest payment schedule is the same as accrual schedule
        notional_factor = -1 if pay_receive == PayReceive.PAY else 1
        for idx, period in enumerate(rate_accrual_periods):
            interest_payments.append(FixedRatePaymentPeriod(
                [period], period.end_date, notional_ccy, notional_factor * notional, self.rate
            ))

        # Assume notional payments on both initial date and final date
        # notional_payments = [NotionalExchange(ccy=notional_ccy, amount=notional,
        #                                       payment_date=interest_payments[0].rate_accrual_period[0].start_date,
        #                                       fx_reset_data=interest_payments[0].fx_reset_data),
        #                      NotionalExchange(ccy=notional_ccy, amount=notional,
        #                                       payment_date=interest_payments[-1].rate_accrual_period[-1].end_date,
        #                                       fx_reset_data=interest_payments[-1].fx_reset_data)
        #                      ]
        notional_payments = []
        rate_calculation_leg = RateCalculationLeg(pay_receive, interest_payments, notional_payments)
        ibor_leg = FixedLeg(rate_calculation_leg, notional)
        return ibor_leg


class VanillaSwap(ITradable):
    def __init__(self, fixed_leg: FixedLeg, ibor_leg: IborLeg):
        self.fixed_leg = fixed_leg
        self.ibor_leg = ibor_leg


class VanillaSwapConvention:
    def __init__(self, name: str, spot_offset: DaysAdjustment, fixed_leg: FixedLegConvention,
                 ibor_leg: IborLegConvention, notional_ccy: Ccy):
        self.name = name
        self.spot_offset = spot_offset
        self.fixed_leg = fixed_leg
        self.ibor_leg = ibor_leg
        self.notional_ccy = notional_ccy

    def create_swap(self, start_date: datetime, tenor: str, spread: float, abs_notional: float,
                    ibor_leg_pay_receive: PayReceive):
        end_date = utils.add_tenor(start_date, tenor)
        ibor_leg = self.ibor_leg.create_leg(start_date, end_date, ibor_leg_pay_receive, abs(abs_notional),
                                            self.notional_ccy, None, spread)
        fixed_leg = self.fixed_leg.create_leg(start_date, end_date, ibor_leg_pay_receive.inverse(), abs(abs_notional),
                                              self.notional_ccy)
        return VanillaSwap(fixed_leg, ibor_leg)
