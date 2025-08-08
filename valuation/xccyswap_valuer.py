from datetime import datetime
from ..constants.ccy import Ccy
from ..infrastructure.fx_pair import FXPair
from ..infrastructure.market import Market
from ..tradable.xccyswap import XCcySwap, IborLeg, RatePaymentPeriod, NotionalExchange
from ..valuation import valuer_utils
from ..interface.ivaluer import IValuer
from ..analytics.utils import float_equal


class XCcySwapValuer(IValuer):
    def __init__(self):
        pass

    @staticmethod
    def price_rate_payment_period(
            rate_payment_period: RatePaymentPeriod, market: Market, proj_curve_name: str, disc_curve_name: str,
            spread_override: float = None, convert_to_usd: bool=True, swap_start_date: datetime=None):
        base_dt = market.get_base_datetime()
        if rate_payment_period.payment_date < base_dt:
            return 0.
        amount = 0.
        for period in rate_payment_period.rate_accrual_period:
            ibor_rate_data = period.ibor_rate_data
            if ibor_rate_data.fixing_date < base_dt:
                raw_rate = market.get_rate_fixing(ibor_rate_data.index.ccy, ibor_rate_data.index.tenor,
                                                  ibor_rate_data.fixing_date)
            else:
                if swap_start_date is None:
                    rate_curve = market.get_spot_rate_curve(ibor_rate_data.index.ccy, proj_curve_name)
                    raw_rate = rate_curve.get_forward_rate_with_dt(
                        ibor_rate_data.reset_start, ibor_rate_data.reset_end, ibor_rate_data.reset_year_fraction)
                else:
                    raw_rate = market.get_forward_rate(ibor_rate_data.index.ccy, proj_curve_name,
                                                       ibor_rate_data.fixing_date)
            rate = raw_rate + period.spread if spread_override is None else raw_rate + spread_override
            amount += rate * period.year_fraction
        notional = rate_payment_period.get_notional(market)
        amount *= notional
        ccy = rate_payment_period.rate_accrual_period[0].ibor_rate_data.index.ccy
        if convert_to_usd:
            disc_ccy = Ccy.USD
            pair = FXPair(ccy, disc_ccy)
            fx_fwd = market.get_fx_fwd(pair, rate_payment_period.payment_date)
            if swap_start_date is not None:
                fx_fwd_st = market.get_fx_fwd(pair, swap_start_date)
                fx_fwd /= fx_fwd_st
            df = market.get_df(disc_ccy, disc_curve_name, rate_payment_period.payment_date)
            res = amount * fx_fwd * df
        else:
            disc_ccy = ccy
            df = market.get_df(disc_ccy, disc_curve_name, rate_payment_period.payment_date)
            res = amount * df
        if swap_start_date is not None:
            df_st = market.get_df(disc_ccy, disc_curve_name, swap_start_date)
            res /= df_st
        return res

    @staticmethod
    def pvbp_rate_payment_period(rate_payment_period: RatePaymentPeriod, market: Market, disc_curve_name: str,
                                 convert_to_usd: bool = True, swap_start_date: datetime=None):
        base_dt = market.get_base_datetime()
        if rate_payment_period.payment_date < base_dt:
            return 0.
        amount = 0.
        for period in rate_payment_period.rate_accrual_period:
            amount += period.year_fraction
        amount *= rate_payment_period.get_notional(market)
        ccy = rate_payment_period.rate_accrual_period[0].ibor_rate_data.index.ccy
        if convert_to_usd:
            disc_ccy = Ccy.USD
            pair = FXPair(ccy, disc_ccy)
            fx_fwd = market.get_fx_fwd(pair, rate_payment_period.payment_date)
            if swap_start_date is not None:
                fx_fwd_st = market.get_fx_fwd(pair, swap_start_date)
                fx_fwd /= fx_fwd_st
            df = market.get_df(disc_ccy, disc_curve_name, rate_payment_period.payment_date)
            res = amount * fx_fwd * df
        else:
            disc_ccy = ccy
            df = market.get_df(disc_ccy, disc_curve_name, rate_payment_period.payment_date)
            res = amount * df
        if swap_start_date is not None:
            df_st = market.get_df(disc_ccy, disc_curve_name, swap_start_date)
            res /= df_st
        return res

    @staticmethod
    def price_notional_payment_period(notional_payment: NotionalExchange, market: Market, disc_curve_name: str,
                                      convert_to_usd: bool = True, swap_start_date: datetime=None):
        base_dt = market.get_base_datetime()
        if notional_payment.payment_date < base_dt:
            return 0.

        ccy = notional_payment.ccy
        notional = notional_payment.get_notional(market)
        if convert_to_usd:
            disc_ccy = Ccy.USD
            pair = FXPair(ccy, disc_ccy)
            fx_fwd = market.get_fx_fwd(pair, notional_payment.payment_date)
            if swap_start_date is not None:
                fx_fwd_st = market.get_fx_fwd(pair, swap_start_date)
                fx_fwd /= fx_fwd_st
            df = market.get_df(disc_ccy, disc_curve_name, notional_payment.payment_date)
            res = notional * fx_fwd * df
        else:
            disc_ccy = ccy
            df = market.get_df(disc_ccy, disc_curve_name, notional_payment.payment_date)
            res = notional * df
        if swap_start_date is not None:
            df_st = market.get_df(disc_ccy, disc_curve_name, swap_start_date)
            res /= df_st
        return res

    def price_leg(self, leg: IborLeg, market: Market, spread_override: float = None, swap_start_date: datetime=None):
        rate_calculation_leg = leg.rate_calculation_leg
        interest_payment_pv = 0.
        for period in rate_calculation_leg.interest_payments:
            interest_payment_pv += self.price_rate_payment_period(
                period, market, leg.proj_curve_name, leg.disc_curve_name,  spread_override,
                swap_start_date=swap_start_date)
        notional_payment_pv = 0.
        for period in rate_calculation_leg.notional_payments:
            notional_payment_pv += self.price_notional_payment_period(period, market, leg.disc_curve_name,
                                                                      swap_start_date=swap_start_date)

        pv_local = interest_payment_pv + notional_payment_pv
        return pv_local, interest_payment_pv, notional_payment_pv

    def pvbp_leg(self, leg: IborLeg, market: Market, swap_start_date: datetime=None):
        rate_calculation_leg = leg.rate_calculation_leg
        pvbp = 0.
        for period in rate_calculation_leg.interest_payments:
            pvbp += self.pvbp_rate_payment_period(period, market, leg.disc_curve_name, swap_start_date=swap_start_date)

        res = pvbp
        return res

    def price(self, xccyswap: XCcySwap, market: Market, calc_types='price', **kwargs):
        results = {}
        valid_calc_types = ["price", "pvbp", "par_rate", "diagnostic"]
        if isinstance(calc_types, str):
            calc_types = [calc_types]
        for calc_type in calc_types:
            if calc_type == "diagnostic":
                spread_leg_pv, spread_leg_interest_pv, spread_leg_notional_pv = \
                    self.price_leg(xccyswap.spread_leg, market, swap_start_date=xccyswap.start_date)
                flat_leg_pv, flat_leg_interest_pv, flat_leg_notional_pv = \
                    self.price_leg(xccyswap.flat_leg, market, swap_start_date=xccyswap.start_date)
                pv = spread_leg_pv + flat_leg_pv
                if float_equal(pv, 0.):
                    results[calc_type] = pv
                else:
                    results[calc_type] = pv
            elif calc_type == "price":
                spread_leg_pv, spread_leg_interest_pv, spread_leg_notional_pv = \
                    self.price_leg(xccyswap.spread_leg, market, swap_start_date=xccyswap.start_date)
                flat_leg_pv, flat_leg_interest_pv, flat_leg_notional_pv = \
                    self.price_leg(xccyswap.flat_leg, market, swap_start_date=xccyswap.start_date)
                pv = spread_leg_pv + flat_leg_pv
                if float_equal(pv, 0.):
                    results[calc_type] = pv
                else:
                    results[calc_type] = pv
            elif calc_type == "pvbp":
                pvbp = self.pvbp_leg(xccyswap.spread_leg, market, swap_start_date=xccyswap.start_date)
                results[calc_type] = pvbp
            elif calc_type == "par_rate":
                spread_leg_pv_with_zero_spread, spread_leg_interest_pv, spread_leg_notional_pv = \
                    self.price_leg(xccyswap.spread_leg, market, spread_override=0., swap_start_date=xccyswap.start_date)
                flat_leg_pv, flat_leg_interest_pv, flat_leg_notional_pv = \
                    self.price_leg(xccyswap.flat_leg, market, swap_start_date=xccyswap.start_date)
                pv = spread_leg_pv_with_zero_spread + flat_leg_pv
                pvbp = self.pvbp_leg(xccyswap.spread_leg, market, swap_start_date=xccyswap.start_date)
                par_rate = -pv / pvbp
                results[calc_type] = par_rate * 10000    # show par rate in bpv for xccy basis
            else:
                raise Exception(f"Unsupport calc type {calc_type}, only support: {','.join(valid_calc_types)}")

        return valuer_utils.return_results_based_on_dictionary(calc_types, results)
