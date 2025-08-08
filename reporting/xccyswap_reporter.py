from ..interface.imarket import IMarket
from ..interface.itradereporter import ITradeReporter
from ..tradable.xccyswap import XCcySwap, IborLeg
from ..valuation.xccyswap_valuer import XCcySwapValuer
from ..analytics.utils import float_equal


class XCcySwapReporter(ITradeReporter):
    def __init__(self, xccyswap: XCcySwap):
        self.tradable = xccyswap

    @staticmethod
    def get_leg_cash_flows(leg: IborLeg, market: IMarket):
        base_dt = market.get_base_datetime()
        rate_calculation_leg = leg.rate_calculation_leg
        interest_payment_cash_flows = 0.
        convert_to_usd = False  # keep original currency without fx conversion
        for period in rate_calculation_leg.interest_payments:
            if period.payment_date == base_dt:
                interest_payment_cash_flows += XCcySwapValuer.price_rate_payment_period(
                    period, market, leg.proj_curve_name, leg.disc_curve_name, convert_to_usd=convert_to_usd)

        notional_payment_cash_flows = 0.
        for period in rate_calculation_leg.notional_payments:
            if period.payment_date == base_dt:
                notional_payment_cash_flows += XCcySwapValuer.price_notional_payment_period(
                    period, market, leg.disc_curve_name, convert_to_usd=convert_to_usd)

        cash_flows_local = interest_payment_cash_flows + notional_payment_cash_flows
        res = {} if float_equal(cash_flows_local, 0) else {leg.get_ccy(): cash_flows_local}
        return res

    def get_cash_flows(self, market: IMarket):
        cash_flows = self.get_leg_cash_flows(self.tradable.spread_leg, market)
        flat_leg_cash_flows = self.get_leg_cash_flows(self.tradable.flat_leg, market)
        for ccy, amount in flat_leg_cash_flows.items():
            cash_flows[ccy] = cash_flows.get(ccy, 0) + amount
        return cash_flows
