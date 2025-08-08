from datetime import datetime, timedelta
from ..dates import utils as date_utils
from ..infrastructure.market import Market
from ..interface.ivaluer import IValuer
from ..tradable.volswap import VolSwap
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
import pytz


class VolSwapVolaValuer(IValuer):
    def __init__(self):
        pass

    @staticmethod
    def get_vola_contract(vol_wap: VolSwap):

        effective_time = date_utils.datetime_to_vola_datetime(vol_wap.inception)
        expiration_time = date_utils.datetime_to_vola_datetime(vol_wap.expiration)
        volatility_strike = vol_wap.strike_in_vol
        holidays = date_utils.get_holidays(vol_wap.cdr_code, vol_wap.inception, vol_wap.expiration)
        t_plus_1 = vol_wap.inception + timedelta(days=vol_wap.lag)
        num_expected_returns = date_utils.count_business_days(t_plus_1, vol_wap.expiration, holidays)
        volatility_cap_factor = vol_wap.get_cap()
        notional = vol_wap.notional
        contract_data = vola.makeVolDerivativeContract(
            volDerivativeType=vola.VOLATILITY_SWAP,
            samplingTimeFirst=effective_time,
            samplingTimeLast=expiration_time,
            expiryTime=expiration_time,
            annualizationFactor=252,
            volatilityStrike=volatility_strike,
            notional=notional,
            numExpectedReturns=num_expected_returns,
            volatilityCapFactor=volatility_cap_factor
        )
        return contract_data

    @staticmethod
    def create_unit_vol_swap(vol_swap: VolSwap, market: Market, force_uncapped=False):
        """
        create new fair strike vol swap start from today for given vol swap, with notional = 1
        """
        vola_surface = market.get_vol_surface(vol_swap.underlying).vola_surface
        base_date = market.get_base_datetime()
        inception = datetime(base_date.year, base_date.month, base_date.day, 16, 0, 0,
                             tzinfo=pytz.timezone('America/New_York'))
        cap = 0 if force_uncapped else vol_swap.cap
        unit_var_swap = VolSwap(
            underlying=vol_swap.underlying, inception=inception, expiration=vol_swap.expiration,
            strike_in_vol=0, notional=1., currency=vol_swap.currency, lag=vol_swap.lag,
            cap=cap, cdr_code=vol_swap.cdr_code, asset_class=vol_swap.asset_class)

        contract = VolSwapVolaValuer.get_vola_contract(unit_var_swap)

        algo_data = vola.makeAlgoDataVolDerivative(normStrikeMin=-100.0, normStrikeMax=100.0)
        fair_strike = vola.VolDerivativeUtils.calcFairStrike(contract, vola_surface, vol0Var=-1, s2Var=0, c2Var=0,
                                                             algoData=algo_data)
        unit_var_swap = VolSwap(
            underlying=vol_swap.underlying, inception=inception, expiration=vol_swap.expiration,
            strike_in_vol=fair_strike, notional=1., currency=vol_swap.currency, lag=vol_swap.lag,
            cap=cap, cdr_code=vol_swap.cdr_code, asset_class=vol_swap.asset_class)

        return unit_var_swap

    def price(self, vol_swap: VolSwap, market: Market, calc_types='price', **kwargs):

        if vol_swap.lag != 1:
            raise Exception(f"Unsupported lagged ({vol_swap.lag}) vol swap")

        contract_data = self.get_vola_contract(vol_swap)

        vol_surface = market.get_vol_surface(vol_swap.underlying)
        holidays = date_utils.get_holidays(vol_swap.cdr_code, vol_swap.inception, vol_swap.expiration)
        fixing_times = []
        fixing_values = []
        dt = vol_swap.inception
        while dt < market.get_base_datetime():
            fixing = market.get_fixing_from_fixing_table(vol_swap.underlying, dt.date())
            fixing_times.append(date_utils.datetime_to_vola_datetime(dt))
            fixing_values.append(fixing)
            dt = date_utils.add_business_days(dt, 1, holidays)

        realized_obj = vola.makeVarianceDataPast(fixing_times, fixing_values, [])
        algo_data = vola.makeAlgoDataVolDerivative(normStrikeMin=-100.0, normStrikeMax=100.0)

        factory = vola.makeFactoryAnalytics()
        pricer = factory.makeVolDerivativePricer(contract_data, algo_data)

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        values = tuple()
        pricer.price(vol_surface.vola_surface, realized_obj)
        for calc_type in calc_types_list:
            if calc_type == "price":
                values = values + (pricer.value(),)
            elif calc_type == "delta":
                values = values + (pricer.delta(),)
            elif calc_type == "gamma":
                values = values + (pricer.gamma(),)
            elif calc_type == "vega":
                values = values + (pricer.vega()/100,)
            elif calc_type == "vega_fair":
                values = values + (pricer.vegaFair(),)
            elif calc_type == "theta":
                values = values + (pricer.theta(),)
            elif calc_type == "vanna":
                values = values + (pricer.vanna(),)
            elif calc_type == "volga":
                values = values + (pricer.volga()/1e4,)
            elif calc_type == "rho":
                values = values + (pricer.rho(),)
            else:
                raise Exception(f"Unsupported greek: {calc_type}")

        if isinstance(calc_types, list):
            return values
        else:
            return values[0]
