from datetime import datetime, timedelta
from ..dates import utils as date_utils
from ..infrastructure import market_utils
from ..infrastructure.market import Market
from ..interface.ivaluer import IValuer
from ..tradable.varianceswap import VarianceSwap
import numpy as np
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
import pytz


class VarianceSwapVolaValuer(IValuer):
    def __init__(self):
        pass

    @staticmethod
    def get_vola_contract(varswap: VarianceSwap):

        effective_time = date_utils.datetime_to_vola_datetime(varswap.inception)
        expiration_time = date_utils.datetime_to_vola_datetime(varswap.expiration)
        volatility_strike = varswap.strike_in_vol
        holidays = date_utils.get_holidays(varswap.cdr_code, varswap.inception, varswap.expiration)
        t_plus_1 = varswap.inception + timedelta(days=varswap.lag)
        num_expected_returns = date_utils.count_business_days(t_plus_1, varswap.expiration, holidays)
        volatility_cap_factor = varswap.get_cap()
        notional = varswap.notional
        contract_data = vola.makeVolDerivativeContract(
            volDerivativeType=vola.VARIANCE_SWAP,
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

    def create_fair_var_swap(self, var_swap: VarianceSwap, market: Market):
        """
        work out fair strike for aged/un-aged var swap based on volar var swap payoff:
        v = df * E[N * (sigma^2 - sigma_k^2)]
        where sigma^2 = 1e4 * var / tau
        var = var_past + var_intrday + var_future
        """
        [tau, var_past, var_intraday, var_future] = \
            self.price(var_swap, market, calc_types=["tau", "varTotalPast", "varTotalIntraday", "varTotalFuture"])
        var = var_past + var_intraday + var_future
        strike_in_vol = np.sqrt(var * 1e4 / tau)
        unit_var_swap = VarianceSwap(
            underlying=var_swap.underlying, inception=var_swap.inception, expiration=var_swap.expiration,
            strike_in_vol=strike_in_vol, notional=1., currency=var_swap.currency, lag=var_swap.lag,
            cap=var_swap.cap, cdr_code=var_swap.cdr_code, asset_class=var_swap.asset_class)
        return unit_var_swap

    @staticmethod
    def create_unit_var_swap(var_swap: VarianceSwap, market: Market, override_inception_date: bool, force_uncapped=False):
        """
        create new fair strike var swap start from today for given vol swap, with notional = 1
        """
        base_date = market.get_base_datetime()
        inception = datetime(base_date.year, base_date.month, base_date.day, 16, 0, 0,
                             tzinfo=pytz.timezone('America/New_York')) if override_inception_date else var_swap.inception
        cap = 0 if force_uncapped else var_swap.cap
        unit_var_swap = VarianceSwap(
            underlying=var_swap.underlying, inception=inception, expiration=var_swap.expiration,
            strike_in_vol=0, notional=1., currency=var_swap.currency, lag=var_swap.lag,
            cap=cap, cdr_code=var_swap.cdr_code, asset_class=var_swap.asset_class)

        contract = VarianceSwapVolaValuer.get_vola_contract(unit_var_swap)
        vola_surface = market.get_vol_surface(var_swap.underlying).vola_surface

        algo_data = vola.makeAlgoDataVolDerivative(normStrikeMin=-100.0, normStrikeMax=100.0)
        fair_strike = vola.VolDerivativeUtils.calcFairStrike(contract, vola_surface, vol0Var=-1, s2Var=0, c2Var=0,
                                                             algoData=algo_data)
        unit_var_swap = VarianceSwap(
            underlying=var_swap.underlying, inception=inception, expiration=var_swap.expiration,
            strike_in_vol=fair_strike, notional=1., currency=var_swap.currency, lag=var_swap.lag,
            cap=cap, cdr_code=var_swap.cdr_code, asset_class=var_swap.asset_class)

        return unit_var_swap

    def price(self, var_swap: VarianceSwap, market: Market, calc_types='price', **kwargs):

        if var_swap.lag != 1:
            raise Exception(f"Unsupported lagged ({var_swap.lag}) var swap")

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        vol_surface = market.get_vol_surface(var_swap.underlying)
        price_datastore = kwargs.get("price_datastore", None)
        base_date = market.get_base_datetime()
        calc_types_list_to_calc = []
        calc_key_map = {}
        if price_datastore is None:
            calc_types_list_to_calc = calc_types_list
        else:
            fixing_table_key = market_utils.create_fixing_table_key()
            if market.has_item(fixing_table_key):
                fixing_table = market.get_item(fixing_table_key)
            else:
                fixing_table = None
            for calc_type in calc_types_list:
                key = (base_date, var_swap, vol_surface, fixing_table, calc_type)
                calc_key_map[calc_type] = key
                if key not in price_datastore:
                    calc_types_list_to_calc.append(calc_type)

        if len(calc_types_list_to_calc) > 0:
            contract_data = self.get_vola_contract(var_swap)

            holidays = date_utils.get_holidays(var_swap.cdr_code, var_swap.inception, var_swap.expiration)
            fixing_times = []
            fixing_values = []
            dt = var_swap.inception
            while dt <= vol_surface.get_base_datetime():
                if market.has_fixing(var_swap.underlying, dt.date()):
                    fixing = market.get_fixing_from_fixing_table(var_swap.underlying, dt.date())
                    fixing_times.append(date_utils.datetime_to_vola_datetime(dt))
                    fixing_values.append(fixing)
                else:
                    # add spot if we ask for intraday fixing, i.e. inception datetime was 1 pm but current market on 3 pm
                    if dt.date() == market.get_base_datetime().date():
                        fixing_times.append(date_utils.datetime_to_vola_datetime(dt))
                        fixing_values.append(market.get_spot(var_swap.underlying))
                    else:
                        raise Exception(f"Not found {dt.strftime('%Y-%m-%d')} {var_swap.underlying} fixings on "
                                        f"{market.get_base_datetime().date().strftime('%Y-%m-%d')}")

                dt = date_utils.add_business_days(dt, 1, holidays)

            realized_obj = vola.makeVarianceDataPast(fixing_times, fixing_values, [])
            algo_data = vola.makeAlgoDataVolDerivative(normStrikeMin=-100.0, normStrikeMax=100.0)

            factory = vola.makeFactoryAnalytics()
            pricer = factory.makeVolDerivativePricer(contract_data, algo_data)
            try:
                pricer.price(vol_surface.vola_surface, realized_obj)
            except:
                raise

        values = tuple()
        for calc_type in calc_types_list:
            if calc_type in calc_types_list_to_calc:
                if calc_type == "price":
                    values = values + (pricer.value(),)
                elif calc_type == "delta":
                    values = values + (pricer.delta(),)
                elif calc_type == "deltausd":
                    spot = market.get_spot(var_swap.underlying)
                    values = values + (pricer.delta() * spot,)
                elif calc_type == "gamma":
                    values = values + (pricer.gamma(),)
                elif calc_type == "gammausd":
                    values = values + (pricer.gamma() * spot * spot / 100,)
                elif calc_type == "vega":
                    values = values + (pricer.vega()/100,)
                elif calc_type == "vega_fair":
                    values = values + (pricer.vegaFair(),)
                elif calc_type == "vega_fair_own":
                    unit_var_swap = self.create_unit_var_swap(var_swap, market, override_inception_date=True, force_uncapped=True)
                    unit_contract_data = self.get_vola_contract(unit_var_swap)
                    unit_pricer = factory.makeVolDerivativePricer(unit_contract_data, algo_data)
                    unit_realized_obj = vola.makeVarianceDataPast([], [], [])
                    unit_pricer.price(vol_surface.vola_surface, unit_realized_obj)
                    d_vol_strike = unit_pricer.vega() / (2 * unit_var_swap.strike_in_vol)
                    vega = pricer.vega()
                    vega_fair = vega / d_vol_strike
                    values = values + (vega_fair, )
                elif calc_type == "theta":
                    values = values + (pricer.theta(),)
                elif calc_type == "vanna":
                    values = values + (pricer.vanna(),)
                elif calc_type == "volga":
                    values = values + (pricer.volga()/1e4,)
                elif calc_type == "rho":
                    values = values + (pricer.rho(),)
                elif calc_type == "tau":
                    values = values + (pricer.tau(),)
                elif calc_type == "varTotalPast":
                    values = values + (pricer.varTotalPast(),)
                elif calc_type == "varTotalIntraday":
                    values = values + (pricer.varTotalIntraday(),)
                elif calc_type == "varTotalFuture":
                    values = values + (pricer.varTotalFuture(),)
                else:
                    raise Exception(f"Unsupported greek: {calc_type}")
                if price_datastore is not None:
                    price_datastore[calc_key_map[calc_type]] = values[-1]
            else:
                values = values + (price_datastore[calc_key_map[calc_type]],)

        if isinstance(calc_types, list):
            return values
        else:
            return values[0]

    def ask_keys(self, var_swap: VarianceSwap, market: Market=None, **kwargs):
        keys = []
        # add vol surface
        und_list = var_swap.get_underlyings()
        for und in und_list:
            keys.append(market_utils.create_vol_surface_key(und))
        # leave holiday and fixing requirement outside
        return keys