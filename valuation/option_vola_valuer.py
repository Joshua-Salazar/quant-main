from datetime import datetime
from ..analytics.options import map_to_vola_calc_type
from ..dates.utils import datetime_to_vola_datetime
from ..constants.underlying_type import UnderlyingType
from ..infrastructure.market import Market
from ..tradable.option import Option
from ..interface.ivaluer import IValuer
from ..valuation.utils import find_fx_for_tradable
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola
from scipy.optimize import fsolve


VOLA_FACTOR_MAP = {
    vola.PricerResultsType.VEGA: 1/100,
    vola.PricerResultsType.THETA: 1/256,
}


class OptionVolaValuer(IValuer):

    def __init__(self, cache_valuer=False, underlying_type=None, valuer=None):
        self.cache_valuer = cache_valuer
        self.underlying_type = underlying_type
        self.valuer = valuer

    @classmethod
    def create(cls, root: str, market: Market):

        vol_surface = market.get_vol_surface(root)
        underlying_type = vol_surface.get_underlying_type()

        factory = vola.makeFactoryAnalytics()
        if underlying_type == UnderlyingType.EQUITY:
            valuer = factory.makeOptionPricerEquity(vol_surface.vola_surface.dividendData,
                                                         vol_surface.vola_surface.modelData)

        elif underlying_type == UnderlyingType.FUTURES:
            valuer = factory.makeOptionPricerFutures(vol_surface.vola_surface.modelData)
        else:
            raise Exception("Unexpected underlying type")
        return cls(cache_valuer=True, underlying_type=underlying_type, valuer=valuer)

    def value(self, market, date_dt, expiration_dt, option, spot, vol, vola_calc_types_list, price_datastore):
        def make_key(date, fixed_option, spot, vol, calc_type):
            return f'{date.year}{date.month}{date.day}{fixed_option.name_str}{spot:.8f}{vol:.8f}{calc_type.name}'

        def price_equity_option(pricer, market, date_dt, expiration_dt, option, spot, vol, calc_types):
            vola_date_dt = datetime_to_vola_datetime(date_dt)
            r = market.get_discount_rate(option.root, expiration_dt)
            q = market.get_borrow_rate(option.root, expiration_dt)
            # floor t0 to surface asoftime
            result_values = pricer.price(
                max(vola_date_dt, pricer.modelData.timeConverterV.minT()),
                datetime_to_vola_datetime(expiration_dt),
                option.strike, option.is_call, option.is_american, spot, vol, r, q,
                calc_types)
            return [VOLA_FACTOR_MAP.get(x, 1) * y for x, y in zip(calc_types, result_values)]

        def price_future_option(pricer, market, date_dt, expiration_dt, option, spot, vol, calc_types):
            vola_date_dt = datetime_to_vola_datetime(date_dt)
            r = market.get_discount_rate(option.root, expiration_dt)
            # floor t0 to surface asoftime
            result_values = pricer.price(
                max(vola_date_dt, pricer.modelData.timeConverterV.minT()),
                datetime_to_vola_datetime(expiration_dt),
                option.strike, option.is_call, option.is_american, spot, vol, r,
                calc_types)
            return [VOLA_FACTOR_MAP.get(x, 1) * y for x, y in zip(calc_types, result_values)]

        # if price_datastore is not None:
        #     data = price_datastore.read_item('price_cache')
        data = price_datastore

        vola_calc_types_list_to_calc = []
        keys = []
        if data is None:
            vola_calc_types_list_to_calc = vola_calc_types_list
        else:
            for calc_type in vola_calc_types_list:
                keys.append(make_key(date_dt, option, spot, vol, calc_type))
                if keys[-1] not in data:
                    vola_calc_types_list_to_calc.append(calc_type)

        values_to_calc = []
        if len(vola_calc_types_list_to_calc) > 0:
            underlying_type = market.get_vol_type(option.root)
            if underlying_type == UnderlyingType.EQUITY:
                values_to_calc = price_equity_option(self.valuer, market, date_dt, expiration_dt, option, spot, vol, vola_calc_types_list_to_calc)
            elif underlying_type == UnderlyingType.FUTURES:
                values_to_calc = price_future_option(self.valuer, market, date_dt, expiration_dt, option, spot, vol, vola_calc_types_list_to_calc)
            else:
                raise RuntimeError('Unknown vola surface type')
        idx = 0
        values = []
        for i, calc_type in enumerate(vola_calc_types_list):
            if calc_type in vola_calc_types_list_to_calc:
                value = values_to_calc[idx]
                idx += 1
                if data is not None:
                    data[keys[i]] = value
            else:
                value = data[keys[i]]
            values.append(value)
        return values

    def price(self, option: Option, market: Market, calc_types='price', **kwargs):
        # if we don't cache valuer we create the valuer and underlying type each time we call price
        if not self.cache_valuer:
            vol_surface = market.get_vol_surface(option.root)
            underlying_type = vol_surface.get_underlying_type()

            factory = vola.makeFactoryAnalytics()
            if underlying_type == UnderlyingType.EQUITY:
                valuer = factory.makeOptionPricerEquity(vol_surface.vola_surface.dividendData,
                                                        vol_surface.vola_surface.modelData)

            elif underlying_type == UnderlyingType.FUTURES:
                valuer = factory.makeOptionPricerFutures(vol_surface.vola_surface.modelData)
            else:
                raise Exception("Unexpected underlying type")

            self.underlying_type = underlying_type
            self.valuer = valuer

        date = market.get_base_datetime()
        if isinstance(option.underlying, str):
            spot = market.get_spot(option.underlying)
        else:
            spot = market.get_future_price(option.root, option.underlying.expiration)

        vol_override = kwargs.get('vol_override', None)

        price_datastore = kwargs.get('price_datastore', None)

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        output_vol = kwargs.get('output_vol', None)
        if option.is_expired(market):
            values = []
            for calc_type in calc_types_list:
                if calc_type == "price":
                    value = option.intrinsic_value(market)
                    values.append(value)
                else:
                    values.append(0.)
        else:
            # can only price a concrete option
            assert isinstance(option.expiration, datetime)

            # where to pick the vol
            if vol_override is None:
                vol = market.get_vol(underlying=option.root, expiry_dt=option.expiration, strike=option.strike)
            else:
                vol = vol_override

            vola_calc_types_list = list(map(lambda x: map_to_vola_calc_type(x), calc_types_list))

            values = self.value(market, date, option.expiration, option, spot, vol, vola_calc_types_list, price_datastore)

            if output_vol is not None:
                values.append( vol / option.contract_size )
            # if isinstance(vola_surface, vola.VolSurfaceEquity):
            #     r = vola_surface.discountCurve.rate(date_vola_datetime,
            #                                         fixed_option_expiration_vola_datetime)
            #     q = vola_surface.borrowCurve.rate(date_vola_datetime,
            #                                       fixed_option_expiration_vola_datetime)
            #     values = self.valuer.price(
            #         date_vola_datetime, fixed_option_expiration_vola_datetime,
            #         fixed_option.strike, fixed_option.is_call, fixed_option.is_american, spot, vol, r, q,
            #         vola_calc_types_list)
            # elif isinstance(vola_surface, vola.VolSurfaceFutures):
            #     r = vola_surface.discountCurve.rate(date_vola_datetime,
            #                                         fixed_option_expiration_vola_datetime)
            #     values = self.valuer.price(
            #         date_vola_datetime, fixed_option_expiration_vola_datetime,
            #         fixed_option.strike, fixed_option.is_call, fixed_option.is_american, spot, vol, r,
            #         vola_calc_types_list)
            # else:
            #     raise RuntimeError('Unknown vola surface type')

        currency = kwargs.get('currency', None)
        fx = find_fx_for_tradable(market, option, currency)
        if isinstance(calc_types, list) or output_vol is not None:
            return [x * option.contract_size * fx for x in list(values)]
        else:
            return values[0] * option.contract_size * fx

    def solve(self, option: Option, market: Market, given='delta', solve_for='strike', value_given=0.5):
        # if we don't cache valuer we create the valuer and underlying type each time we call price
        if not self.cache_valuer:
            vol_surface = market.get_vol_surface(option.root)
            underlying_type = vol_surface.get_underlying_type()

            factory = vola.makeFactoryAnalytics()
            if underlying_type == UnderlyingType.EQUITY:
                valuer = factory.makeOptionPricerEquity(vol_surface.vola_surface.dividendData,
                                                        vol_surface.vola_surface.modelData)

            elif underlying_type == UnderlyingType.FUTURES:
                valuer = factory.makeOptionPricerFutures(vol_surface.vola_surface.modelData)
            else:
                raise Exception("Unexpected underlying type")

            self.underlying_type = underlying_type
            self.valuer = valuer

        if option.is_expired(market):
            raise RuntimeError(f'Cannot solve {solve_for} from {given} when option has expired')
        else:
            # can only price a concrete option
            assert isinstance(option.expiration, datetime)

            def value_func(_underlying, _date, _strike, _expiration, _is_call, _is_american, _contract_size, _spot, _vol, _r, _q):
                given_type = map_to_vola_calc_type(given)
                if _vol is None:
                    _vol = market.get_vol(underlying=_underlying, expiry_dt=_expiration, strike=_strike)
                value = self.valuer.price(
                    max(datetime_to_vola_datetime(_date), self.valuer.modelData.timeConverterV.minT()),
                    datetime_to_vola_datetime(_expiration),
                    _strike, _is_call, _is_american, _spot, _vol, _r, _q,
                    [given_type])

                value = value[0] * _contract_size
                return value

            # spot
            if isinstance(option.underlying, str):
                spot = market.get_spot(option.underlying)
            else:
                spot = market.get_future_price(option.root, option.underlying.expiration)

            # r and q
            r = market.get_discount_rate(option.root, option.expiration)
            q = market.get_borrow_rate(option.root, option.expiration)

            if solve_for == 'strike':
                sol = fsolve(lambda x: value_func(option.underlying, market.get_base_datetime(), x[0], option.expiration, option.is_call, option.is_american, option.contract_size, spot, None, r, q) - value_given, spot)
                return sol[0]
            else:
                raise RuntimeError(f'solving for {solve_for} has not been implemented')
