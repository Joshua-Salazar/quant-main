from ..analytics.symbology import option_underlying_type_from_ticker, ticker_from_future_root, OPTION_FUTURE_TICKER_MAP
from ..interface.ivaluer import IValuer
from ..infrastructure.market import Market
from ..tradable.future import Future
from ..valuation.utils import find_fx_for_tradable


class FutureValuer(IValuer):
    def __init__(self, direct_fut_val=False, imply_delta_from_spot=True):
        self.direct_fut_val = direct_fut_val
        self.imply_delta_from_spot = imply_delta_from_spot

    def price(self, future: Future, market: Market, calc_types='price', **kwargs):
        if self.direct_fut_val:
            future_price = market.get_future_price(future, None)
        else:
            try:
                future_price = market.get_future_price(future.root, future.expiration)
            except Exception as e:
                try:
                    future_price = market.get_vol_surface(ticker_from_future_root(future.root)).get_forward(future.expiration)
                except Exception as e:
                    try:
                        future_price = market.get_spot_rates(future.root).get_forward(future.expiration)
                    except Exception as e:
                        if future.expiration.date() == market.get_base_datetime().date():
                            future_price = market.get_spot(future.root)
                        else:
                            raise Exception(str(e))

        if not isinstance(calc_types, list):
            calc_types_list = [calc_types]
        else:
            calc_types_list = calc_types

        # TODO: this is an approximation, need to implement properly
        values = []
        for calc_type in calc_types_list:
            if calc_type == 'price':
                values.append(future_price)
            elif calc_type == 'delta':
                if self.imply_delta_from_spot:
                    ticker = future.root if future.root in OPTION_FUTURE_TICKER_MAP["ticker"].values else ticker_from_future_root(future.root)
                    if option_underlying_type_from_ticker(ticker, market=market) == "equity":
                        spot = market.get_spot(ticker)
                        delta = future_price / spot
                    else:
                        delta = 1.0
                else:
                    delta = 1.0
                values.append(delta)
            elif calc_type == 'gamma':
                values.append(0.)
            elif calc_type == 'vega':
                values.append(0.)
            elif calc_type == 'theta':
                values.append(0.)
            elif calc_type == 'vanna':
                values.append(0.)
            elif calc_type == 'volga':
                values.append(0.)
            elif calc_type == 'rho':
                values.append(0.)
            else:
                raise RuntimeError('Unknown calc type ' + calc_type)

        currency = kwargs.get('currency', None)
        fx = find_fx_for_tradable(market, future, currency)
        if isinstance(calc_types, list):
            return [x * future.contract_size * fx for x in list(values)]
        else:
            return values[0] * future.contract_size * fx
