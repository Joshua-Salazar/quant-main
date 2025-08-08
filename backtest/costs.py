from ..tradable.cash import Cash
from ..tradable.constant import Constant
from ..tradable.position import Position, ValuedPosition
from ..tradable.portfolio import Portfolio
from ..tradable.option import Option
from ..tradable.varianceswap import VarianceSwap
from ..tradable.FXforward import FXforward
from datetime import datetime


class TradingCost:
    def __init__(self):
        pass

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, path_to_put_costs=()):
        pass


class PriceBasedTradingCostByTradableType(TradingCost):
    def __init__(self, cost_info, absolute=False):
        self.cost_info = cost_info
        self.absolute = absolute

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp, reprice, market, valuer_map):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()
        trading_costs = []
        trading_costs_by_tradable = {}
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)

                cost_rate_for_tradable = self.cost_info[type(tradable)]

                if self.absolute:
                    tc = units_traded * cost_rate_for_tradable
                else:
                    if reprice:
                        price = tradable.price(market, valuer=valuer_map.get(type(tradable), None), calc_types='price')
                    else:
                        price = getattr(net_positions.get(k, net_positions_pre_trading.get(k, None)), 'price')
                    tc = units_traded * abs(price) * cost_rate_for_tradable
                trading_costs.append(ValuedPosition(Cash(currency), -tc, price=1))
                trading_costs_by_tradable[tradable] = tc
        return trading_costs, trading_costs_by_tradable

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(), reprice=False, market=None, valuer_map={},
              **kwargs):
        trading_costs, trading_costs_by_tradable_by_risk = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp, reprice, market, valuer_map)
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)
        return trading_costs, trading_costs_by_tradable_by_risk


class RiskBasedTradingCostByTradableType(TradingCost):
    def __init__(self, cost_info):
        self.cost_info = cost_info

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp, reprice, market, valuer_map):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()
        trading_costs = []
        trading_costs_by_tradable_by_risk = {}
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)

                cost_info_for_tradable = self.cost_info[type(tradable)]
                for risk_type, cost_rate in cost_info_for_tradable.items():
                    if reprice:
                        risk_value = tradable.price(market, valuer=valuer_map.get(type(tradable), None), calc_types=risk_type)
                    else:
                        risk_value = getattr(net_positions.get(k, net_positions_pre_trading.get(k, None)), risk_type)
                    tc = units_traded * abs(risk_value) * cost_rate
                    trading_costs.append(ValuedPosition(Cash(currency), -tc, price=1))
                    trading_costs_by_tradable_by_risk.setdefault(tradable, {})[risk_type] = tc
        return trading_costs, trading_costs_by_tradable_by_risk

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(), reprice=False, market=None, valuer_map={},
              **kwargs):
        trading_costs, trading_costs_by_tradable_by_risk = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp, reprice, market, valuer_map)
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)
        return trading_costs, trading_costs_by_tradable_by_risk


class FlatTradingCost(TradingCost):
    def __init__(self, delta_cost, vega_cost = None, per_unit=False, free_unwind=False):
        self.delta_cost = delta_cost
        if vega_cost is None:
            self.vega_cost = delta_cost
        else:
            self.vega_cost = vega_cost
        self.per_unit = per_unit
        self.free_unwind = free_unwind

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp, reprice, market, valuer_map):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()
        trading_costs = []
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                #fx = net_positions.get(k, net_positions_pre_trading.get(k, None)).fx
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)

                # get delta/vega cost rate. assume anything that's not an option is charged at delta_cost
                if isinstance( tradable, Option ):
                    cost_rate = self.vega_cost
                else:
                    # flat cost could be different per type of tradable, in this case the input is a dict indexed by tradable type
                    if isinstance(self.delta_cost, dict):
                        cost_rate = self.delta_cost[type(tradable)]
                    else:
                        cost_rate = self.delta_cost
                # do not charge on expiry
                if (time_stamp is not None and hasattr( tradable, 'expiration' ) and tradable.expiration.date() == time_stamp.date() ) \
                        or (self.free_unwind and units == 0):
                    tc = 0
                else:
                    if units_traded > 0:
                        if reprice:
                            price = tradable.price(market, valuer=valuer_map.get(type(tradable), None))
                        else:
                            price = net_positions.get(k, net_positions_pre_trading.get(k, None)).price
                        tc = cost_rate * units_traded * (1 if self.per_unit else price) #* fx
                    else:
                        tc = 0
                trading_costs.append( ValuedPosition(Cash(currency), -tc, price = 1))
        return trading_costs

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(), reprice=False, market=None, valuer_map={},
              **kwargs):
        trading_costs = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp, reprice, market, valuer_map)
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)



class FlatVegaTradingCost(TradingCost):

    # currently handles varswaps hedged with same-expiry synthetic fwds
    # TODO: generalise

    def __init__(self, delta_cost, vega_cost = None ):
        self.delta_cost = delta_cost
        if vega_cost is None:
            self.vega_cost = delta_cost
        else:
            self.vega_cost = vega_cost

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp ):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()

        varswap_pos_pre_trading = [ k for k, v in net_positions_pre_trading.items() if isinstance( v.tradable, VarianceSwap ) ]
        varswap_pos = [ k for k,v in net_positions.items() if isinstance( v.tradable, VarianceSwap ) ]

        opt_pos_pre_trading = [ k for k, v in net_positions_pre_trading.items() if isinstance( v.tradable, Option ) ]
        opt_pos = [ k for k,v in net_positions.items() if isinstance( v.tradable, Option ) ]

        if len( varswap_pos ) > 0 or len( varswap_pos_pre_trading ) > 0:
            opt_pos = [ k for k in opt_pos if net_positions[ k ].tradable.is_call ]
            opt_pos_pre_trading = [ k for k in opt_pos_pre_trading if net_positions_pre_trading[ k ].tradable.is_call ]
            k_to_charge_on = set( varswap_pos_pre_trading ).union( set( varswap_pos ) ).union( set( opt_pos_pre_trading ) ).union( set( opt_pos ) )
        else:
            k_to_charge_on = set(net_positions.keys()).union(set(net_positions_pre_trading.keys()))

        trading_costs = []
        for k in k_to_charge_on:
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):

                #fx = net_positions.get(k, net_positions_pre_trading.get(k, None)).fx
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)

                # get delta/vega cost rate. assume anything that's not an option is charged at delta_cost
                if isinstance( tradable, Option ) or isinstance( tradable, VarianceSwap ):
                    cost_rate = self.vega_cost
                    if k in varswap_pos or k in varswap_pos_pre_trading:
                        charge_qty = net_positions.get(k, net_positions_pre_trading.get(k, None)).vega
                        charge = abs( cost_rate * charge_qty )
                    else:
                        cost_rate = self.delta_cost
                        charge_c = net_positions.get(k, net_positions_pre_trading.get(k, None)).price
                        if k in net_positions:
                            same_exp_puts = { ky:v for ky,v in net_positions.items() if
                                              isinstance(v.tradable, Option) and
                                              v.tradable.expiration == net_positions[k].tradable.expiration and
                                              not v.tradable.is_call }
                        elif k in net_positions_pre_trading:
                            same_exp_puts = { ky:v for ky,v in net_positions_pre_trading.items() if
                                              isinstance(v.tradable, Option) and
                                              v.tradable.expiration == net_positions_pre_trading[k].tradable.expiration and
                                              not v.tradable.is_call }
                        assert len( same_exp_puts ) == 1
                        k = list( same_exp_puts.keys() )[ 0 ]
                        charge_p = net_positions.get(k, net_positions_pre_trading.get(k, None)).price
                        charge = abs( charge_c - charge_p ) * cost_rate
                else:
                    # flat cost could be different per type of tradable, in this case the input is a dict indexed by tradable type
                    if isinstance(self.delta_cost, dict):
                        cost_rate = self.delta_cost[type(tradable)]
                    else:
                        cost_rate = self.delta_cost
                    charge_qty = net_positions.get(k, net_positions_pre_trading.get(k, None)).price
                # do not charge on expiry
                if time_stamp is not None and hasattr( tradable, 'expiration' ) and tradable.expiration.date() == time_stamp.date():
                    tc = 0
                else:
                    tc = charge * units_traded #* fx
                trading_costs.append( ValuedPosition(Cash(currency), -tc, price = 1))
        return trading_costs

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(),
              **kwargs):
        trading_costs = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp )
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)


class FlatVegaCostStock(TradingCost):
    def __init__(self, delta_cost, vega_cost = None , delta_per_unit=False):
        self.delta_cost = delta_cost
        if vega_cost is None:
            self.vega_cost = delta_cost
        else:
            self.vega_cost = vega_cost
        self.delta_per_unit = delta_per_unit

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()
        trading_costs = []
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                #fx = net_positions.get(k, net_positions_pre_trading.get(k, None)).fx
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)

                # get delta/vega cost rate. assume anything that's not an option is charged at delta_cost
                if isinstance( tradable, Option ):
                    cost_rate = self.vega_cost
                    charge = net_positions.get(k, net_positions_pre_trading.get(k, None)).vega
                else:
                    # flat cost could be different per type of tradable, in this case the input is a dict indexed by tradable type
                    if isinstance(self.delta_cost, dict):
                        cost_rate = self.delta_cost[type(tradable)]
                    else:
                        cost_rate = self.delta_cost
                    charge = 1 if self.delta_per_unit else net_positions.get(k, net_positions_pre_trading.get(k, None)).price
                # do not charge on expiry
                if time_stamp is not None and hasattr( tradable, 'expiration' ) and tradable.expiration.date() == time_stamp.date():
                    tc = 0
                else:
                    assert charge >= 0
                    tc = cost_rate * charge * units_traded
                trading_costs.append( ValuedPosition(Cash(currency), -tc, price = 1))
        return trading_costs

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(),
              **kwargs):
        trading_costs = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp )
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)


class BidOfferCost(TradingCost):
    def __init__(self, delta_cost, vega_cost = None, use_otm=False, free_unwind=False):
        self.delta_cost = delta_cost
        if vega_cost is None:
            self.vega_cost = delta_cost
        else:
            self.vega_cost = vega_cost
        self.use_otm = use_otm
        self.free_unwind = free_unwind

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp,
                      market, valuer_map):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()
        trading_costs = []
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                #fx = net_positions.get(k, net_positions_pre_trading.get(k, None)).fx
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)
                if units_traded == 0:
                    continue
                # get delta/vega cost rate. assume anything that's not an option is charged at delta_cost
                if isinstance( tradable, Option ):
                    cost_rate = self.vega_cost
                    if self.use_otm and tradable.intrinsic_value(market, valuer_map[type(tradable.underlying)]) > 0:
                        otm_option = Option(tradable.root, tradable.underlying, tradable.currency, tradable.expiration,
                                            tradable.strike, not tradable.is_call, tradable.is_american,
                                            tradable.contract_size, tradable.tz_name, None)
                        otm_option.listed_ticker = tradable.listed_ticker.replace('Call', 'Put') if tradable.is_call \
                            else tradable.listed_ticker.replace('Put', 'Call')
                        bid, ask = valuer_map[type(tradable)].price(otm_option, market, calc_types=['bid', 'ask'])
                    else:
                        bid = net_positions.get(k, net_positions_pre_trading.get(k, None)).bid
                        ask = net_positions.get(k, net_positions_pre_trading.get(k, None)).ask
                    charge = (ask - bid)

                else:
                    # flat cost could be different per type of tradable, in this case the input is a dict indexed by tradable type
                    if isinstance(self.delta_cost, dict):
                        cost_rate = self.delta_cost[type(tradable)]
                    else:
                        cost_rate = self.delta_cost
                    charge = net_positions.get(k, net_positions_pre_trading.get(k, None)).price

                # do not charge on expiry
                if (time_stamp is not None and hasattr( tradable, 'expiration' ) and tradable.expiration.date() == time_stamp.date())\
                        or (self.free_unwind and units==0) or (isinstance( tradable, Option ) and ask==0):
                    tc = 0
                else:
                    assert charge >= 0
                    tc = cost_rate * charge * units_traded
                trading_costs.append( ValuedPosition(Cash(currency), -tc, price = 1))
        return trading_costs

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp = None, path_to_put_costs=(),
              market=None, valuer_map=None, **kwargs):
        trading_costs = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp, market=market,
                                           valuer_map=valuer_map)
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)



class VariableVegaCost(TradingCost):
    def __init__(self, cost_cap, cost_floor, cost_scale, delta_cost, backtest_market, valuer_map={}):
        self.cost_cap = cost_cap
        self.cost_floor = cost_floor
        self.cost_scale = cost_scale
        self.delta_cost = delta_cost
        self.backtest_market = backtest_market
        self.valuer_map = valuer_map

    def get_risk(self, time_stamp, tradable ):
        if tradable.underlying in [ 'AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'USD' ]:
            market = self.backtest_market.get_market(time_stamp)

            if isinstance(tradable, Option):
                return tradable.price(market=market, valuer=self.valuer_map[Option], return_struc=True,
                                      calc_types=['price', 'vega', 'vol'])
            elif isinstance(tradable, FXforward):
                fwd = tradable.price(market=market, valuer=self.valuer_map[FXforward], return_struc=True,
                                     calc_types='price')
                return { 'price' : fwd }
        else:
            raise RuntimeError( 'Unsupported tradable ' + tradable.name() )

    def trading_costs(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp: datetime):
        net_positions = portfolio.net_positions()
        net_positions_pre_trading = pre_trading_portfolio.net_positions()

        trading_costs = []
        for k in set(net_positions.keys()).union(set(net_positions_pre_trading.keys())):
            tradable = net_positions.get(k, net_positions_pre_trading.get(k, None)).tradable
            if not (isinstance(tradable, Cash) or isinstance(tradable, Constant)):
                tradable_pos = net_positions.get( k, net_positions_pre_trading.get(k, None) )
                currency = tradable.currency
                units = net_positions[k].quantity if k in net_positions else 0.0
                units_pre_trading = net_positions_pre_trading[k].quantity if k in net_positions_pre_trading else 0.0
                units_traded = abs(units - units_pre_trading)
                risks = self.get_risk( time_stamp, tradable_pos.tradable )
                if isinstance( tradable, Option ):
                    cost = max( self.cost_floor, min( self.cost_cap, 100 * risks[ 'vol' ] * risks[ 'vega' ] ) )
                    cost *= self.cost_scale
                else:
                    cost = self.delta_cost * risks[ 'price' ]
                # do not charge on expiry
                if tradable.expiration.date() == time_stamp.date():
                    tc = 0
                else:
                    tc = units_traded * cost
                trading_costs.append(Position(Cash(currency), -tc))
        return trading_costs

    def apply(self, portfolio: Portfolio, pre_trading_portfolio: Portfolio, time_stamp: datetime, path_to_put_costs=(),
              **kwargs):
        trading_costs = self.trading_costs(portfolio, pre_trading_portfolio, time_stamp )
        for item in trading_costs:
            portfolio.add_position(item.tradable, item.quantity, path_to_put_costs)
