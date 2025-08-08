from datetime import datetime, date

from ...analytics.swaptions import cube_interpolate_old
from ...backtest.costs import FlatTradingCost
from ...backtest.strategy import Event, StrategyState, DailyStrategy
from ...infrastructure.bmarket import BMarket
from ...tradable.swaption import Swaption
from ...tradable.forwardstartswap import ForwardStartSwap
from ...backtest.rolls import RollSwaptionContracts
from ...backtest.delta_hedges import SwaptionDailyDH
from ...valuation.forwardstartswap_valuer import ForwardStartSwapValuerOld
from ...valuation.swaption_norm_valuer import SwaptionNormValuerOld

from ...tradable.position import Position
from ...tradable.cash import Cash


class RollingSwaptionsState(StrategyState):
    def __init__(self, time_stamp, portfolio, price, cost, legs_info):
        self.price = price
        self.cost = cost
        self.legs_info = legs_info
        super().__init__(time_stamp, portfolio)


class EODTradeEvent(Event):
    def execute(self, state: StrategyState):
        # copy the starting portfolio
        portfolio = state.portfolio.clone()
        non_cash_pfo = {ky: v for ky, v in portfolio.net_positions().items() if not isinstance(v.tradable, Cash)}

        market = self.strategy.backtest_market.get_market(self.time_stamp)

        # trade
        if portfolio.is_empty() or len(non_cash_pfo) < 1:
            # this is the start day, we need to trade into each leg and tranche
            for leg_name, rolls_by_tranche in self.parameters['rolls_by_leg_tranche'].items():
                leg_config = self.parameters['legs'][leg_name]
                for tranche_name, roll in rolls_by_tranche.items():
                    initial_swaption = roll.find_initial_contract(self.time_stamp)
                    initial_swaption_price = initial_swaption.price(market=market, valuer=self.strategy.valuer_map[Swaption],
                                                                    calc_types='price')
                    portfolio.trade(initial_swaption, leg_config['sizing'], initial_swaption_price,
                                    leg_config['currency'], position_path=(leg_name, tranche_name))
        else:
            # roll
            for leg_name, rolls_by_tranche in self.parameters['rolls_by_leg_tranche'].items():
                for tranche_name, roll in rolls_by_tranche.items():
                    roll.roll(self.time_stamp, portfolio.get_position((leg_name, tranche_name)),
                              roll_type='fixed_units',
                              roll_geq = ( 'roll_geq' in self.parameters['legs'][leg_name] ) and
                                         ( self.parameters['legs'][leg_name]['roll_geq'] ) )

        #update delta and delta_hedge
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta'],
                                            valuer_map_override=self.strategy.valuer_map)
        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                if self.parameters['legs'][leg_name]['delta_hedge'] == 'Daily':
                    self.parameters['delta_hedger'].delta_hedge(
                        self.time_stamp, portfolio.get_position((leg_name, tranche_name)))
                elif self.parameters['legs'][leg_name]['delta_hedge'] == 'Unhedged':
                    pass
                else:
                    raise RuntimeError(f'Unknown Delta Hedge Type {self.parameters["legs"][leg_name]["delta_hedge"]}')

        # cost
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta'],
                                            valuer_map_override=self.strategy.valuer_map)
        price_pre_cost = portfolio.aggregate('price')
        self.parameters['trading_cost'].apply(portfolio, state.portfolio)

        # value portfolio
        portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta'], valuer_map_override=self.strategy.valuer_map)
        price = portfolio.aggregate('price')

        legs_info = {}
        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                this_swaption = portfolio.get_position((leg_name, tranche_name)).find_children_of_tradable_type(Swaption)
                assert len(this_swaption) == 1
                this_swaption = this_swaption[0][1]

                vol_cube = {self.time_stamp: market.get_swaption_vol_cube(this_swaption.tradable.currency)}
                fwd_rate_curve = {self.time_stamp: market.get_forward_rates(this_swaption.tradable.currency)}
                strike_vol = cube_interpolate_old(vol_cube, fwd_rate_curve, self.time_stamp, this_swaption.tradable.tenor, this_swaption.tradable.expiration, this_swaption.tradable.strike, linear_in_vol=True)

                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['forward_rate'] = this_swaption.forward_rate
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['strike_vol'] = strike_vol
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['price'] = this_swaption.price
                legs_info.setdefault(leg_name, {}).setdefault(tranche_name, {})['delta'] = this_swaption.delta

        # force unwind on 10Jun24 - JPM discontinued LIBOR Swaption data => need to reinitiate positions
        # as SOFR swaps/swaptions
        if not portfolio.is_empty() and self.time_stamp.date() >= date( 2024, 6, 7 ) and not self.strategy.is_SOFR:
            for leg_name in list(self.parameters['legs'].keys()):
                leg_portfolio = portfolio.get_position(leg_name)
                if leg_portfolio is not None:
                    leg_position_names = list(leg_portfolio.get_positions().keys())
                    for tranche_name in leg_position_names:
                        tranche_portfolio = leg_portfolio.get_position(tranche_name)
                        tranche_position_names = list(tranche_portfolio.get_positions().keys())
                        for position_name in tranche_position_names:
                            position = tranche_portfolio.get_position(position_name)
                            if isinstance(position.tradable, Cash):
                                continue
                            else:
                                unwind_price = position.tradable.price(market,
                                                                       self.strategy.valuer_map[type(position.tradable)],
                                                                       calc_types='price')
                                print( 'Rolling LIBOR position to SOFR position on %s (%s - %s - %s).'
                                       %(str(self.time_stamp.date()),
                                         leg_name,
                                         tranche_name,
                                         position_name))
                                portfolio.unwind((leg_name, tranche_name, position_name), unwind_price,
                                                 position.tradable.currency, cash_path=())
            # value portfolio
            portfolio.value_positions_at_market(market, fields=['price', 'forward_rate', 'delta'],
                                                valuer_map_override=self.strategy.valuer_map)
            price = portfolio.aggregate('price')
            price_pre_cost = price
            self.strategy.is_SOFR = True

        return RollingSwaptionsState(self.time_stamp, portfolio, price, price - price_pre_cost, legs_info)

class RollingSwaptions(DailyStrategy):
    def preprocess(self):
        super().preprocess()

        # TODO: move this to base class
        backtest_market = BMarket()
        for name, data in self.data_containers.items():
            backtest_market.add_item(data.get_market_key(), data)
        self.backtest_market = backtest_market
        self.valuer_map = {
            Swaption: SwaptionNormValuerOld(df_rates_type=self.parameters.get('df_rates_type', 'OIS')),
            ForwardStartSwap: ForwardStartSwapValuerOld(),
        }

        if self.start_date.date() > date( 2024, 6, 7 ):
            self.is_SOFR = True


        # initialize the trading cost module
        self.parameters['trading_cost'] = FlatTradingCost(self.parameters['tc_rate'])
        # initialize the roll logic module by leg and tranche
        rolls_by_leg_tranche = {}
        dh_by_leg_tranch = {}
        for leg_name, leg in self.parameters['legs'].items():
            for tranche_name, tranche in self.parameters['tranches'].items():
                rolls_by_leg_tranche.setdefault(leg_name, {})[tranche_name] = RollSwaptionContracts(
                    leg['expiry'], leg['tenor'], leg['strike'], leg['strike_type'], leg['style'], leg['currency'], None,
                    month_list=tranche['roll_months'], day=tranche['roll_day'], holidays=self.holidays,
                    backtest_market=self.backtest_market, valuer_map=self.valuer_map, old_data_format=True,
                )

                delta_hedger = (SwaptionDailyDH(backtest_market=self.backtest_market, valuer_map=self.valuer_map)
                                )
        self.parameters['rolls_by_leg_tranche'] = rolls_by_leg_tranche
        self.parameters['delta_hedger'] = delta_hedger

    def generate_events(self, dt: datetime):
        return [EODTradeEvent(dt, self)]
