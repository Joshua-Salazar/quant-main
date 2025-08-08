from .indicator import ExternalIndicator
from .strategy import Event, StrategyState
from ..tradable.cash import Cash
from ..tradable.constant import Constant
from ..tradable.portfolio import Portfolio


class BaseUnwindEvent(Event):
    def __init__(self, dt, strategy):
        super().__init__(dt, strategy)
        self.need_unwind = False
        self.target_names = []
        self.init()

    def init(self):
        for leg_name, leg in self.parameters["legs"].items():
            if "unwind" in leg:
                self.need_unwind = True
                if leg["unwind"].name not in self.target_names:
                    self.target_names.append(leg["unwind"].name)
        if self.need_unwind and "price" not in self.target_names:
            self.target_names.append("price")

    def unwind_pfo(self, target_tradable_type, state: StrategyState):
        assert self.need_unwind
        portfolio = state.portfolio.clone()
        market = self.strategy.backtest_market.get_market(self.time_stamp)
        # value portfolio then apply costs then price portfolio
        portfolio.value_positions_at_market(market, fields=self.target_names, valuer_map_override=self.strategy.valuer_map)
        for leg_name, leg in self.parameters["legs"].items():
            if "unwind" not in leg:
                continue
            # unwind leg
            leg_pfo = portfolio.get_position(leg_name)
            if leg_pfo is not None:
                unwind_pos_names = []
                for pos_name, pos in leg_pfo.net_positions(tradable_type=target_tradable_type).items():
                    if isinstance(leg["unwind"], ExternalIndicator):
                        indicator = leg["unwind"].get_indicator(self.time_stamp)
                        if indicator != 0:
                            unwind_pos_names.append(pos_name)
                    else:
                        target_value = getattr(pos, leg["unwind"].name)
                        if leg["unwind"].one_range[0] <= target_value <= leg["unwind"].one_range[1]:
                            unwind_pos_names.append(pos_name)
                if len(unwind_pos_names) > 0:
                    for pos_name in list(leg_pfo.get_positions().keys()):
                        pos = portfolio.get_position((leg_name, pos_name))
                        if isinstance(pos, Portfolio):
                            if pos_name == "delta_hedge":   # delta hedge pfo under leg
                                portfolio.unwind_single_pfo((leg_name, pos_name))
                            else:
                                for pfo_pos_name in list(pos.get_positions().keys()):
                                    if pfo_pos_name == "delta_hedge":   # delta hedge pfo under leg/tranch
                                        portfolio.unwind_single_pfo((leg_name, pos_name, pfo_pos_name))
                                    else:
                                        pos = portfolio.get_position((leg_name, pos_name, pfo_pos_name))
                                        if pfo_pos_name in unwind_pos_names:
                                            portfolio.unwind((leg_name, pos_name, pfo_pos_name), pos.price)
                        else:
                            if pos_name in unwind_pos_names:
                                portfolio.unwind((leg_name, pos_name), pos.price)

        if "trading_cost" in self.parameters:
            pre_trade_ptf = state.portfolio
            post_trade_ptf = portfolio
            self.parameters['trading_cost'].apply(post_trade_ptf, pre_trade_ptf, self.time_stamp, market=market, valuer_map=self.strategy.valuer_map)
        return portfolio
