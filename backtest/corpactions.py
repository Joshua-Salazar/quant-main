from datetime import datetime

from ..tradable.option import Option
from ..tradable.stock import Stock
from ..tradable.cash import Cash
from ..tradable.portfolio import Portfolio
from ..tradable.position import Position
from ..infrastructure.corpaction_data_container import SpecialCash, Split
from ..data.market import find_nearest_listed_options


class CorpActionsProcessor:
    def __init__(self):
        pass

    def apply(self, dt: datetime, portfolio: Portfolio, market, use_listed, position_path=()):
        children = {k: v for k, v in portfolio.get_position(position_path).get_positions().items()}
        for k, v in children.items():
            if isinstance(v, Position):
                if isinstance(v.tradable, Stock):
                    dividends = market.get_dividend(v.tradable.ticker)
                    corpactions = market.get_corpaction(v.tradable.ticker)
                    if dividends is not None:
                        for dividend in dividends:
                            if dividend.ticker == v.tradable.ticker:
                                portfolio.add_position(Cash(dividend.currency), v.quantity * dividend.amount, position_path=position_path)
                    if corpactions is not None:
                        for corpaction in corpactions:
                            if corpaction.ticker == v.tradable.ticker:
                                if isinstance(corpaction, SpecialCash):
                                    portfolio.add_position(Cash(corpaction.currency), v.quantity * corpaction.amount, position_path=position_path)
                                elif isinstance(corpaction, Split):
                                    portfolio.add_position(v.tradable, v.quantity * (corpaction.r_factor - 1), position_path=position_path)
                                else:
                                    raise RuntimeError(f'Unknown corp action type {corpaction}')
                elif isinstance(v.tradable, Option):
                    corpactions = market.get_corpaction(v.tradable.underlying)
                    if corpactions is not None:
                        for corpaction in corpactions:
                            if corpaction.ticker == v.tradable.underlying:
                                # TODO: this is approximate, but we move out of the old option, and move in the new option with equivalent strike and quantity
                                # need to check this is pnl neutral
                                if isinstance(corpaction, SpecialCash):
                                    portfolio.add_position(v.tradable, -v.quantity, position_path=position_path)
                                    new_strike = v.tradable.strike - corpaction.amount
                                    new_quantity = v.quantity
                                    if use_listed:
                                        option_universe = market.get_option_universe(v.tradable.root, return_as_dict=False)
                                        new_option = find_nearest_listed_options(v.tradable.expiration, new_strike,
                                                                                 'C' if v.tradable.is_call else 'P',
                                                                                 option_universe,
                                                                                 return_as_tradables=True)
                                        assert len(new_option) >= 1
                                        new_option = new_option[0]
                                        new_option = Option(v.tradable.root, v.tradable.underlying, v.tradable.currency,
                                                            new_option.expiration, new_option.strike, v.tradable.is_call,
                                                            v.tradable.is_american, v.tradable.contract_size,
                                                            v.tradable.tz_name, new_option.listed_ticker)
                                    else:
                                        new_option = Option(v.tradable.root, v.tradable.underlying, v.tradable.currency, v.tradable.expiration, new_strike, v.tradable.is_call, v.tradable.is_american, v.tradable.contract_size, v.tradable.tz_name, v.tradable.listed_ticker)
                                    portfolio.add_position(new_option, new_quantity, position_path=position_path)
                                elif isinstance(corpaction, Split):
                                    portfolio.add_position(v.tradable, -v.quantity, position_path=position_path)
                                    new_strike = v.tradable.strike / corpaction.r_factor
                                    new_quantity = v.quantity * corpaction.r_factor
                                    if use_listed:
                                        option_universe = market.get_option_universe(v.tradable.root, return_as_dict=False)
                                        new_option = find_nearest_listed_options(v.tradable.expiration, new_strike,
                                                                                 'C' if v.tradable.is_call else 'P',
                                                                                 option_universe,
                                                                                 return_as_tradables=True)
                                        assert len(new_option) >= 1
                                        new_option = new_option[0]
                                        new_option = Option(v.tradable.root, v.tradable.underlying, v.tradable.currency,
                                                            new_option.expiration, new_option.strike, v.tradable.is_call,
                                                            v.tradable.is_american, v.tradable.contract_size,
                                                            v.tradable.tz_name, new_option.listed_ticker)
                                    else:
                                        new_option = Option(v.tradable.root, v.tradable.underlying, v.tradable.currency,
                                                            v.tradable.expiration, new_strike, v.tradable.is_call,
                                                            v.tradable.is_american, v.tradable.contract_size,
                                                            v.tradable.tz_name, v.tradable.listed_ticker)

                                    portfolio.add_position(new_option, new_quantity, position_path=position_path)
                                else:
                                    raise RuntimeError(f'Unknown corp action type {corpaction}')
            else:
                self.apply(dt, portfolio, market, use_listed, position_path=position_path + (k,))
