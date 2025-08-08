import numbers

from ..analytics.utils import float_equal
from ..dates.utils import MAX_DATETIME, coerce_timezone
from ..interface.imarket import IMarket
from ..tradable.cash import Cash
from ..tradable.constant import Constant
from ..tradable.future import Future
from ..interface.itradable import ITradable
from ..reporting.trade_reporter import TradeReporter
from ..tradable.position import Position, ValuedPosition
from datetime import datetime
import typing


class Portfolio:
    def __init__(self, positions, remove_zero=True):
        self.float_equal_threshold = 1e-12
        self.root = {}
        positions_tree = {}
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, tuple):
                    name = pos[0]
                    positions_tree[name] = pos[1].add(positions_tree.get(name, None))
                else:
                    name = pos.tradable.name()
                    positions_tree[name] = pos.add(positions_tree.get(name, None))
        elif isinstance(positions, dict):
            positions_tree = positions
        else:
            raise RuntimeError('input positions has to be a list')
        self.root = positions_tree
        if remove_zero:
            self.remove_zero_positions()

    def clone(self, remove_zero=True):
        new_root = {}
        for k, v in self.root.items():
            # TODO: need to properly check key doesn't need to be cloned
            assert isinstance(k, str) or isinstance(k, datetime) or isinstance(k, numbers.Number)
            new_root[k] = v.clone(remove_zero) if isinstance(v, Portfolio) else v.clone()
        return Portfolio(new_root, remove_zero)

    def get_positions(self):
        return self.root

    def get_position(self, name):
        if isinstance(name, tuple):
            if len(name) == 0:
                return self
            if len(name) == 1:
                return self.root.get(name[0], None)
            sub_portfolio = self.root.get(name[0], None)
            return None if sub_portfolio is None else sub_portfolio.get_position(name[1:])
        else:
            return self.root.get(name, None)

    def find_children_of_tradable_type(self, tradable_type):
        children = []
        if not isinstance(tradable_type, list):
            tradable_type = [tradable_type]
        for k, v in self.root.items():
            if isinstance(v, Position) and any([isinstance(v.tradable, type) for type in tradable_type]):
                children.append((k, v))
        return children

    def find_children_of_tradable_type_recursive(self, tradable_type):
        children = []
        if not isinstance(tradable_type, list):
            tradable_type = [tradable_type]
        for k, v in self.root.items():
            if isinstance(v, Position) and any([isinstance(v.tradable, type) for type in tradable_type]):
                children.append((k, v))
            elif isinstance(v, Portfolio):
                children = children + v.find_children_of_tradable_type_recursive(tradable_type)
        return children

    def filter_tradable_type(self, tradable_type, remove_zero=True):
        children = self.find_children_of_tradable_type_recursive(tradable_type)
        return Portfolio(children, remove_zero=remove_zero)

    def scale(self, scaler):
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                v.scale(scaler)
            else:
                v.quantity *= scaler

    def set_position(self, name, position):
        if isinstance(name, tuple):
            if len(name) == 1:
                self.root[name[0]] = position
            else:
                sub_portfolio = self.root.setdefault(name[0], Portfolio([]))
                sub_portfolio.set_position(name[1:], position)
        else:
            self.root[name] = position

    def value_position(self, name, value_func, **kwargs):
        pos = self.get_position(name)
        if isinstance(pos, ValuedPosition):
            attributes = pos.get_additional_attributes()
        else:
            attributes = {}
        attributes.update(value_func(pos.tradable, **kwargs))
        self.set_position(name, ValuedPosition(pos.tradable, pos.quantity, **attributes))

    def _net_positions(self, net_positions, tradable_type=None):
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                v._net_positions(net_positions, tradable_type)
            else:
                if tradable_type is None:
                    net_positions[k] = v.add(net_positions.get(k, None))
                elif isinstance(v, Position) and isinstance(v.tradable, tradable_type):
                    net_positions[k] = v.add(net_positions.get(k, None))

    def net_positions(self, tradable_type=None):
        net_positions = {}
        self._net_positions(net_positions, tradable_type)
        return net_positions

    def value_positions(self, value_func, **kwargs ):
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                v.value_positions(value_func, **kwargs)
            else:
                self.value_position(k, value_func, **kwargs)

    def value_positions_at_market(self, market, fields, valuer_map_override={}, currency=None, defaults=None, **kwargs ):
        if defaults is None:
            defaults = [None for i in range(len(fields))]

        def value_func(_x):
            _results = _x.price(market, valuer=valuer_map_override.get(type(_x), None), calc_types=fields, currency=currency, **kwargs )
            if not isinstance(_results, tuple):
                _results = [_results]
            _struct_to_return = {}
            for _f, _r, _d in zip(fields, _results, defaults):
                _struct_to_return[_f] = _d if _r is None else _r
            return _struct_to_return

        self.value_positions(value_func)

    def remove_zero_positions(self):
        remove_list = []
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                v.remove_zero_positions()
                if v.is_empty():
                    remove_list.append(k)
            else:
                if isinstance(v.quantity, numbers.Number):
                    if float_equal(v.quantity, 0, self.float_equal_threshold):
                        remove_list.append(k)
        for k in remove_list:
            del self.root[k]

    def is_empty(self):
        return len(self.root) == 0

    def remove_all(self):
        self.root = {}

    def aggregate(self, fields_or_func_of_position, default_value=None):

        field_or_func_of_position_list = fields_or_func_of_position if isinstance(fields_or_func_of_position, list) else [fields_or_func_of_position]
        values = []
        for field_or_func_of_position in field_or_func_of_position_list:
            if isinstance(field_or_func_of_position, str):
                func_of_position = lambda x: getattr(x, field_or_func_of_position, default_value)
            else:
                func_of_position = field_or_func_of_position

            value = 0
            for k, v in self.root.items():
                if isinstance(v, Portfolio):
                    value += v.aggregate(func_of_position, default_value=default_value)
                else:
                    func_pos = func_of_position(v)
                    #TODO: generalise bid-ask err fix below
                    if ('Cash' in k and func_pos is None) or ('USDStock' in k and func_pos is None):
                        continue
                    if default_value is None and func_pos is None:
                        # skip if func_pos is None and we have default is also None
                        # e.g. option ask for iv and future not have iv
                        continue
                    value += func_pos * v.quantity
            values.append(value)
        return values if isinstance(fields_or_func_of_position, list) else values[0]

    def price(self, value_func, field, **kwargs):
        self.value_positions(value_func, **kwargs)
        return self.aggregate(field)

    def price_at_market(self, market, fields: typing.Union[str, list], valuer_map_override={}, currency=None, default=None, **kwargs):
        def value_func(_x, **kwargs):
            _result = _x.price(market, valuer=valuer_map_override.get(type(_x), None), calc_types=fields, currency=currency, **kwargs )
            if _result is None:
                _result = default
            if isinstance(fields, str):
                return {fields: _result}
            else:
                if len(fields) == 1:
                    if isinstance(_result, tuple) or isinstance(_result, list):
                        return {fields[0]: _result[0]}
                    else:
                        return {fields[0]: _result}
                else:
                    return dict(zip(fields, _result))

        return self.price(value_func, fields, **kwargs)

    def remove_empty_portfolio_along_path(self, position_path):
        if len(position_path) == 0:
            return
        else:
            if self.get_position(position_path).is_empty():
                del self.get_position(position_path[:-1]).root[position_path[-1]]
                self.remove_empty_portfolio_along_path(position_path[:-1])
            else:
                return

    def add_position(self, tradable_to_trade: ITradable, quantity_to_trade, position_path=(), remove_zero=True):
        position_key = tradable_to_trade.name()
        position_name = position_path + (position_key,)
        existing_position = self.get_position(position_name)
        new_pos = Position(tradable_to_trade, quantity_to_trade).add(existing_position)
        if remove_zero:
            if not (isinstance(new_pos.quantity, numbers.Number) and float_equal(new_pos.quantity, 0, self.float_equal_threshold)):
                self.set_position(position_name, new_pos)
            else:
                if existing_position is not None:
                    del self.get_position(position_path).root[position_key]
                    self.remove_empty_portfolio_along_path(position_path)
        else:
            self.set_position(position_name, new_pos)
        return position_name

    def move(self, tradable_to_trade, quantity_to_trade, from_path, to_path):
        self.add_position(tradable_to_trade, -quantity_to_trade, from_path)
        self.add_position(tradable_to_trade, quantity_to_trade, to_path)

    # def add_position_by_name(self, position_name_to_trade, quantity_to_trade):
    #     existing_position = self.get_position(position_name_to_trade)
    #     if existing_position is None:
    #         raise RuntimeError('If position is specified by position name, it has to exist already')
    #     else:
    #         self.set_position(position_name_to_trade, existing_position.add(quantity_to_trade))
    #         self.remove_zero_positions()
    #     return position_name_to_trade

    def trade(self, tradable_to_trade, quantity_to_trade, execution_price, execution_currency=None, position_path=(), cash_path=None, remove_zero=True):
        if execution_currency is None:
            execution_currency = tradable_to_trade.currency
        if cash_path is None:
            cash_path = position_path
        new_position_name = self.add_position(tradable_to_trade, quantity_to_trade, position_path=position_path, remove_zero=remove_zero)
        if isinstance(tradable_to_trade, Future):
            self.add_position(Constant(execution_currency), -execution_price * quantity_to_trade,
                              position_path=position_path, remove_zero=remove_zero)
        else:
            self.add_position(Cash(execution_currency), -execution_price * quantity_to_trade,
                              position_path=cash_path, remove_zero=remove_zero)
        return new_position_name

    def unwind(self, position_name, unwind_price, unwind_currency=None, cash_path=None, remove_zero=True):
        pos = self.get_position(position_name)
        if isinstance(position_name, tuple):
            self.trade(pos.tradable, -pos.quantity, unwind_price, unwind_currency, position_path=position_name[:-1], cash_path=cash_path, remove_zero=remove_zero)
        else:
            self.trade(pos.tradable, -pos.quantity, unwind_price, unwind_currency, cash_path=cash_path, remove_zero=remove_zero)

    def unwind_single_pfo(self, pfo_path):
        pfo = self.get_position(pfo_path)
        assert isinstance(pfo, Portfolio)
        for pos_name in list(pfo.get_positions().keys()):
            pos_path = pfo_path + (pos_name,)
            pos = self.get_position(pos_path)
            assert not isinstance(pos, Portfolio)
            if not isinstance(pos.tradable, Constant) and not isinstance(pos.tradable, Cash):
                self.unwind(pos_path, pos.price)

    def get_cash_flows(self, market: IMarket):
        cash_flows = {}
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                for ccy, amount in v.get_cash_flows(market).items():
                    cash_flows[ccy] = cash_flows.get(ccy, 0.) + amount
            else:
                for ccy, amount in TradeReporter(v.tradable).get_cash_flows(market).items():
                    cash_flows[ccy] = cash_flows.get(ccy, 0.) + amount
        return cash_flows

    def get_expiration(self, skip_future=False):
        """
        return minimum expiration over all legs
        """
        min_expiry = None
        for k, v in self.root.items():
            if isinstance(v, Portfolio):
                expiry = v.get_expiration()
                min_expiry = expiry if min_expiry is None else min(min_expiry, expiry)
            else:
                if skip_future and isinstance(v.tradable, Future):
                    continue
                if v.tradable.has_expiration():
                    expiry = v.tradable.expiration
                    min_expiry = expiry if min_expiry is None else min(coerce_timezone(min_expiry, expiry))
        return MAX_DATETIME if min_expiry is None else min_expiry
