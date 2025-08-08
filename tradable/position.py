import numbers
import numpy as np
from datetime import datetime
from ..interface.itradable import ITradable


class Position:
    def __init__(self, tradable: ITradable, quantity: float):
        self.tradable = tradable
        self.quantity = quantity

    def clone(self):
        return Position(self.tradable.clone(), self.quantity)

    def add(self, another):
        if another is None:
            return self
        if isinstance(another, Position):
            assert self.tradable.name() == another.tradable.name()
            new_position = self.clone()
            new_position.quantity = new_position.quantity + another.quantity
            return new_position
        elif isinstance(another, numbers.Number):
            new_position = self.clone()
            new_position.quantity = new_position.quantity + another
            return new_position
        else:
            raise RuntimeError('Unknown type of another ' + type(another))

    def value(self, market, fields='price', valuer_map_override={}, currency=None, defaults=None, **kwargs):
        if defaults is None and isinstance(fields, list):
            defaults = [None] * len(fields)

        def value_func(_x, **kwargs):
            _results = _x.price(market, valuer=valuer_map_override.get(type(_x), None), calc_types=fields,
                                currency=currency, **kwargs)
            if _results is None:
                _results = defaults
            if not isinstance(_results, tuple):
                _results = [_results]
            return _results
        res = value_func(self.tradable, **kwargs)
        return np.array(res) * self.quantity


class ValuedPosition(Position):
    def __init__(self, tradable, quantity, **kwargs):
        self.tradable = tradable
        self.quantity = quantity
        for k, v in kwargs.items():
            # check everything is clonable
            assert type(v) in [str, int, np.int64, float, np.float64, np.array, np.ndarray, datetime] or v is None,\
                'Attribute a valued position cannot be of type ' + str(type(v))
            setattr(self, k, v)

    def get_additional_attributes(self):
        attributes = {}
        for k, v in self.__dict__.items():
            if k not in ['tradable', 'quantity']:
                attributes[k] = v
        return attributes

    def clone(self):
        new_object = ValuedPosition(self.tradable.clone(), self.quantity)
        for k, v in self.__dict__.items():
            if k not in ['tradable', 'quantity']:
                setattr(new_object, k, v)
        return new_object
