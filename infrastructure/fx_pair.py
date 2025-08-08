from ..constants.ccy import Ccy
from ..tools.root import get_root
import os
import pandas as pd


class FXPair:
    def __init__(self, base_ccy: Ccy, term_ccy: Ccy):
        self.base_ccy = Ccy(base_ccy) if isinstance(base_ccy, str) else base_ccy
        self.term_ccy = Ccy(term_ccy) if isinstance(term_ccy, str) else term_ccy

    def clone(self):
        return FXPair(self.base_ccy, self.term_ccy)

    @classmethod
    def from_string(cls, pair_str: str):
        return cls(Ccy(pair_str[:3]), Ccy(pair_str[3:]))

    def to_string(self):
        return f"{self.base_ccy.value}{self.term_ccy.value}"

    def contains(self, ccy: Ccy):
        return ccy == self.base_ccy or ccy == self.term_ccy

    def inverse(self):
        return FXPair(self.term_ccy, self.base_ccy)

    def get_market_convention(self):
        market_convention_pairs = pd.read_csv(os.path.join(get_root(), "data/pair_order.csv"))
        return self if self.to_string() in market_convention_pairs.values else self.inverse()

    def __eq__(self, other):
        if not isinstance(other, FXPair):
            return False
        return self.base_ccy == other.base_ccy and self.term_ccy == other.term_ccy

