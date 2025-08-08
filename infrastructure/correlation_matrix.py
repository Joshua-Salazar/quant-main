from ..infrastructure import market_utils
from ..interface.ishock import IShock, ShockType
from ..interface.market_item import MarketItem


class CorrelationMatrix(MarketItem):

    def __init__(self, corr_matrix):
        self.corr_matrix = corr_matrix
        self.market_key = market_utils.create_correlation_matrix(corr_matrix)

    def get_market_key(self):
        return self.market_key

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        cloned_corr_matrix = self.clone()
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                # return same spot for now. most case we use as fx conversion
                continue
            else:
                raise Exception("Not implemented yet")
        return cloned_corr_matrix

    def clone(self):
        return CorrelationMatrix(self.corr_matrix)