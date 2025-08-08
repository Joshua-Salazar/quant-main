from ..infrastructure.fx_pair import FXPair
from ..infrastructure import market_utils
from ..interface.ishock import IShock, ShockType
from ..interface.market_items.ispot import ISpot
from ..interface.market_item import MarketItem


class FXSpot(MarketItem, ISpot):
    def __init__(self, pair: FXPair, spot: float):
        self.pair = pair
        self.spot = spot

    def clone(self):
        return FXSpot(self.pair, self.spot)

    def get_market_key(self):
        return market_utils.create_fx_spot_key(self.pair)

    def get_spot(self):
        return self.spot

    def clone(self):
        return FXSpot(self.pair, self.spot)

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        cloned_spot = self.clone()
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                # return same spot for now. most case we use as fx conversion
                continue
            else:
                raise Exception("Not implemented yet")
        return cloned_spot