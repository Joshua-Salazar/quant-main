from ..infrastructure import market_utils
from ..interface.ishock import IShock, ShockType
from ..interface.market_items.ispot import ISpot
from ..interface.market_item import MarketItem


class EQSpot(MarketItem, ISpot):
    def __init__(self, underlying: str, spot: float):
        self.underlying = underlying
        self.spot = spot

    def get_market_key(self):
        return market_utils.create_spot_key(self.underlying)

    def get_spot(self):
        return self.spot

    def clone(self):
        return EQSpot(self.underlying, self.spot)

    def apply(self, shocks: [IShock], original_market, **kwargs) -> MarketItem:
        cloned_spot = self.clone()
        for shock in shocks:
            if shock.type == ShockType.DATETIMESHIFT:
                # return same spot for now. most case we use as fx conversion
                continue
            else:
                raise Exception("Not implemented yet")
        return cloned_spot
