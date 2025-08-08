from datetime import datetime
from ..constants.underlying_type import UnderlyingType


def find_underlying(underlying_type: UnderlyingType, root: str, expiration: datetime, tz_name: str=""):
    if underlying_type == UnderlyingType.EQUITY:
        return root
    elif underlying_type == UnderlyingType.FUTURES:
        # TODO: for now we only return a meaningful root and expiration
        return (root, expiration, tz_name)
    else:
        raise RuntimeError('Unknown vola surface type')