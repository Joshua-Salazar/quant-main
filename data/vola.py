from ..constants.underlying_type import UnderlyingType
from ..tradable.future import Future


def find_underlying(underlyig_type: UnderlyingType, root, expiration, tz_name='', currency=""):
    if underlyig_type == UnderlyingType.EQUITY:
        return root
    elif underlyig_type == UnderlyingType.FUTURES:
        # TODO: for now we only return a meaningful root and expiration
        return Future(root, currency, expiration, "", "", tz_name)
    else:
        raise RuntimeError('Unknown vola surface type')
