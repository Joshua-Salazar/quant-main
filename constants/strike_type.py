from enum import Enum
from .. import ENABLE_PYVOLAR
if ENABLE_PYVOLAR:
    import pyvolar as vola


class StrikeType(Enum):
    """
    support vola strike type
    https://docs.voladynamics.com/docs/latest/API/cpp/html/group___strike_type.html#ga76e7ce4e1cf3e46bc438042e15933629
    """
    K=vola.StrikeType.K if ENABLE_PYVOLAR else "K"
    NS=vola.StrikeType.NS if ENABLE_PYVOLAR else "NS"
    KF=vola.StrikeType.KF if ENABLE_PYVOLAR else "KF"
    DELTA=vola.StrikeType.DELTA if ENABLE_PYVOLAR else "DELTA"
