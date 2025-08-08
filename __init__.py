from . import analytics
import importlib
ENABLE_PYVOLAR = importlib.util.find_spec("pyvolar") is not None
try:
    from ctp.models.vol import NxEQVolatilityEngine
    ENABLE_NXP = True
except:
    ENABLE_NXP = False
ENABLE_CTP = not ENABLE_NXP and importlib.util.find_spec("ctp") is not None
