""" The Arctic TimeSeries and Tick store."""

from .arctic import Arctic, register_library_type
from .arctic import VERSION_STORE, TICK_STORE, CHUNK_STORE
from .store._ndarray_store import NdarrayStore
from .store._pandas_ndarray_store import PandasDataFrameStore, PandasSeriesStore, PandasPanelStore
from .store.version_store import register_versioned_storage, register_version

try:
    from pkg_resources import get_distribution
    # The github version of Arctic will attempt to identify its version
    # information via package discovery within the environment.  That doesn't
    # work for us, because we dont install our local version into our conda
    # environment. -Darren
    str_version = "1.8.4" # Hard coded. -Darren
    int_parts = tuple(int(x) for x in str_version.split('.'))
    num_version = sum([1000 ** i * v for i, v in enumerate(reversed(int_parts))])
    register_version(str_version, num_version)
except Exception:
    __version__ = None
    __version_parts__ = tuple()
    __version_numerical__ = 0
else:
    __version__ = str_version
    __version_parts__ = int_parts
    __version_numerical__ = num_version


register_versioned_storage(PandasDataFrameStore)
register_versioned_storage(PandasSeriesStore)
# register_versioned_storage(PandasPanelStore) # PJC edit 02.03.2022
register_versioned_storage(NdarrayStore)
