
# imports
from arcticdb.exceptions import NoDataFoundException  # noqa
from ..util.auth import get_credentials
import getpass
import pandas as pd
from datetime import datetime as dt
import numpy as np
import pytz

# Delimiter used to partition collection and name, into an arctic "symbol".
delim = "@"


def get_arcticdb_credentials(credentials_filename=None):

    # allow creds file to be overriden
    if credentials_filename is None:
        user = getpass.getuser()
        credentials_str = f'arcticdb_{user}'
    else:
        credentials_str = credentials_filename

    # setup dictionary
    credentials = get_credentials(credentials_str)

    # return
    return credentials


def get_arcticdb_connection(credentials):
    import arcticdb as adb  # noqa
    if 'uri' in credentials:
        return adb.Arctic(credentials['uri'])
    else:
        uri = "s3://s3.{rgn}.amazonaws.com:{bkt}?access={access}&secret={secret}".format(rgn=credentials['region'],
                                                                                         bkt=credentials['bucket'],
                                                                                         access=credentials['access'],
                                                                                         secret=credentials['secret'])
        return adb.Arctic(uri)


class ArcticDBDataStore:
    """
    @DataStore Hierarchy:
            # db
            #   library
            #     collection
            #       item

    ... is mapped to ArcticDB as follows:
           # arclibdb library = "db/library"    (following naming convention here:
                                                 https://docs.arcticdb.io/latest/aws_permissions/)
           # arcticdb symbol = "collection@item"
    """

    def __init__(self, credentials):

        # default credentials
        if credentials is None or isinstance(credentials, str):
            credentials = get_arcticdb_credentials(credentials)
        elif isinstance(credentials, dict):
            raise Exception("credentials must be None, dict or str")

        # connect
        self.conn = get_arcticdb_connection(credentials)
        self.__conn_info = self.conn.get_uri()

    @property
    def connection_info(self):
        return self.__conn_info

    def __get_library(self, library, db):
        if library is None:
            raise ValueError("datastore library cannot be None")
        if db is None:
            raise ValueError("datastore database cannot be None")

        # get live connection
        conn = self.__get_conn()

        # return reference to library
        adb_library = f"{db}/{library}"
        lib = conn.get_library(adb_library, create_if_missing=True)
        return lib

    def __get_conn(self):
        return self.conn

    def read_item(self, item_name, collection, snapshot, library, db, version):
        # get library
        lib = self.__get_library(library, db)

        if version is not None:
            # A version is more specific than a snapshot. If the snapshot is specified as well, that is unnecessary and
            # we ignore it
            symbol = f"{collection}{delim}{item_name}"
            return lib.read(symbol, as_of=version).data
        else:
            full_item_name = f"{collection}{delim}{item_name}"
            return lib.read(full_item_name, as_of=snapshot).data

    def write_item(self, data, item_name, collection, library, db):

        # get library
        lib = self.__get_library(library, db)

        # write
        full_item_name = f"{collection}{delim}{item_name}"
        lib.write(full_item_name, data)

    def read_item_3d(self, item_name, collection, snapshot, library, db, dates):
        lib = self.__get_library(library, db)
        existing_dates = self.list_item_3d_dates(
            item_name, collection, snapshot, library, db
        )
        dates_actual = [
            d for d in dates if d in existing_dates
        ]  # note: use list to preserve original order
        dates_missing = set(dates).difference(existing_dates)
        if len(dates_missing) > 0:
            raise Exception(f"{len(dates_missing)} dates missing")
        data = dict()
        for d in dates_actual:
            full_item_name = "_3d_{}_{}{}{}".format(
                self._serialise_datetime(d), collection, delim, item_name
            )
            data[d] = lib.read(full_item_name, as_of=snapshot).data
        return data

    def stat_item(self, item_name, collection, library, db):
        """For an item in the library/collection, return basic shape of the data, if
        available
        """
        lib = self.__get_library(library, db)
        full_item_name = "%s%s%s" % (collection, delim, item_name)
        item_info = lib.get_description(full_item_name)
        versions = lib.list_versions(full_item_name, latest_only=True)
        # TODO: check len is 1
        lastest_dt = versions[list(versions.keys())[0]].date  # datetime.datetime
        ncols = None
        if "col_names" in item_info:
            ncols = len(item_info.columns)
        return {
            "nrows": item_info.row_count,
            "size": (item_info.row_count, ncols),
            "date": lastest_dt.astimezone(pytz.utc),
            "ncols": ncols,
        }

    @staticmethod
    def _serialise_datetime(x):
        # Normalise python various timestamp types to a string
        if isinstance(x, pd.Timestamp) or isinstance(x, dt):
            return x.strftime("%Y%m%d-%H%M%S")
        else:
            raise Exception(
                "unable to serialise '{}' to date/time string; please use suitable date/time type".format(
                    x
                )
            )

    def write_item_3d(
            self, data, item_name, collection, library, db, full_replacement=True
    ):
        if not full_replacement:
            raise NotImplementedError("'full_replacement' not implemented")
        lib = self.__get_library(library, db)
        for date, item in data.items():
            rowid = ArcticDBDataStore._serialise_datetime(date)
            full_item_name = "_3d_{}_{}{}{}".format(rowid, collection, delim, item_name)
            lib.write(full_item_name, item)

    def delete_item(self, item_name, collection, library, db):

        # get library
        lib = self.__get_library(library, db)

        # delete
        # note: this function only deletes flat (non-3d) items [quietly ignores 3d items]
        full_item_name = "%s%s%s" % (collection, delim, item_name)
        if not self._is_3d_item(full_item_name):
            lib.delete(full_item_name)

    def delete_item_3d(self, item_name, collection, library, db, dates):
        # get library
        lib = self.__get_library(library, db)

        # get dates to delete (if not provided)
        if dates is None:
            dates = self.list_item_3d_dates(item_name, collection, None, library, db)

        # loop and delete all dates
        # note: this function will only delete 3d items [quietly ignores flat items]
        for date in dates:
            rowid = ArcticDBDataStore._serialise_datetime(date)
            full_item_name = "_3d_{}_{}{}{}".format(rowid, collection, delim, item_name)
            lib.delete(full_item_name)

    @staticmethod
    def _is_3d_item(symbol: str):
        return symbol.startswith("_3d")

    @staticmethod
    def _split_3d_item_name(symbol: str):
        # expect format to be `_3d_20200201-000000_.*`
        delim_loc = 12  # len("_3d_20200201")
        if len(symbol) <= delim_loc:
            raise Exception(f"name '{symbol}' does not match pattern of a 3d-item")
        return symbol[4: delim_loc + 6 + 1], symbol[delim_loc + 6 + 2:]

    def _list_symbols(self, snapshot, library, db):
        lib = self.__get_library(library, db)
        return lib.list_symbols(snapshot_name=snapshot)

    def _list_symbols_and_types(self, snapshot, library, db):
        """Return a set of pairs (<COL>_<NAME>, <TYPE>)."""
        typed_symbols = set()

        for symbol in self._list_symbols(snapshot, library, db):
            if symbol.startswith("_3d_"):
                date, name = ArcticDBDataStore._split_3d_item_name(symbol)
                typed_symbols.add((name, "3d"))
            else:
                typed_symbols.add((symbol, "flat"))
        return typed_symbols

    @staticmethod
    def _filter_collection_symbols(symbols, collection):
        prefix = f"{collection}{delim}"
        return [s.replace(prefix, "", 1) for s in symbols if s.startswith(prefix)]

    def list_items_3d(self, collection, snapshot, library, db):
        typed_symbols = self._list_symbols_and_types(snapshot, library, db)
        symbols_3d = [n for n, t in typed_symbols if t == "3d"]
        return self._filter_collection_symbols(symbols_3d, collection)

    def list_items(self, collection, snapshot, library, db):
        # note: this function deliberately does NOT return 3d items
        typed_symbols = self._list_symbols_and_types(snapshot, library, db)
        symbols_flat = [n for n, t in typed_symbols if t == "flat"]
        return self._filter_collection_symbols(symbols_flat, collection)

    def list_item_3d_dates(self, item_name, collection, snapshot, library, db):
        lib = self.__get_library(library, db)
        result = []
        expected_col_item_name = f"{collection}{delim}{item_name}"
        for raw_symbol in lib.list_symbols(snapshot_name=snapshot):
            if self._is_3d_item(raw_symbol):
                d, col_item_name = self._split_3d_item_name(raw_symbol)
                if col_item_name == expected_col_item_name:
                    result.append(dt.strptime(d, "%Y%m%d-%H%M%S"))
        return result

    def list_libraries(self, db):
        # get connection
        conn = self.__get_conn()

        # get all libraries
        libs = conn.list_libraries()

        # filter on this user
        user_libs = [ul.replace(db + ".", "", 1) for ul in libs if ul.startswith(db + ".")]
        return user_libs

    def delete_library(self, library, db):
        # get connection
        conn = self.__get_conn()

        # get library name
        full_lib_name = "%s.%s" % (db, library)

        # delete
        conn.delete_library(full_lib_name)

    def list_collections(self, snapshot, library, db):
        # get library
        lib = self.__get_library(library, db)

        # get symbols
        all_symbols = lib.list_symbols(snapshot_name=snapshot)

        # convert 3d symbols into item-symbol
        item_names = set()
        for raw_symbol in all_symbols:
            if self._is_3d_item(raw_symbol):
                item_names.add(self._split_3d_item_name(raw_symbol)[1])
            else:
                item_names.add(raw_symbol)
        all_symbols = item_names

        # extract collection
        all_collections = list(np.unique([s.split(delim)[0] for s in all_symbols]))
        return all_collections

    def list_snapshots(self, library, db):
        lib = self.__get_library(library, db)
        return list(lib.list_snapshots().keys())

    def create_snapshot(self, snapshot_name, collection, library, db):
        lib = self.__get_library(library, db)
        if collection is None:
            lib.snapshot(snapshot_name)
        else:
            all_symbols_in_library = self._list_symbols(None, library, db)
            all_symbols_in_collection = [
                s for s in all_symbols_in_library if s.find(collection + delim) > -1
            ]
            symbols_to_exclude = set(all_symbols_in_library).difference(
                all_symbols_in_collection
            )
            lib.snapshot(snapshot_name, skip_symbols=list(symbols_to_exclude))

    def delete_snapshot(self, snapshot_name, library, db):
        lib = self.__get_library(library, db)
        lib.delete_snapshot(snapshot_name)

    def list_versions(self, item_name, collection, library, db):
        lib = self.__get_library(library, db)
        symbol = f"{collection}@{item_name}"
        versions = lib.list_versions(symbol)

        return versions

    def exists(self, item_name, collection, snapshot, library, db, version):
        try:
            self.read_item(
                item_name=item_name,
                collection=collection,
                snapshot=snapshot,
                library=library,
                db=db,
                version=version,
            )
        except NoDataFoundException:
            return False

        return True
