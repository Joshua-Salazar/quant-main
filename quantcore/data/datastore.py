import copy
import getpass
import logging
from functools import reduce

import pandas as pd

from ..data.arctic import ArcticDataStore, delim, get_default_arctic_credentials
from ..data.localparquetstore import LocalParquetDataStore
from ..data.memstore import MemoryDataStore

def replicate_items_to_ds(source, target, items_to_copy):
    for item in items_to_copy:
        data = source.read_item(item)
        target.write_item(data, item)


def get_ds_for_latest_snapshot(ds):
    try:
        snaps = sorted([s for s in ds.list_snapshots() if s.startswith(ds.collection)])
        return DataStore(
            db=ds.db,
            library=ds.library,
            collection=ds.collection,
            snapshot=snaps[-1],
            backend=ds.backend,
            credentials=ds.credentials,
        )
    except:   # noqa
        return ds


def read_item_across_collections(item_name, collections, library, db, snapshot=None):
    to_concat = []
    for collection in collections:
        library_ = library if type(library) == str else library[collection]
        db_ = db if type(db) == str else db[collection]
        ds = DataStore(db=db_, library=library_, collection=collection)
        if item_name in ds.list_items():
            df = ds.read_item(item_name=item_name, snapshot=snapshot)
            to_concat.append(df)
        else:
            raise Exception(f"Item {item_name} does not exist in datastore {ds}.")

    if all([type(elt) == pd.DataFrame for elt in to_concat]):
        if all([elt.shape[1] == 1 for elt in to_concat]):
            all_together = pd.concat(to_concat, axis=0)
        else:
            # Data in the format timestamps x stock IDs
            all_together = pd.concat(to_concat, axis=1)
    elif all([type(elt) == pd.Series for elt in to_concat]):
        all_together = pd.concat(to_concat, axis=0)
    elif all([type(elt) == list for elt in to_concat]):
        all_together = reduce(lambda x, y: x + y, to_concat)
    else:
        raise NotImplementedError(
            "Concatenation not implemented for this type of data !"
        )

    return all_together


def read_3d_item_across_collections(
    item_name, collections, library, db, snapshot, dates
):
    dic_3d_items = dict()
    for collection in collections:
        library_ = library if type(library) == str else library[collection]
        db_ = db if type(db) == str else db[collection]
        ds = DataStore(db=db_, library=library_, collection=collection)
        if item_name in ds.list_items_3d():
            try:
                data_item_3d = ds.read_item_3d(
                    item_name=item_name, snapshot=snapshot, dates=dates
                )
                dic_3d_items[collection] = data_item_3d
            except:   # noqa
                raise Exception(f'failed reading {item_name} from collection {collection} in datastore {ds}')
        else:
            raise Exception(f"Item {item_name} does not exist in datastore {ds}.")

    dic_concatenated = dict()
    for date in dates:
        to_concatenate = []
        for collection in collections:
            to_concatenate.append(dic_3d_items[collection][date].stack())
        dic_concatenated[date] = pd.concat(to_concatenate, axis=0).unstack(level=-1)

    return dic_concatenated


class DataStore:
    """
    DataStore class to load/save data, abstracting away the backend so that users can switch with no code change

    Inputs
    db      :  string - name of the user from which to read data (e.g. 'prod' or 'research' or 'pchambers')
                             defaults to <username>

    # Optional Inputs
    library     : e.g. 'core_data'
    collection  : e.g. 'EU'
    snapshot    : e.g. 'prod-20220101'

    Optional Inputs (power users)
    # backend              : e.g. 'arctic' (default)
    # credentials          : dictionary of credentials to access backend
    # reindex_df_on_read   : dict (e.g. {'index':[dt(2020,1,1), dt(2020,1,2), dt(2020,1,3)]}

    Example usage
    -------------
    ds = DataStore(db='quant', library='core_data', collection='EU')
    data = ds.read_item('unadjusted_price')    # read european prices from quant core_data library
    """

    def __init__(
        self,
        db,
        library=None,
        collection=None,
        snapshot=None,
        backend=None,
        credentials=None,
        reindex_df_on_read=None,
    ):
        # if db is not supplied, default to windows/linux username
        if db is not None:
            self.db = db
        else:
            self.db = getpass.getuser()

        self.backend = backend
        self.credentials = credentials

        # init object to implement interface for particular backenbd
        if backend == "arctic" or backend is None:
            arctic_credentials = credentials if credentials is not None else get_default_arctic_credentials(db=self.db)
            self.store = ArcticDataStore(arctic_credentials)
        elif backend == "in_memory":
            self.store = MemoryDataStore(credentials)
        elif backend == "parquet":
            self.store = LocalParquetDataStore(credentials=credentials)
        elif backend == 'arcticDB':
            from ..data.arcticDB import ArcticDBDataStore
            self.store = ArcticDBDataStore(credentials)
        elif backend in [
            "pystore",
        ]:
            raise NotImplementedError('backend "%s" not implemented yet' % backend)
        else:
            raise Exception('unknown backend "%s"' % backend)

        # save defaults for library and collection
        self.library = library
        self.collection = collection
        self.snapshot = snapshot

        if collection is not None and collection.startswith("_"):
            raise Exception("collection cannot begin with leading underscore")
        if collection is not None and collection == "":
            raise Exception("collection cannot be the empty string")
        if collection is not None and delim in collection:
            raise Exception(f"collection cannot contain {delim} character")

        # save default for re-index on read
        self.reindex_df_on_read = reindex_df_on_read

    def __repr__(self):
        parts = ["DataStore("]
        parts.append(self.store.connection_info)
        if self.db is not None:
            parts.append(",db:{}".format(self.db))   # noqa
        if self.library is not None:
            parts.append(",library:{}".format(self.library))
        if self.snapshot is not None:
            parts.append(",snapshot:{}".format(self.snapshot))
        if self.collection is not None:
            parts.append(",collection:{}".format(self.collection))
        parts.append(")")
        return "".join(parts)

    def read_item(
        self,
        item_name,
        collection=None,
        snapshot=None,
        library=None,
        db=None,
        index=None,
        columns=None,
        version=None,
    ):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        msg = (
            f"reading item: item {item_name}, collection {collection}, snapshot {snapshot}, library {library}, "
            + f"db {self.__db(db)}"
        )
        if version is not None:
            msg = f"{msg}, item version {version}"
        logging.info(msg)
        data = self.store.read_item(
            item_name=item_name,
            collection=collection,
            snapshot=snapshot,
            library=library,
            db=self.__db(db),
            version=version,
        )
        return self.__reindex_df(data, index, columns)

    def read_item_3d(
        self,
        item_name,
        collection=None,
        snapshot=None,
        library=None,
        db=None,
        dates=None,
    ):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        logging.info(
            "reading item 3d: item %s, collection %s, snapshot %s, library %s, db %s"
            % (item_name, collection, snapshot, library, self.__db(db))
        )
        if dates is None:
            dates = self.list_item_3d_dates(
                item_name, collection, snapshot, library, db
            )
        return self.store.read_item_3d(
            item_name, collection, snapshot, library, self.__db(db), dates
        )

    def stat_item(self, item_name, collection=None, library=None, db=None):
        collection, _, library = self.__resolve_params(collection, None, library)    # noqa
        logging.info(
            "stat item: item %s, collection %s, library %s, db %s"
            % (item_name, collection, library, self.__db(db))
        )
        return self.store.stat_item(item_name, collection, library, self.__db(db))

    def write_item(self, data, item_name, collection=None, library=None, db=None):
        if type(data) == str:
            raise Exception("first argument should be item to save, not a string")
        collection, _, library = self.__resolve_params(collection, None, library)     # noqa
        logging.info(
            "writing item: item %s, collection %s, library %s, db %s"
            % (item_name, collection, library, self.__db(db))
        )

        # write
        self.store.write_item(data, item_name, collection, library, self.__db(db))

    def update_item(
        self,
        data,
        item_name,
        collection=None,
        library=None,
        db=None,
        allow_overwrite=False,
        overwrite_only_last_index=False,
        index=None,
        columns=None
    ):
        if isinstance(data, str):
            raise Exception("first argument should be item to save, not a string")
        if not isinstance(data, pd.DataFrame):
            raise Exception("update only implemented for pd.DataFrame")
        collection, _, library = self.__resolve_params(collection, None, library)   # noqa
        logging.info(
            "updating item: item %s, collection %s, library %s, db %s (allow_overwrite=%s)"
            % (item_name, collection, library, self.__db(db), allow_overwrite)
        )

        # read previous item and append if exists
        existing_items = self.list_items(collection, None, library, db)
        if item_name in existing_items:
            existing_item = self.read_item(item_name, collection, None, library, db)
            if not isinstance(existing_item, pd.DataFrame):
                raise Exception(
                    "update only implemented where both items are of type pd.DataFrame"
                )
            existing_index_keys = set(existing_item.index)
            new_index_keys = set(data.index)
            overlap_keys = list(existing_index_keys.intersection(new_index_keys))
            if allow_overwrite:
                if overwrite_only_last_index:
                    # if the latest key is the same in both items (e.g. we have run a job twice on the same day),
                    # drop that key from the existing item, so then we can replace it with the new item
                    last_key = max(existing_index_keys)
                    if last_key == max(new_index_keys):
                        existing_item = existing_item.drop(index=last_key)
                        logging.info("update_item: dropped last row from existing item")
                        overlap_keys.remove(last_key)
                    # if the latest key is NOT the same in both items (which will usually be the case),
                    # then we recover the same behaviour as allow_overwrite=False (i.e. only append new keys
                    # from the new item)
                    if len(overlap_keys) > 0:
                        data = data.drop(index=overlap_keys)
                        logging.info(f"update_item: dropped {len(overlap_keys)} overlapping indices from new item")
                # drop common keys from existing item, to be replaced with the new item
                else:
                    existing_item = existing_item.drop(index=overlap_keys)
                    logging.info(f"update_item: dropped {len(overlap_keys)} overlapping indices from existing item")
            else:
                # drop common keys from new item,
                # so then we can append the new keys from the new item onto the existing item
                if len(overlap_keys) > 0:
                    data = data.drop(index=overlap_keys)
                    logging.info(f"update_item: dropped {len(overlap_keys)} overlapping indices from new item")

            if len(data.index) > 0 and len(existing_item.index) > 0:
                logging.info(f"merging item (min index = {min(data.index)}) with existing item " +
                             f"(max index = {max(existing_item.index)})")
            elif len(data.index) == 0:
                logging.info("no new data to merge")
            elif len(existing_item.index) == 0:
                logging.info("overwriting all rows in existing data")

            # actual merge
            data = pd.concat([existing_item, data], axis=0)

        # (optionally) re-shape
        if index is not None:
            data = data.reindex(index=index)
        if columns is not None:
            data = data.reindex(columns=columns)

        # write
        self.store.write_item(data, item_name, collection, library, self.__db(db))
        return data

    def write_item_3d(
        self,
        data,
        item_name,
        collection=None,
        library=None,
        db=None,
        full_replacement=True,
    ):
        collection, _, library = self.__resolve_params(collection, None, library)   # noqa

        if not full_replacement:
            raise NotImplementedError("'full_replacement' not implemented")

        logging.info(
            "writing item 3d: item %s, date/times %i, collection %s, library %s, db %s"
            % (item_name, len(data), collection, library, self.__db(db))
        )
        self.store.write_item_3d(data, item_name, collection, library, self.__db(db))

    def delete_item(self, item_name, collection=None, library=None, db=None):
        collection, _, library = self.__resolve_params(collection, None, library)    # noqa
        logging.info(
            "deleting item: item %s, collection %s, library %s, db %s"
            % (item_name, collection, library, self.__db(db))
        )
        self.store.delete_item(item_name, collection, library, self.__db(db))

    def delete_item_3d(self, item_name, collection=None, library=None, db=None, dates=None):
        collection, _, library = self.__resolve_params(collection, None, library)    # noqa
        logging.info(
            "deleting 3d item: item %s, collection %s, library %s, db %s"
            % (item_name, collection, library, self.__db(db))
        )
        self.store.delete_item_3d(item_name, collection, library, self.__db(db), dates)

    def list_item_3d_dates(
        self, item_name, collection=None, snapshot=None, library=None, db=None
    ):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        return sorted(
            self.store.list_item_3d_dates(
                item_name, collection, snapshot, library, self.__db(db)
            )
        )

    def list_items(self, collection=None, snapshot=None, library=None, db=None):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        return sorted(
            self.store.list_items(collection, snapshot, library, self.__db(db))
        )

    def list_items_3d(self, collection=None, snapshot=None, library=None, db=None):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        return sorted(
            self.store.list_items_3d(collection, snapshot, library, self.__db(db))
        )

    def list_snapshots(self, library=None, db=None):
        self.__validate_library(library)
        return self.store.list_snapshots(self.__library(library), self.__db(db))

    def create_snapshot(self, snapshot_name, collection=None, library=None, db=None):
        self.__validate_library(library)
        logging.info(
            "creating snapshot: snapshot %s, collection %s, library %s, db %s"
            % (
                snapshot_name,
                self.__collection(collection),
                self.__library(library),
                self.__db(db),
            )
        )
        self.store.create_snapshot(
            snapshot_name,
            self.__collection(collection),
            self.__library(library),
            self.__db(db),
        )

    def delete_snapshot(self, snapshot_name, library=None, db=None):
        self.__validate_library(library)
        logging.info(
            "deleting snapshot: snapshot %s, library %s, db %s"
            % (snapshot_name, self.__library(library), self.__db(db))
        )
        self.store.delete_snapshot(
            snapshot_name, self.__library(library), self.__db(db)
        )

    def list_libraries(self, db=None):
        return self.store.list_libraries(self.__db(db))

    def delete_library(self, library, db=None):
        self.store.delete_library(library, self.__db(db))

    def delete_collection(self, collection, library=None, db=None):
        # remove items
        items = self.list_items(collection, library=library, db=db)
        for item in items:
            self.delete_item(item, collection=collection, library=library, db=db)

        # remove 3d items
        items_3d = self.list_items_3d(collection, library=library, db=db)
        for item in items_3d:
            self.delete_item_3d(item, collection=collection, library=library, db=db)

    def list_collections(self, snapshot=None, library=None, db=None):
        return self.store.list_collections(
            self.__snapshot(snapshot), self.__library(library), self.__db(db)
        )

    def list_versions(self, item_name, collection=None, library=None, db=None):
        return self.store.list_versions(
            item_name=item_name,
            collection=self.__collection(collection),
            library=self.__library(library),
            db=self.__db(db),
        )

    def __resolve_params(self, collection: str, snapshot: str, library: str):
        # resolve
        collection = self.__collection(collection)
        snapshot = self.__snapshot(snapshot)
        library = self.__library(library)

        # validate
        if not library:
            raise Exception(
                "'library' must be specified - either as a DataStore constructor parameter or function parameter"
            )
        if not collection:
            raise Exception(
                "'collection' must be specified - either as a DataStore constructor parameter or function parameter"
            )

        # return a tuple
        return collection, snapshot, library

    def __validate_library(self, library):
        if library is None and self.library is None:
            raise Exception(
                "library must be specified - either as an object level parameter or as a keyword argument"
            )

    def __collection(self, c):
        return c if c is not None else self.collection

    def __db(self, u):
        return u if u is not None else self.db

    def __library(self, lib):
        return lib if lib is not None else self.library

    def __snapshot(self, s):
        return s if s is not None else self.snapshot

    def __reindex_df(self, data, index, columns):
        # validate inputs
        if (self.reindex_df_on_read is not None) and (
            (index is not None) or (columns is not None)
        ):
            raise Exception(
                "cannot specify either index or columns if @DataStore.reindex_df_on_read has been set"
            )

        # nothing to do if all None
        if (self.reindex_df_on_read is None) and (index is None) and (columns is None):
            return data

        # re-indexing only on DataFrames
        if type(data) not in [pd.DataFrame, pd.Series]:
            return data

        # what to reindex?
        if self.reindex_df_on_read is not None:
            reindex_dict = copy.deepcopy(self.reindex_df_on_read)
        else:
            reindex_dict = dict()
            if index is not None:
                reindex_dict["index"] = index
            if columns is not None:
                reindex_dict["columns"] = columns
        if type(data) == pd.Series and "columns" in reindex_dict:
            reindex_dict.pop("columns")

        # re-index
        for k in reindex_dict:
            data = data.reindex(**{k: reindex_dict[k]})

        return data

    def exists(
        self,
        item_name,
        collection=None,
        snapshot=None,
        library=None,
        db=None,
        version=None,
    ):
        collection, snapshot, library = self.__resolve_params(
            collection, snapshot, library
        )
        item_exists = self.store.exists(
            item_name=item_name,
            collection=collection,
            snapshot=snapshot,
            library=library,
            db=self.__db(db),
            version=version,
        )
        return item_exists


    def find_items(self, partial_name, collection=None, snapshot=None, library=None, db=None):
        """
        Find items based on a full or partial name match.

        Parameters
        ----------
        partial_name: str
            The full or partial name of the item to search for.
        collection: str, optional
            The collection to search in.
        snapshot: str, optional
            The snapshot to search in.
        library: str, optional
            The library to search in.
        db: str, optional
            The database to search in.

        Returns
        -------
        list
             A list of item names that match the provided partial name.
        """
        collection, snapshot, library = self.__resolve_params(collection, snapshot, library)
        all_items = self.store.list_items(collection=collection, snapshot=snapshot, library=library, db=self.__db(db))
        matching_items = [item for item in all_items if partial_name in item]
        return matching_items
