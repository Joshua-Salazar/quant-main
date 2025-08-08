class MemoryDataStore:
    """
    Hierarchy
            # db
            # library
            # collection
            # item

    """

    def __init__(self, credentials):
        self.__data = dict()
        self.connection_info = 'in_memory'

    def read_item(self, item_name, collection, snapshot, library, db, version):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        if version is not None:
            raise NotImplementedError('Versions not supported for the memory datastore')
        if self.__check_exits(item_name, collection, snapshot, library, db, 'standard'):
            return self.__data[db][library][collection]['standard'][item_name]
        else:
            raise Exception(
                f'item {item_name} does not exist in collection {collection}, snapshot {snapshot}, library {library}, user database {db}')

    def read_item_3d(self, item_name, collection, snapshot, library, db, dates):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        if not self.__check_exits(item_name, collection, snapshot, library, db, '3d'):
            raise Exception(f'3d item {item_name} does not exist in collection {collection}, snapshot {snapshot}, library {library}, user database {db}')
        missing_dates = list(set(dates).difference(self.list_item_3d_dates(item_name, collection, snapshot, library, db)))
        if len(missing_dates) > 0:
            raise Exception('3d dates not available: {}'.format(','.join(missing_dates)))

        return dict([(d, self.__data[db][library][collection]['3d'][item_name][d]) for d in dates])

    def write_item(self, data, item_name, collection, library, db):
        self.__init_if_not_present(collection, library, db)
        self.__data[db][library][collection]['standard'][item_name] = data

    def write_item_3d(self, data, item_name, collection, library, db, full_replacement=True):
        self.__init_if_not_present(collection, library, db)
        if type(data) != dict:
            raise Exception('3d items must be dictionaries')
        if full_replacement:
            self.__data[db][library][collection]['3d'][item_name] = data
        else:
            if item_name not in self.__data[db][library][collection]['3d']:
                self.__data[db][library][collection]['3d'][item_name] = dict()
            for d in data.keys():
                self.__data[db][library][collection]['3d'][item_name][d] = data[d]

    def delete_item(self, item_name, collection, library, db):
        if not self.__check_exits(item_name, collection, None, library, db, 'standard'):
            raise Exception(
                f'item {item_name} does not exist in collection {collection}, library {library}, user database {db}')
        self.__data[db][library][collection]['standard'].pop(item_name)

    def delete_item_3d(self, item_name, collection, library, db, dates):
        if dates is not None:
            raise Exception('deleting individual dates from 3d items not yet supported for in memory datastore')
        if not self.__check_exits(item_name, collection, None, library, db, '3d'):
            raise Exception(
                f'item {item_name} does not exist in collection {collection}, library {library}, user database {db}')
        self.__data[db][library][collection]['3d'].pop(item_name)

    def list_items_3d(self, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        self.__init_if_not_present(collection, library, db)
        return list(self.__data[db][library][collection]['3d'].keys())

    def list_items(self, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        self.__init_if_not_present(collection, library, db)
        return list(self.__data[db][library][collection]['standard'].keys())

    def list_item_3d_dates(self, item_name, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        if not self.__check_exits(item_name, collection, snapshot, library, db, '3d'):
            raise Exception(
                f'item {item_name} does not exist in collection {collection}, library {library}, user database {db}')
        return list(self.__data[db][library][collection]['3d'][item_name].keys())

    def list_libraries(self, db):
        if db not in self.__data:
            raise Exception(f'user database {db} does not exist')
        return list(self.__data[db].keys())

    def delete_library(self, library, db):
        if db not in self.__data:
            raise Exception(f'user database {db} does not exist')
        if library not in self.__data[db]:
            raise Exception(f'library {library} does not exist in user database {db}')
        self.__data[db].pop(library)

    def list_collections(self, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError('snapshots not supported for memory datastore')
        self.__init_if_not_present(None, library, db)
        return list(self.__data[db][library].keys())

    def list_snapshots(self, library, db):
        raise NotImplementedError('snapshots not supported for memory datastore')

    def create_snapshot(self, snapshot_name, collection, library, db):
        raise NotImplementedError('snapshots not supported for memory datastore')

    def delete_snapshot(self, snapshot_name, library, db):
        raise NotImplementedError('snapshots not supported for memory datastore')

    def list_versions(self, item_name, collection, library, db):
        raise NotImplementedError('Versions not supported for the memory datastore')

    def __init_if_not_present(self, collection, library, db):
        if db not in self.__data:
            self.__data[db] = dict()
        if library not in self.__data[db]:
            self.__data[db][library] = dict()
        if collection not in self.__data[db][library] and collection is not None:
            self.__data[db][library][collection] = {'standard': dict(), '3d': dict()}

    def __check_exits(self, item_name, collection, snapshot, library, db, item_type):
        if db not in self.__data:
            return False
        if library not in self.__data[db]:
            return False
        if collection not in self.__data[db][library]:
            return False
        if item_name is not None and item_name not in self.__data[db][library][collection][item_type]:
            return False
        return True
