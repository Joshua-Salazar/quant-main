import copy
import json
import os
import pickle
import shutil
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Functionalities to do with snapshots are not implemented
_snapshots_not_implemented = "Snapshots not supported for local file system datastore !"


class UnsupportedStorageFormat(Exception):
    pass


class UnsupportedDataType(Exception):
    pass


class NoDataFoundException(Exception):
    pass


WriteableDataStorageType = Literal["series", "dataframe", "pyarrow_table"]
PickleDataStorageType = Literal["pickle"]
DataStorageType = Union[WriteableDataStorageType, PickleDataStorageType]


def _validate_transform_data(
    data: Union[pa.Table, pd.Series, pd.DataFrame, Any]
) -> Union[Tuple[Union[pa.Table, pd.DataFrame], WriteableDataStorageType], Tuple[Any, PickleDataStorageType]]:
    if isinstance(data, pd.Series):
        return data.to_frame(), "series"
    elif isinstance(data, pd.DataFrame):
        return data, "dataframe"
    elif isinstance(data, pa.Table):
        return data, "pyarrow_table"
    else:
        return data, "pickle"


def _create_empty_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(name=path)


def determine_version_sensitive_path(item_path: str, version: Optional[int]) -> str:
    # Get the versions
    versions = [int(elt) for elt in os.listdir(item_path)]
    if version is None:
        # If no version is supplied, by default we take the latest one
        latest_version = max(versions)
        item_path = os.path.join(item_path, str(latest_version))
    else:
        item_path = os.path.join(item_path, str(version))
    return item_path


def _clean_up_versions(item_path: str) -> None:
    max_nb_versions = 2
    existing_versions = os.listdir(item_path)
    existing_versions = [elt for elt in existing_versions]
    existing_versions = sorted(
        existing_versions,
        key=lambda x: int(x),
    )
    versions_to_remove = existing_versions[:-max_nb_versions]
    for v in versions_to_remove:
        shutil.rmtree(os.path.join(item_path, v))


class LocalFileSystemStore(ABC):
    """
    Hierarchy
            # db
            # library
            # collection
            # item

    """

    @classmethod
    def string_date_format(cls):
        return "%Y_%m_%d_%H_%M_%S"

    @classmethod
    @abstractmethod
    def storage_format(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def extension(cls) -> str:
        pass

    @property
    def connection_info(self):
        return self.storage_root

    def __init__(
        self,
        credentials: Optional[dict] = None,
    ):
        if credentials is not None:
            self.storage_root = credentials["storage_root"]
            try:
                assert Path(self.storage_root).is_absolute()
            except AssertionError:
                msg = f"'storage_root' must be an absolute path. Received {self.storage_root}"
                raise Exception(msg)
        else:
            self.storage_root = os.path.join(Path.home(), ".localfilesystemstore")

    def _get_folder_non_3d_items(self, db, library, collection):
        folder_all_non_3d_items = os.path.join(
            self.storage_root, db, library, collection, "non_3d"
        )

        return folder_all_non_3d_items

    def _get_folder_3d_items(self, db, library, collection):
        folder_all_3d_items = os.path.join(
            self.storage_root, db, library, collection, "3d"
        )

        return folder_all_3d_items

    # Reading methods
    @abstractmethod
    def read_function(self, item_name: str, item_filepath: str) -> pd.DataFrame:
        pass

    def read_item(self, item_name, collection, snapshot, library, db, version):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder in which to look for the item
        enclosing_folder = self._get_folder_non_3d_items(
            db=db, library=library, collection=collection
        )
        # Create the path
        item_path = os.path.join(enclosing_folder, item_name)

        try:
            item_path = determine_version_sensitive_path(
                item_path=item_path, version=version
            )

            # Reading the metadata
            item_metadata_path = os.path.join(item_path, "metadata.json")
            with open(item_metadata_path, "r") as f:
                metadata = json.load(f)
            # Reading the data
            if metadata["type"] == "pickle":
                item_data_path = os.path.join(item_path, "data.pkl")
                with open(item_data_path, "rb") as f:
                    data = pickle.load(f)
            elif metadata["type"] == "pyarrow_table":
                item_data_path = os.path.join(item_path, "data.parquet")
                data = pq.read_table(source=item_data_path)
            else:
                item_data_path = os.path.join(item_path, f"data.{self.extension()}")
                data = self.read_function(
                    item_name=item_name, item_filepath=item_data_path
                )
                if metadata["type"] == "series":
                    data = data.squeeze()
        except FileNotFoundError:
            raise NoDataFoundException(
                f"No data found for {collection}@{item_name} in library local_file_system_{db}.{library}"
            )

        return data

    def read_item_3d(self, item_name, collection, snapshot, library, db, dates):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder in which the item was stored
        folder_all_3d_items = self._get_folder_3d_items(
            db=db, library=library, collection=collection
        )
        folder_3d_item = os.path.join(folder_all_3d_items, item_name)

        res = dict()
        for date in dates:
            date_str = date.strftime(self.__class__.string_date_format())
            file = f"{date_str}.{self.__class__.extension()}"
            data = self.read_function(
                item_name=date_str, item_filepath=os.path.join(folder_3d_item, file)
            )
            res[date] = data

        return res

    # Writing methods
    @abstractmethod
    def write_function(
        self, df: pd.DataFrame, item_name: str, item_filepath: str
    ) -> None:
        pass

    def write_item(self, data, item_name, collection, library, db):
        # Check that the type of the data is one supported
        to_save, data_type = _validate_transform_data(data=data)

        # Get the location of the folder in which to store
        enclosing_folder = self._get_folder_non_3d_items(
            db=db, library=library, collection=collection
        )

        # Create the folder if it does not exist
        os.makedirs(name=enclosing_folder, exist_ok=True)

        # Create the temporary path for storing the item (metadata and data)
        temp_item_path = os.path.join(self.storage_root, ".temporary_files", item_name)
        _create_empty_folder(path=temp_item_path)

        # Create the metadata
        metadata = dict(type=data_type)

        # Save the metadata
        with open(os.path.join(temp_item_path, "metadata.json"), "w") as output:
            json.dump(metadata, output)

        # Save the data
        if metadata["type"] == "pickle":
            with open(os.path.join(temp_item_path, "data.pkl"), "wb") as output:
                pickle.dump(to_save, output)
        elif metadata["type"] == "pyarrow_table":
            pq.write_table(table=to_save, where=os.path.join(temp_item_path, "data.parquet"))
        else:
            self.write_function(
                df=to_save,
                item_name=item_name,
                item_filepath=os.path.join(temp_item_path, f"data.{self.extension()}"),
            )

        # Make sure the final destination folder exists
        item_path = os.path.join(enclosing_folder, item_name)
        os.makedirs(item_path, exist_ok=True)
        epoch_time = str(time.time_ns())
        versioned_item_path = os.path.join(item_path, epoch_time)
        os.makedirs(versioned_item_path, exist_ok=False)

        # Move the data and metadata in atomic fashion
        os.rename(src=temp_item_path, dst=versioned_item_path)

        # Handle versions which we want to delete as too old
        _clean_up_versions(item_path=item_path)

    def write_item_3d(self, data, item_name, collection, library, db):
        # Get the location of the folder in which to store
        folder_all_3d_items = self._get_folder_3d_items(
            db=db, library=library, collection=collection
        )
        # Create the folder if it does not exist
        os.makedirs(name=folder_all_3d_items, exist_ok=True)

        # Temporary local folder to make writing safer
        temp_folder = os.path.join(Path.home(), ".temporary_files", "3d", item_name)
        _create_empty_folder(path=temp_folder)

        for date, data_on_date in data.items():
            # A Series is first transformed to dataframe
            if isinstance(data_on_date, pd.Series) or isinstance(
                data_on_date, pd.DataFrame
            ):
                date_str = date.strftime(self.__class__.string_date_format())
                temporary_filepath = os.path.join(
                    temp_folder, f"{date_str}.{self.__class__.extension()}"
                )
            else:
                raise UnsupportedDataType(
                    f"Data of type {type(data_on_date)}. Not supported. Must be pandas Series or DataFrame."
                )

            self.write_function(
                df=data_on_date, item_name=date_str, item_filepath=temporary_filepath
            )

        # Move the 3d item to its final destination
        folder_3d_item = os.path.join(folder_all_3d_items, item_name)
        _create_empty_folder(path=folder_3d_item)
        os.rename(src=temp_folder, dst=folder_3d_item)

    # Deletion methods
    def delete_item(self, item_name, collection, library, db):
        # Get the location of the folder in which to look for the item
        enclosing_folder = self._get_folder_non_3d_items(
            db=db, library=library, collection=collection
        )
        # Create the filepath
        item_path = os.path.join(enclosing_folder, f"{item_name}")

        shutil.rmtree(path=item_path)

    def delete_item_3d(self, item_name, collection, library, db, dates):
        if dates is not None:
            raise Exception('deleting individual dates from 3d items not yet supported for in memory datastore')
        # Get the location of the folder in which to look for the item
        folder_all_3d_items = self._get_folder_3d_items(
            db=db, library=library, collection=collection
        )
        # Create the path to the folder containing the 3d object
        folder_3d_item = os.path.join(folder_all_3d_items, item_name)

        # Delete the 3d item data
        shutil.rmtree(folder_3d_item)

    def delete_library(self, library, db):
        # Get the location of the folder for the library
        library_folder = os.path.join(self.storage_root, db, library)

        # Delete the library and all its contents
        shutil.rmtree(library_folder)

    def delete_collection(self, collection, library, db):
        # Get the location of the folder for the library
        collection_folder = os.path.join(self.storage_root, db, library, collection)

        # Delete the library and all its contents
        shutil.rmtree(collection_folder)

    # List methods
    def list_items(self, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder in which to look for the item
        enclosing_folder = self._get_folder_non_3d_items(
            db=db, library=library, collection=collection
        )

        try:
            # Get the list of contents but ignore paths that don't have metadata.json
            # files and are therefore not written by the datastore
            contents = list(
                set(
                    p.parent.parent.name
                    for p in Path(enclosing_folder).glob("*/*/metadata.json")
                )
            )
        except FileNotFoundError:
            contents = []

        return contents

    def list_items_3d(self, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder that contains all 3d items
        folder_all_3d_items = self._get_folder_3d_items(
            db=db, library=library, collection=collection
        )

        try:
            # Get the folders (they correspond to 3d objects)
            contents = os.listdir(folder_all_3d_items)
            contents = [
                elt
                for elt in contents
                if os.path.isdir(os.path.join(folder_all_3d_items, elt))
            ]
        except FileNotFoundError:
            contents = []

        return contents

    def list_item_3d_dates(self, item_name, collection, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder that contains all 3d items
        folder_all_3d_items = self._get_folder_3d_items(
            db=db, library=library, collection=collection
        )

        # Create the folder if it does not exist
        os.makedirs(name=folder_all_3d_items, exist_ok=True)

        path_3d_item = os.path.join(folder_all_3d_items, item_name)

        try:
            contents_3d_item_folder = os.listdir(path_3d_item)
            res = []
            for file in contents_3d_item_folder:
                date_str = file.split(".")[0]
                date = datetime.strptime(date_str, self.__class__.string_date_format())
                res.append(date)
            res = sorted(res)
        except FileNotFoundError:
            res = []

        return res

    def list_libraries(self, db):
        # Get the location of the folder in which to look for the item
        enclosing_folder = os.path.join(self.storage_root, db)

        try:
            # Get the list of contents
            contents = os.listdir(enclosing_folder)
        except FileNotFoundError:
            contents = []

        return contents

    def list_collections(self, snapshot, library, db):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder in which to look for the item
        enclosing_folder = os.path.join(self.storage_root, db, library)

        try:
            # Get the list of contents
            contents = os.listdir(enclosing_folder)
        except FileNotFoundError:
            contents = []

        return contents

    # Not implemented
    def stat_item(self, item_name, collection, library, db):
        raise NotImplementedError("Method 'stat_item' has not been implemented yet.")

    def update_item(self, data, item_name, collection, library, db, allow_overwrite):
        raise NotImplementedError("Method 'update_item' has not been implemented.")

    # TODO : snapshots will be handled in V2
    def list_snapshots(self, library, db):
        raise NotImplementedError("Snapshots are not handled at present.")

    def create_snapshot(self, snapshot_name, collection, library, db):
        raise NotImplementedError("Snapshots are not handled at present.")

    def delete_snapshot(self, snapshot_name, library, db):
        raise NotImplementedError("Snapshots are not handled at present.")

    def exists(self, item_name, collection, snapshot, library, db, version):
        if snapshot is not None:
            raise NotImplementedError(
                "LocalFileSystemStore does not support snapshots !"
            )

        # Get the location of the folder in which to look for the item
        enclosing_folder = self._get_folder_non_3d_items(
            db=db, library=library, collection=collection
        )
        # Create the path
        item_path = os.path.join(enclosing_folder, item_name)

        if os.path.exists(item_path):
            versions = os.listdir(item_path)

            if version is None:
                return len(versions) > 0
            else:
                return str(version) in versions
        else:
            return False
