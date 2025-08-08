import os
from typing import Optional

import pandas as pd

from ..data.localfilesystemstore import LocalFileSystemStore


class LocalParquetDataStore(LocalFileSystemStore):
    @classmethod
    def storage_format(cls) -> str:
        return "parquet"

    @classmethod
    def extension(cls) -> str:
        return "parquet"

    def __init__(
        self,
        credentials: Optional[dict] = None,
    ):
        super().__init__(credentials=credentials)
        self.storage_root = os.path.join(
            self.storage_root, self.__class__.storage_format()
        )
        os.makedirs(name=self.storage_root, exist_ok=True)

    def read_function(self, item_name: str, item_filepath: str) -> pd.DataFrame:
        data = pd.read_parquet(path=item_filepath)

        return data

    def write_function(
        self, df: pd.DataFrame, item_name: str, item_filepath: str
    ) -> None:
        df.to_parquet(path=item_filepath, compression="snappy")
