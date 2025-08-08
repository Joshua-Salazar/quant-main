from datetime import date
import pandas as pd


class FixingTable:
    def __init__(self, fixing_table: pd.DataFrame):
        self.fixing_table = fixing_table
        self.validate()
        self.fixing_table.set_index(["date", "underlying"], inplace=True)

    def validate(self):
        if not self.fixing_table.empty:
            target_cols = ["date", "underlying", "fixing"]
            if set(self.fixing_table.columns) != set(target_cols):
                raise Exception(f"Unexpected fixing table columns: {','.join(self.fixing_table.columns)}. "
                                f"Only support column names: {','.join(target_cols)}")

    def has_fixing(self, underlying: str, fixing_date: date):
        found = False
        if (fixing_date, underlying) in self.fixing_table.index:
            fixing = self.fixing_table.loc[(fixing_date, underlying)]
            found = not fixing.empty
        return found

    def get_fixing(self, underlying: str, fixing_date: date):
        if not self.has_fixing(underlying, fixing_date):
            raise Exception(f"Not found fixing for {underlying} on {fixing_date.strftime('%Y-%m-%d')}")

        fixing = self.fixing_table.loc[(fixing_date, underlying)]
        if fixing.shape[0] > 1:
            raise Exception(f"found multiple fixings ({fixing.shape[0]}) for {underlying} on {fixing_date.strftime('%Y-%m-%d')}")
        return fixing.fixing

    def add_fixing(self, underlying: str, fixing_date: date, fixing: float):
        if (fixing_date, underlying) in self.fixing_table.index:
            if self.fixing_table.loc[(fixing_date, underlying)].fixing != fixing:
                raise Exception(f"Found fixing for {underlying} on {fixing_date.strftime('%Y-%m-%d')}")
        else:
            self.fixing_table.loc[(fixing_date, underlying), "fixing"] = fixing

    def merge_fixing_table(self, fixing_table):
        other_fixing_table = fixing_table.fixing_table
        common_index = self.fixing_table.index.intersection(other_fixing_table.index)
        if not other_fixing_table.loc[common_index, :].equals(self.fixing_table.loc[common_index, :]):
            raise Exception(f"Found inconsistent fixing")
        else:
            fixing_table_to_add = other_fixing_table[~other_fixing_table.index.isin(common_index)]
            self.fixing_table = pd.concat([self.fixing_table, fixing_table_to_add])

    def clone(self):
        return FixingTable(self.fixing_table.reset_index())

    def __eq__(self, other):
        if not isinstance(other, FixingTable):
            return False
        return self.fixing_table.equals(other.fixing_table)

    def __hash__(self):
        return hash(tuple(self.fixing_table.itertuples(index=False, name=None)))
