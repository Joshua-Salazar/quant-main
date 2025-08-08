import os
import pathlib
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = pathlib.Path(__file__).parent.resolve()


def get_root():
    root = os.path.join(ROOT, "..")
    return root


def get_temp_folder():
    return os.path.join(get_root(), "tmp")


def get_real_path(path_in):
    return path_in.replace("$ROOT", get_root())


def get_test_data_folder(subfolder: str=None, subsubfolder: str=None):

    test_data_folder = os.path.join(get_root(), "test_data")

    if subfolder:
        test_data_folder = os.path.join(test_data_folder, subfolder)
        if not os.path.exists(test_data_folder):
            os.mkdir(test_data_folder)
        if subsubfolder:
            test_data_folder = os.path.join(test_data_folder, subsubfolder)
            if not os.path.exists(test_data_folder):
                os.mkdir(test_data_folder)

    elif subsubfolder:
        raise Exception("missing subfolder name")

    return test_data_folder


def get_test_src_folder(subfolder: str=None, subsubfolder: str=None):

    test_data_folder = get_test_data_folder(subfolder, subsubfolder)
    test_src_folder = os.path.join(test_data_folder, "source")
    if not os.path.exists(test_src_folder):
        os.mkdir(test_src_folder)

    return test_src_folder


def assert_dataframe(actual_df: pd.DataFrame, target_file: str, rebase: bool, ignore_cols=[]):
    correct_file = target_file + ".correct"
    if os.path.isfile(correct_file):
        os.remove(correct_file)

    actual_df.to_csv(correct_file)

    actual_res_temp = pd.read_csv(correct_file).drop(columns=ignore_cols)
    if not os.path.isfile(target_file) and rebase:
        os.rename(correct_file, target_file)
        return True

    expected_res = pd.read_csv(target_file).drop(columns=ignore_cols)
    try:
        assert_frame_equal(actual_res_temp, expected_res)
        os.remove(correct_file)
    except:
        if rebase:
            os.remove(target_file)
            os.rename(correct_file, target_file)
        else:
            raise
