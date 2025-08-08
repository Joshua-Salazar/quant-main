import os
import pathlib


ROOT = pathlib.Path(__file__).parent.resolve()


def get_root():
    root = os.path.join(ROOT, "..")
    return root
