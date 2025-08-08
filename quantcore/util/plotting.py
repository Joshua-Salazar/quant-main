import os
import matplotlib
import platform

CHANGE_MATPLOTLIB_SETTINGS = not os.getenv("QCORE_DISABLE_MATPLOTLIB_SETTINGS")

if CHANGE_MATPLOTLIB_SETTINGS:
    if platform.system() == "Windows":
        matplotlib.use("Qt5Agg")
    else:
        if "DISPLAY" in os.environ:
            # matplotlib.use('TkAgg')
            matplotlib.use("Qt5Agg")
        else:
            matplotlib.use("Agg")

import seaborn as sns
from matplotlib import pyplot as plt

if CHANGE_MATPLOTLIB_SETTINGS:
    sns.set(rc={"figure.figsize": (16, 8)})
    for style in ["seaborn-whitegrid", "fast", "fivethirtyeight", "classic"]:
        if style in plt.style.available:
            plt.style.use(style)
            break


# Quick ideal/target plot of a portfolio post optimisation
def plot_ideal_against_target(ideal, target):
    pmax = max(ideal.max(), target.max())
    pmin = min(ideal.min(), target.min())
    plt.scatter(ideal, target, alpha=0.5)
    plt.plot([pmin, pmax], [pmin, pmax], "--")
    plt.xlabel("ideal")
    plt.ylabel("target")
