import os
import pathlib
from functools import wraps
from datetime import datetime
import copy
import pytz

ROOT = pathlib.Path(__file__).parent.resolve()


def get_root():
    root = os.path.join(ROOT, "..")
    return root


def timer(func):
    @wraps(func)
    def timing(*args, **kwargs):
        if "timing" in kwargs and kwargs["timing"]:
            unit = kwargs["unit"] if "unit" in kwargs else "sec"
            start = datetime.now()
            try:
                return func(*args, **kwargs)
            finally:
                end = datetime.now()
                elapsed_time = count(start, end, unit)
                print(f"func {func.__name__} execution time: {elapsed_time} in {unit}")
        else:
            return func(*args, **kwargs)
    return timing


def count(start_time: datetime, end_time: datetime, unit_time: str):
    elapsed_time = (end_time - start_time).total_seconds()
    if unit_time == "msec":
        elapsed_time *= 1000
    if unit_time == "min":
        elapsed_time /= 60
    elif unit_time == "hour":
        elapsed_time /= 3600
    return round(elapsed_time, 2)


class Timer:
    def __init__(self, func_name: str = "", unit: str = "sec", timezone: pytz.timezone = None,
                 verbose: bool = True, save_log: bool = False, log_file: str = ""):
        self.func_name = func_name
        self.unit_time = unit
        self.timezone = timezone
        self.verbose = verbose
        self.start_time = None
        self.reset_time = None
        self.prefix_msg = f"func {self.func_name}" if self.func_name else "func"
        self.fmt = "%Y-%m-%d %H:%M:%S %Z%z"

        self.end_time = None
        self.elapsed_time = None
        self.save_log = save_log
        self.log_file = get_root() + "/tools/log.txt" if log_file == "" else log_file

    def get_now(self):
        return datetime.now() if self.timezone is None else datetime.now(tz=self.timezone)

    def start(self):
        self.start_time = self.get_now()
        self.reset_time = copy.deepcopy(self.start_time)
        if self.verbose:
            msg = f"{self.prefix_msg} start at {self.start_time.strftime(self.fmt)}"
            if self.save_log:
                with open(self.log_file, "a+") as f:
                    f.write(msg + "\n")
            else:
                print(msg)

    def reset_start(self, msg: str = ""):
        self.reset_time = self.get_now()
        if self.verbose:
            msg = f"{self.prefix_msg}: {msg} at {self.reset_time.strftime(self.fmt)}"
            if self.save_log:
                with open(self.log_file, "a+") as f:
                    f.write(msg + "\n")
            else:
                print(msg)

        self.reset_time = self.get_now()

    def reset(self, msg: str = ""):
        end_time = self.get_now()
        elapsed_time = count(self.reset_time, end_time, self.unit_time)
        if self.verbose:
            msg = f"{self.prefix_msg}: {msg} at {end_time.strftime(self.fmt)} and excution time:  {elapsed_time} {self.unit_time}"
            if self.save_log:
                with open(self.log_file, "a+") as f:
                    f.write(msg + "\n")
            else:
                print(msg)
        self.reset_time = self.get_now()

    def print(self, msg: str):
        if self.verbose:
            msg = f"{self.prefix_msg}: {msg} at {self.get_now().strftime(self.fmt)}"
            if self.save_log:
                with open(self.log_file, "a+") as f:
                    f.write(msg + "\n")
            else:
                print(msg)

    def end(self):
        self.end_time = self.get_now()
        self.elapsed_time = count(self.start_time, self.end_time, self.unit_time)
        if self.verbose:
            msg = f"{self.prefix_msg} end at {self.end_time.strftime(self.fmt)} and excution time:  {self.elapsed_time} {self.unit_time}"
            if self.save_log:
                with open(self.log_file, "a+") as f:
                    f.write(msg + "\n")
            else:
                print(msg)


@timer
def test(**kwargs):
    a = 0
    for i in range(1000000):
        a += i
    print("finished")


if __name__ == "__main__":
    test(timing=True, unit="sec")
