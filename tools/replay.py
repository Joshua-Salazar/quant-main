from functools import wraps
from datetime import datetime
import numpy as np
import pandas as pd
import json
import inspect


def replay(func):
    """
    auto generate replay script
    """
    @wraps(func)
    def replay_func(*args, **kwargs):
        if "replay" in kwargs and kwargs["replay"]:
            file = kwargs["replay_file"]
            with open(file, "w") as out:
                args_dict = {}
                args_names = [str(param) for param in inspect.signature(func).parameters]
                module_path = inspect.getmodule(func).__file__
                module_name = inspect.getmodulename(module_path)
                import_module = "from .." \
                                + module_path[module_path.find("/tools/")+1: module_path.find("/"+module_name+".py")
                                  ].replace("/", ".") \
                                + f" import {module_name}"
                import_modules = set()
                import_modules.add(import_module)
                call_line = f"{module_name}.{func.__name__}"
                call_line += "(\n"
                nspace = "    "
                for idx, arg in enumerate(args):
                    call_line += nspace
                    arg_name = args_names[idx]
                    if isinstance(arg, pd.DataFrame):
                        df = arg.reset_index(drop=True).to_json()
                        args_dict[arg_name] = df
                        call_line += f"{arg_name}=" + "pd.DataFrame.from_dict(" + args_dict[arg_name] + "),\n"
                        import_modules.add("import pandas as pd")
                    elif isinstance(arg, np.ndarray):
                        args_dict[arg_name] = arg.tolist()
                        call_line += f"{arg_name}=" + "np.asarragy(" + json.dumps(args_dict[arg_name]) + "),\n"
                        import_modules.add("import numpy as nps")
                    elif isinstance(arg, datetime):
                        args_dict[arg_name] = arg.isoformat()
                        call_line += f"{arg_name}=" + "datetime.fromisoformat(" + json.dumps(args_dict[arg_name]) + "),\n"
                        import_modules.add("from datetime import datetime")
                    else:
                        args_dict[arg_name] = arg
                        call_line += f"{arg_name}=" + json.dumps(args_dict[arg_name]) + ",\n"

                call_line += ")"
                if len(import_modules) > 0:
                    [out.write(import_module + "\n") for import_module in sorted(import_modules)]
                    out.write("\n")
                out.write(call_line)
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return replay_func


@replay
def test(dt, fund, existing_df, optimal_df, **kwargs):
    print(dt, fund)
    print(existing_df + optimal_df)


if __name__ == "__main__":
    dt = datetime.now()
    fund = "Convex"
    replay = True
    existing_df = pd.DataFrame([[1,2], [1, 1]], columns=["a", "b"])
    optimal_df = pd.DataFrame([[1,2], [0, 0]], columns=["a", "b"])
    test(dt, fund, existing_df, optimal_df, replay=True, replay_file="test_replay.py")
