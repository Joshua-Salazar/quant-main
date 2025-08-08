
# imports
import sys

def get_arg_from_command_line(argument, default_value):
    for a in sys.argv:
        if a.startswith(argument):
            return a.replace(argument + '=','')
    return default_value