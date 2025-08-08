import inspect
import os
from datetime import datetime


class Logger(object):

    def __init__(self, loglevel=3, logpath=None):
        self.loglevel = loglevel
        self.logpath = logpath
        self.fhandle = None

        if self.logpath:
            try:
                self.fhandle = open(self.logpath, 'a+')
                self.fhandle.write("\n")
                self.fhandle.flush()
            except Exception as ex:
                self.error('fail to write to file: %s' % ex)

    def __del__(self):
        if self.fhandle is not None:
            self.fhandle.close()

    def getCaller(self):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        return "%s@%s|%s" % (calframe[3][1].split("/")[-1][:-3], calframe[3][3], calframe[2][3])

    def printout(self, input, level="NotSet"):
        buffer = "[%s][%s][%s] %s" % (str(datetime.now())[:-3], self.getCaller(), level, input)
        if self.fhandle is not None:
            self.fhandle.write(buffer + '\n')
            self.fhandle.flush()
        print(buffer)

    def debug(self, input):
        if self.loglevel >= 5: self.printout(input, "Debug")

    def log(self, input):
        if self.loglevel >= 4: self.printout(input, "Log")

    def info(self, input):
        if self.loglevel >= 3: self.printout(input, "Info")

    def warning(self, input):
        if self.loglevel >= 2: self.printout(input, "Warning")

    def error(self, input):
        if self.loglevel >= 1: self.printout(input, "ERROR")
