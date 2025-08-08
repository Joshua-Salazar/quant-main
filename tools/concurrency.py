from datetime import datetime
from threading import Thread, Condition, Lock, Event, RLock


class ThreadTimer(Thread):
    counter = 0

    def __init__(self, event, fcn, interval=1, name=None, immediate=False):
        super(ThreadTimer, self).__init__()
        if not name:
            self.name = "ThreadTimer-{}".format(ThreadTimer.counter)
            ThreadTimer.counter += 1
        else:
            self.name = name
        self.immediate = immediate
        self.stopped = event
        self.interval = interval
        self.last_called = None
        self.fcn = fcn
        self.daemon = True

    def run(self):
        if self.immediate:
            try:
                self.fcn()
            except:
                print("{} failed".format(self.fcn))
            finally:
                self.last_called = datetime.now()
                self.immediate = False

        while not self.stopped.wait(self.interval):
            try:
                self.fcn()
            except:
                print("{} failed".format(self.fcn))
            finally:
                self.last_called = datetime.now()

    def __str__(self):
        return "{}<every {:.4f} seconds>".format(self.name, self.interval)

    def __repr__(self):
        return "{}({!r},{!r},{!r},{!r})".format(self.__class__.__name__
                                                , self.stopped
                                                , self.fcn
                                                , self.interval
                                                , self.name)
