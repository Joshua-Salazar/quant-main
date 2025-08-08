from abc import ABC, abstractmethod


class IDataSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, data_request):
        pass
