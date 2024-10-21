from abc import *

class BaseNode(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.__identifier = None
        self.__model = None

    @property
    def id(self):
        return self.__identifier

    @id.setter
    def id(self, identifier):
        self.__identifier = identifier

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError