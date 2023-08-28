from abc import ABC, abstractmethod
from typing import Tuple


class BaseObjectParametrization(ABC):
    def __init__(self, object_type: str, max_size: float, min_size: float):
        self.object_type = object_type
        self.max_size = max_size
        self.min_size = min_size

    @abstractmethod
    def get_geometry(self, size_ratio: float, orientation: Tuple):
        pass


class BaseStructureParametrization(ABC):
    def __init__(self, name: str):
        self.structure_name = name

    @property
    @abstractmethod
    def bounds(self):
        pass

    @abstractmethod
    def get_geometry(self, params):
        pass
