from abc import ABC, abstractmethod
from typing import Any


class BaseExperiment(ABC):
    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def results(self):
        pass

    @abstractmethod
    def save_results(self, **result: Any) -> Any:
        pass
