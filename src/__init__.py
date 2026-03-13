import abc
from typing import TYPE_CHECKING, Type, TypeVar

if TYPE_CHECKING:
    from .core import VACore
    
T = TypeVar("T", bound="AI")


class AI(metaclass=abc.ABCMeta):
    def __init__(self):
        self._core = None

    @property
    def core(self):
        if self._core is None:
            raise ValueError("Core is not set for this AI instance.")
        return self._core

    @core.setter
    def core(self, core: "VACore"):
        self._core = core

    @abc.abstractmethod
    def check(self) -> bool: ...

    def find(self, ai_type: Type[T]) -> T:
        return self.core.find_ai(ai_type)