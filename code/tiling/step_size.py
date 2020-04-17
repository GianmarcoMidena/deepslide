class StepSize:
    def __init__(self, value: int, n_patches: int):
        self._value = value
        self._n_patches = n_patches

    @property
    def value(self) -> int:
        return self._value

    @property
    def n_patches(self) -> int:
        return self._n_patches

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value

    def __add__(self, other) -> int:
        if isinstance(other, self.__class__):
            return self.value + other.value
        elif isinstance(other, int):
            return self.value + other
        raise Exception(f"Attention: addition operation is not implemented between {self.__class__} "
                        f"and {type(other)} objects!")

    def __sub__(self, other) -> int:
        if isinstance(other, self.__class__):
            return self.value - other.value
        elif isinstance(other, int):
            return self.value - other
        raise Exception(f"Attention: addition operation is not implemented between {self.__class__} "
                        f"and {type(other)} objects!")
