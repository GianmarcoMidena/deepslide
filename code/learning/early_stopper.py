from enum import Enum


class EarlyStopper:
    class Mode(Enum):
        MIN = 1
        MAX = 2

    def __init__(self, patience: int = 15, mode: Mode = Mode.MIN, min_delta: int = 0):
        self._patience = patience
        self._mode = mode
        self._min_delta = min_delta
        self._last_values = []

    def update(self, score) -> None:
        if len(self._last_values) > 0:
            best_score = self._last_values[0]
            if ((score < (best_score - self._min_delta)) and (self._mode == self.Mode.MIN)) or \
               ((score > (best_score + self._min_delta)) and (self._mode == self.Mode.MAX)):
                self._last_values = [score]
                return

        # otherwise: score is the first value or it is not an improvement w.r.t. the last value
        self._last_values.append(score)

    def is_stopping(self) -> bool:
        return (len(self._last_values) - 1) >= self._patience
