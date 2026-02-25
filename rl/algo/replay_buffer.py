"""Simple replay buffer for SAC training."""

import random
from typing import Any, Dict, List


class ReplayBuffer:
    """Uniform replay buffer with fixed capacity."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: List[Dict[str, Any]] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Dict[str, Any]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.storage, batch_size)
