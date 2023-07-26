from dataclasses import dataclass, field
from typing import Optional, Callable, Sequence

import numpy as np

# To add readability in network.py and not have str everywhere
PoolName = str


@dataclass
class Pool:
    name: PoolName
    half_life: int
    src_pools: set = field(default_factory=set)
    dst_pools: set = field(default_factory=set)
    long_name: Optional[str] = None
    category: Optional[str] = None
    decay_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    carbon_stock: Optional[np.ndarray] = None
    substitution_factor: Optional[np.ndarray] = None
    substitution: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.long_name is None:
            self.long_name = self.name

@dataclass
class Flow:
    factor: np.ndarray | None
    delay: int = 0
    recycling: bool = False
    values: Optional[np.ndarray] = None
