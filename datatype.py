from typing import List, Set, Dict, Tuple, Callable
from dataclasses import dataclass
import numpy as np

Point = int # using the number as the point

@dataclass(frozen=True)
class Agent:
    a: Point
    b: Point 

@dataclass
class TrSPInstance:
    agents: List[Agent]
    candidates: List[Point]
    k: int 
    walk_matrix: np.ndarray
    transit_matrix: np.ndarray