import numpy as np
from typing import List, Union

class SimulationResult:
    def __init__(self,
                 investor_numbers: Union[np.ndarray, List[float]],
                 potential_numbers: Union[np.ndarray, List[float]],
                 deinvestor_numbers: Union[np.ndarray, List[float]],
                 capital: Union[np.ndarray, List[float]],
                 dt: float):
        self.investor_numbers = np.array(investor_numbers)
        self.potential_numbers = np.array(potential_numbers)
        self.deinvestor_numbers = np.array(deinvestor_numbers)
        self.capital = np.array(capital)
        self.dt = dt