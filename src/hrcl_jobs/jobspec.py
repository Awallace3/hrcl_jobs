from dataclasses import dataclass
import numpy as np


@dataclass
class mp_js:
    id_label: int
    RA: np.array
    RB: np.array
    ZA: np.array
    ZB: np.array
    TQA: np.array
    TQB: np.array
    level_theory: str
    mem: str

@dataclass
class grimme_js:
    id_label: int
    geometry: np.array
    monAs: np.array
    monBs: np.array
    level_theory: [str]
    mem: str

@dataclass
class saptdft_js:
    id_label: int
    geometry: np.array
    monAs: np.array
    monBs: np.array
    charges: np.array
    level_theory: [str]
    mem: str
