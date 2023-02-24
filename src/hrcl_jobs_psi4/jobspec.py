from dataclasses import dataclass
import numpy as np

"""
All dataclass_js should have id_label for ms_sl() usage to update sql db
correctly
"""


@dataclass
class example_js:
    id_label: int
    val: float


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
class mp_mon_js:
    id_label: int
    R: np.array
    Z: np.array
    TQ: np.array
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
