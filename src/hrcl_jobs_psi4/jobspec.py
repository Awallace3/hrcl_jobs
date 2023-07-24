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


def mp_js_headers():
    return [
        "id",
        "RA",
        "RB",
        "ZA",
        "ZB",
        "TQA",
        "TQB",
    ]


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
class sapt0_js:
    id_label: int
    geometry: np.array
    monAs: np.array
    monBs: np.array
    charges: np.array
    extra_info: {}
    mem: str

def sapt0_js_headers():
    return [
        "id",
        "Geometry",
        "monAs",
        "monBs",
        'charges',
    ]


@dataclass
class saptdft_js:
    id_label: int
    geometry: np.array
    monAs: np.array
    monBs: np.array
    charges: np.array
    level_theory: [str]
    mem: str

def saptdft_js_headers():
    return [
        "id_label",
        "geometry",
        "monAs",
        "monBs",
        "charges",
        "level_theory",
        "mem",
    ]


@dataclass
class saptdft_mon_grac_js:
    id_label: int
    geometry: np.array
    monNs: np.array
    charges: np.array
    extra_info: {}
    mem: str

def saptdft_mon_grac_js_headers(monNs="monAs"):
    return [
        "id",
        "Geometry",
        f"{monNs}",
        "charges",
    ]

def saptdft_mon_grac_js_headers(monNs="monAs"):
    return [
        "id",
        "geometry",
        f"{monNs}",
        "charges",
    ]

@dataclass
class psi4_dimer_js:
    id_label: int
    geometry: np.array
    monAs: np.array
    monBs: np.array
    extra_info: {}
    mem: str


def psi4_dimer_js_headers():
    return [
        "id",
        "Geometry",
        "monAs",
        "monBs",
    ]



