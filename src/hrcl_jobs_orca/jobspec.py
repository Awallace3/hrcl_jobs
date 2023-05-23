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
class dlpno_ie_js:
    id_label: int
    DB: str
    sys_ind: int
    RA: np.array
    RB: np.array
    ZA: np.array
    ZB: np.array
    charges: np.array
    extra_info: dict
    mem: str

def dlpno_ie_sql_headers() -> []:
        return [
            "id",
            "DB",
            "sys_ind",
            "RA",
            "RB",
            "ZA",
            "ZB",
            "charges",
        ]
