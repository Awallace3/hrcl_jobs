from . import basis_sets

# List of supported simple methods
simple_methods = [
    "hf",
    "mp2",
    "ccsd",
    "ccsd(t)",
    "sapt0",
    "pbe0",
    "pbe",
    "sapt2+3(ccd)dmp2",
    "sapt(dft)",
]


def get_methods(in_method: str) -> str:
    """
    Returns the full name of the computational method from its shortened form.
    
    Args:
        in_method (str): Shortened or full method name
    
    Returns:
        str: Full method name recognized by Psi4
    
    Raises:
        ValueError: If the method name is not recognized
    
    Examples:
        >>> get_methods("sapt_dft")
        "SAPT(DFT)"
        >>> get_methods("b3lyp-d3bj")
        "b3lyp-d3bj"
    """
    if in_method.lower() in simple_methods:
        return in_method.lower()
    if "MBIS" in in_method:
        method = in_method.split("_")[1]
    # convert to dictionary
    method_dict = {
            "hf": "HF",
            "mp2": "MP2",
            "ccsd": "CCSD",
            "ccsd(t)": "CCSD(T)",
            "sapt0": "SAPT0",
            "pbe0": "PBE0",
            "pbe": "PBE",
            "sapt2+3(ccd)dmp2": "SAPT2+3(CCD)DMP2",
            "sapt(dft)": "SAPT(DFT)",
            "wb97x": "wb97x",
            "b97-0": "B97-0",
            "b97-1": "B97-1",
            "b2plyp": "b2plyp",
            "b3lyp": "b3lyp",
            "b3lyp-d3bj": "b3lyp-d3bj",
            "pbe-d3bj": "pbe-d3bj",
            "pbeh3c": "pbeh3c",
            "ccsd(t)cbs": "ccsd(t)cbs",
    }
    method = method_dict.get(in_method.lower())
    return method

def get_method_basis(in_method: str) -> tuple:
    """
    Parses a string containing method and basis set separated by "/" into components.
    
    Args:
        in_method (str): String in format "method/basis_set" or just "method"
    
    Returns:
        tuple: (method, basis) where method is the computational method string and
              basis is the basis set string or None if no basis set was specified
    
    Examples:
        >>> get_method_basis("mp2/adz")
        ("mp2", "aug-cc-pvdz")
        >>> get_method_basis("b3lyp")
        ("b3lyp", None)
    """
    if "/" in in_method:
        method, basis = in_method.split("/")
        method = get_methods(method)
        basis = basis_sets.get_basis_set(basis)
    else:
        method = get_methods(in_method)
        basis = None
    return method, basis
