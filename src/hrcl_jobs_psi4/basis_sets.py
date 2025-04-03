# List of supported simple basis sets
simple_basis_sets = [
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
    "jun-cc-pvdz",
    "jun-cc-pvtz",
    "cc-pvdz",
    "cc-pvtz",
    "may-cc-pvtz",
]

basis_dict = {
    "adz": "aug-cc-pvdz",
    "atz": "aug-cc-pvtz",
    "qz": "cc-pvqz",
    "aqz": "aug-cc-pvqz",
    "jdz": "jun-cc-pvdz",
    "jtz": "jun-cc-pvtz",
    "dz": "cc-pvdz",
    "tz": "cc-pvtz",
    "mtz": "may-cc-pvtz",
    "adtz": "aug-cc-pv[dt]z",
    "dtz": "cc-pv[dt]z",
    "atqz": "aug-cc-pv[tq]z",
    "tqz": "cc-pv[tq]z",
}

def get_basis_set(basis_str: str) -> str:
    """
    Converts abbreviated basis set names to their full form.

    Args:
        basis_str (str): Abbreviated or full basis set name

    Returns:
        str: Full basis set name recognized by Psi4

    Raises:
        ValueError: If the basis set abbreviation is not recognized

    Examples:
        >>> get_basis_set("adz")
        "aug-cc-pvdz"
        >>> get_basis_set("tz")
        "cc-pvtz"
    """
    basis_str = basis_str.lower()
    if basis_str in simple_basis_sets:
        return basis_str
    return basis_dict[basis_str]
