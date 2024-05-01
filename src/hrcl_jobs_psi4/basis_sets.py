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

def get_basis_set(basis_str: str) -> str:
    basis_str = basis_str.lower()
    if basis_str in simple_basis_sets:
        return basis_str
    elif basis_str == "adz":
        basis = "aug-cc-pvdz"
    elif basis_str == "atz":
        basis = "aug-cc-pvtz"
    elif basis_str == "aqz":
        basis = "aug-cc-pvqz"
    elif basis_str == "jdz":
        basis = "jun-cc-pvdz"
    elif basis_str == "jtz":
        basis = "jun-cc-pvtz"
    elif basis_str == "dz":
        basis = "cc-pvdz"
    elif basis_str == "tz":
        basis = "cc-pvtz"
    elif basis_str == "mtz":
        basis = "may-cc-pvtz"
    else:
        print(f"Basis Set Abbreviation: {basis_str}")
        raise ValueError(
            "Basis Set Abbreviation not recognized by hrcl_jobs_psi4.basis_sets.py"
        )
    return basis
