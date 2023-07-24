def get_basis_set(basis_str: str) -> str:
    if basis_str == "adz":
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
        raise ValueError(
            "Basis Set Abbreviation not recognized by hrcl_jobs_psi4.basis_sets.py"
        )
    return basis
