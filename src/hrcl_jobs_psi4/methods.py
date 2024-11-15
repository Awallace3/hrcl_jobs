from . import basis_sets

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
    """Return the full name of the method from the shortened name."""
    if in_method.lower() in simple_methods:
        return in_method.lower()
    if in_method.lower() == "sapt_dft":
        method = "SAPT(DFT)"
    elif in_method.lower() == "wb97x":
        method = "wb97x"
    elif in_method.lower() == "b97-0":
        method = "B97-0"
    elif in_method.lower() == "b97-1":
        method = "B97-1"
    elif in_method.lower() == "b2plyp":
        method = "b2plyp"
    elif in_method.lower() == "b3lyp":
        method = "b3lyp"
    elif in_method.lower() == "b3lyp-d3bj":
        method = "b3lyp-d3bj"
    elif in_method.lower() == "pbe-d3bj":
        method = "pbe-d3bj"
    elif in_method.lower() == "pbeh3c":
        method = "pbeh3c"
    elif "MBIS" in in_method:
        method = in_method.split("_")[1]
    elif "MP2" in in_method.upper():
        method = 'MP2'
    elif in_method.lower() == "ccsd(t)":
        method = "ccsd"
    elif in_method.lower() == "ccsd_t_ie":
        method = "ccsd(t)"
    elif in_method.lower() == "ccsd(t)cbs":
        # untested, will need to be with aDZ basis set
        method = "mp2/aug-cc-pV[TQ]Z + d:ccsd(t)"
    else:
        print(f"Method Shortened: {in_method}")
        raise ValueError("in_method not recognized by hrcl_jobs_psi4.methos.py")
    return method

def get_method_basis(in_method: str) -> str:
    if "/" in in_method:
        method, basis = in_method.split("/")
        method = get_methods(method)
        basis = basis_sets.get_basis_set(basis)
    else:
        method = get_methods(in_method)
        basis = None
    return method, basis
