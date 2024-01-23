simple_methods = [
    "hf",
    "mp2",
    "ccsd",
    "ccsd(t)",
    "sapt0",
    "pbe0",
    "pbe",
    "sapt2+3(ccd)dmp2",
]


def get_methods(in_method: str) -> str:
    """Return the full name of the method from the shortened name."""
    if in_method.lower() in simple_methods:
        return in_method.lower()
    if in_method.lower() == "SAPT_DFT":
        method = "SAPT(DFT)"
    elif "MBIS" in in_method:
        method = in_method.split("_")[1]
    else:
        print(f"Method Shortened: {in_method}")
        raise ValueError("in_method not recognized by hrcl_jobs_psi4.methos.py")
    return method
