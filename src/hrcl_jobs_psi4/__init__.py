# check is psi4 is installed
import sys
try:
    from . import psi4_inps
    from . import jobspec
    from . import basis_sets
    from . import methods
    def get_level_of_theory(level_of_theory_name: str):
        col_split = level_of_theory_name.split("_")
        basis_str = col_split[-1]
        basis = basis_sets.get_basis_set(basis_str)
        method = "_".join(col_split[:-1])
        method = methods.get_methods(method)
        return method, basis

    def get_parallel_functions(method):
        if method == "SAPT0":
            return jobspec.sapt0_js, jobspec.sapt0_js_headers, psi4_inps.run_sapt0_components
        else:
            print(f"Method {method} not implemented yet!")
            sys.exit(1)
except ModuleNotFoundError:
    pass

def get_col_check(method, basis_str):
    array_methods = ["sapt0", "sapt2+3(ccd)dmp2"]
    method_str = method.replace("(", "_LP_").replace(")", "_RP_").replace("+", "_plus_")
    col_check = f"{method_str}_{basis_str}"
    if method.lower() in array_methods:
        col_type = "array"
    else:
        col_type = "real"
    table_cols = {
        col_check: col_type,
    }
    return table_cols, col_check

def get_parallel_functions(method):
    if method.upper() == "SAPT0":
        return jobspec.sapt0_js, jobspec.sapt0_js_headers, psi4_inps.run_sapt0_components
    if method.upper() == "SAPT2+3(CCD)DMP2":
        return jobspec.sapt_js, jobspec.sapt_js_headers, psi4_inps.run_sapt0_components
    else:
        print(f"Method {method} not implemented yet!")
        sys.exit(1)
