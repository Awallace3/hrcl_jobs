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
