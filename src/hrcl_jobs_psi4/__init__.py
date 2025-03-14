# check is psi4 is installed
import sys

try:
    from . import psi4_inps
    from . import jobspec
    from . import basis_sets
    from . import methods

    def get_level_of_theory(level_of_theory_name: str) -> tuple:
        """
        Parses a level of theory string into method and basis set components.
        
        Args:
            level_of_theory_name (str): String containing method and basis set information
                                        separated by an underscore (e.g., "SAPT0_adz")
        
        Returns:
            tuple: (method, basis) where method is the computational method string and 
                  basis is the basis set string
        """
        col_split = level_of_theory_name.split("_")
        basis_str = col_split[-1]
        basis = basis_sets.get_basis_set(basis_str)
        method = "_".join(col_split[:-1])
        method = methods.get_methods(method)
        return method, basis

    def get_parallel_functions(method: str) -> tuple:
        """
        Returns the appropriate jobspec functions for a given method.
        
        Args:
            method (str): The computational method name (e.g., "SAPT0")
        
        Returns:
            tuple: (jobspec_function, header_function, run_function) where:
                  - jobspec_function is the dataclass for job specification
                  - header_function provides column headers
                  - run_function is the function that executes the calculation
        
        Raises:
            SystemExit: If the method is not implemented
        """
        if method == "SAPT0":
            return (
                jobspec.sapt0_js,
                jobspec.sapt0_js_headers,
                psi4_inps.run_sapt0_components,
            )
        else:
            print(f"Method {method} not implemented yet!")
            sys.exit(1)

except ModuleNotFoundError as e:
    print(e)
    pass


def get_col_check(method: str, basis_str: str) -> tuple:
    """
    Determines the column type and name for database storage of results.
    
    Args:
        method (str): The computational method name
        basis_str (str): The basis set name or abbreviation
    
    Returns:
        tuple: (table_cols, col_check) where:
              - table_cols is a dictionary mapping column names to their types
              - col_check is the column name string
    """
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


def get_parallel_functions(method: str) -> tuple:
    """
    Returns the appropriate jobspec functions for a given method.
    
    Args:
        method (str): The computational method name (e.g., "SAPT0", "SAPT2+3(CCD)DMP2")
    
    Returns:
        tuple: (jobspec_function, header_function, run_function) where:
              - jobspec_function is the dataclass for job specification
              - header_function provides column headers
              - run_function is the function that executes the calculation
    
    Raises:
        SystemExit: If the method is not implemented
    """
    if method.upper() == "SAPT0":
        return (
            jobspec.sapt0_js,
            jobspec.sapt0_js_headers,
            psi4_inps.run_sapt0_components,
        )
    if method.upper() == "SAPT2+3(CCD)DMP2":
        return jobspec.sapt_js, jobspec.sapt_js_headers, psi4_inps.run_sapt0_components
    else:
        print(f"Method {method} not implemented yet!")
        sys.exit(1)
