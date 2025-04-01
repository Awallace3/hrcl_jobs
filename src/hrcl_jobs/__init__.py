import warnings
from . import sqlt
from . import serial
from . import jobspec
import_error = None
try:
    from . import pgsql
except ImportError:
    warnings.warn("Could not import all modules from hrcl_jobs.pgsql (install psycopg2)")
    import_error = f"{ImportError}"
    pass
try:
    from . import parallel
    from . import setup
    from . import examples
    from . import utils
except ImportError:
    # warnings.warn("Could not import all modules from hrcl_jobs.parallel (install mpi4py)")
    import_error = f"{ImportError}"
    pass

try:
    from . import dataset 
except ImportError:
    # warnings.warn("Could not import all modules from hrcl_jobs.dataset (install mpi4py and psi4)")
    import_error = f"{ImportError}"
    pass

