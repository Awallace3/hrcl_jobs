import os
import numpy as np
from . import jobspec
from periodictable import elements
from qm_tools_aw import tools
import qcelemental as qcel
from pprint import pprint as pp


def run_orca_input(job_sub_dir, inp_fn) -> float:
    def_dir = os.getcwd()
    os.chdir(job_sub_dir)
    os.system(f"`which orca` {inp_fn}.inp > {inp_fn}.out")
    with open(f"{inp_fn}.out", "r") as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                e = float(line.split()[4])
                break
    os.chdir(def_dir)
    return e


def orca_dlpno_ccsd_ie(js: jobspec.dlpno_ie_js, omp_threads=1):
    el_dc = tools.create_el_num_to_symbol()
    energies = []
    conv = qcel.constants.conversion_factor("hartree", "kcal/mol")
    for lt in js.level_theory:
        lt, TCutPNO, TCutPairs, TCutMKN = lt
        lt_es = []
        ZA = np.array([el_dc[i] for i in js.ZA])
        ZB = np.array([el_dc[i] for i in js.ZB])
        Z = np.concatenate((js.ZA, js.ZB))
        R = np.concatenate((js.RA, js.RB))
        # Running monomer A
        ma = tools.print_cartesians_pos_carts(js.ZA, js.RA, only_results=True)
        mb = tools.print_cartesians_pos_carts(js.ZB, js.RB, only_results=True)
        dimer = tools.print_cartesians_pos_carts(Z, R, only_results=True)

        job_dir = f"data/{js.DB}/{js.DB}_{js.sys_ind}"
        os.makedirs(job_dir, exist_ok=True)
        labels = ["dimer", "mA", "mB"]
        geoms = [dimer, ma, mb]
        for l, c, g in zip(labels, js.charges, geoms):
            c = " ".join([str(i) for i in c])
            os.makedirs(f"{job_dir}/{l}", exist_ok=True)
            if omp_threads != 1:
                inp = f"""! {lt}
%PAL NPROCS {omp_threads} END
%mdci  TCutPNO    {TCutPNO:e2} # default 3.33e-7
       TCutPairs  {TCutPairs:e2}    # default 1e-4
       TCutMKN    {TCutMKN:e2}    # default 1e-3
       end
*xyz {c}
{g}*"""
            else:
                inp = f"""! {lt}
%mdci  TCutPNO    1e-8 # default 3.33e-7
       TCutPairs  1e-5    # default 1e-4
       TCutMKN    1e-3    # default 1e-3
       end
*xyz {c}
{g}*"""
            # print(inp)
            with open(f"{job_dir}/{l}/{l}.inp", "w") as f:
                f.write(inp)
            e = run_orca_input(f"{job_dir}/{l}", l)
            lt_es.append(e)
        ie = (lt_es[0] - lt_es[1] - lt_es[2])
        lt_es = [ie, *lt_es]
        print(lt_es)
        energies.append(np.array(lt_es))
    print(energies)
    return energies
