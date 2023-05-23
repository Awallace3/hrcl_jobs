import os
import numpy as np
from . import jobspec
from periodictable import elements
from qm_tools_aw import tools
import qcelemental as qcel
from pprint import pprint as pp
from mpi4py import MPI
import subprocess
from subprocess import PIPE


def run_orca_input(job_sub_dir, inp_fn) -> float:
    def_dir = os.getcwd()
    os.chdir(job_sub_dir)
    out = subprocess.run(["which", "orca"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
    out = out.stdout.strip()

    # MPI.COMM_SELF.Spawn(f"{out} {inp_fn}.inp > {inp_fn}.out", maxprocs=4)
    # os.system(f"conda activate orca_dlpno")
    os.system(f"`which orca` {inp_fn}.inp > {inp_fn}.out")
    e = None
    with open(f"{inp_fn}.out", "r") as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                e = float(line.split()[4])
                break
    os.chdir(def_dir)
    return e


def orca_dlpno_ccsd_ie(js: jobspec.dlpno_ie_js, openmpi_threads=1):
    el_dc = tools.create_el_num_to_symbol()
    energies = []
    conv = qcel.constants.conversion_factor("hartree", "kcal/mol")
    for lt in js.extra_info:
        lt,sub_dir, TCutPNO, TCutPairs, TCutMKN = lt
        lt_es = []
        ZA = np.array([el_dc[i] for i in js.ZA])
        ZB = np.array([el_dc[i] for i in js.ZB])
        Z = np.concatenate((js.ZA, js.ZB))
        R = np.concatenate((js.RA, js.RB))
        # Running monomer A
        ma = tools.print_cartesians_pos_carts(js.ZA, js.RA, only_results=True)
        mb = tools.print_cartesians_pos_carts(js.ZB, js.RB, only_results=True)
        dimer = tools.print_cartesians_pos_carts(Z, R, only_results=True)
        job_dir = f"data/{js.DB}/{js.DB}_{js.sys_ind}/{sub_dir}"
        os.makedirs(job_dir, exist_ok=True)
        labels = ["dimer", "mA", "mB"]
        geoms = [dimer, ma, mb]
        for l, c, g in zip(labels, js.charges, geoms):
            c = " ".join([str(i) for i in c])
            os.makedirs(f"{job_dir}/{l}", exist_ok=True)
            parralel = ""
            if openmpi_threads != 1 and openmpi_threads < 8:
                parralel = f"PAL{openmpi_threads}"
            elif openmpi_threads >= 8:
                parralel = f"\n%PAL NPROCS {openmpi_threads}END\n"
                # added scf block for converging s22_19

            inp = f"""! {lt} {parralel}
%mdci  TCutPNO    {TCutPNO:.3e} # default 3.33e-7
       TCutPairs  {TCutPairs:.3e}    # default 1e-4
       TCutMKN    {TCutMKN:.3e}    # default 1e-3
       end
%scf
MaxIter 150
directresetfreq 10 # Default value is 15. A value of 1 (very expensive) is sometimes required. A value between 1 and 15 may be more cost-effective.
end
! Bohrs
*xyz {c}
{g}*"""
            with open(f"{job_dir}/{l}/{l}.inp", "w") as f:
                f.write(inp)
            e = run_orca_input(f"{job_dir}/{l}", l)
            lt_es.append(e)
        ie = (lt_es[0] - lt_es[1] - lt_es[2])
        lt_es = [ie, *lt_es]
        energies.append(np.array(lt_es))
    return energies

def orca_dlpno_ccsd_ie_CP(js: jobspec.dlpno_ie_js, openmpi_threads=1):
    el_dc = tools.create_el_num_to_symbol()
    energies = []
    conv = qcel.constants.conversion_factor("hartree", "kcal/mol")
    for lt in js.extra_info:
        lt, sub_dir, TCutPNO, TCutPairs, TCutMKN = lt
        sub_dir += "_CP"
        lt_es = []
        ZA = np.array([el_dc[i] for i in js.ZA])
        ZB = np.array([el_dc[i] for i in js.ZB])
        Z = np.concatenate((js.ZA, js.ZB))
        R = np.concatenate((js.RA, js.RB))
        # Running monomer A
        ma = tools.print_cartesians_pos_carts_symbols(js.ZA, js.RA, only_results=True)
        mb = tools.print_cartesians_pos_carts_symbols(js.ZB, js.RB, only_results=True)
        ma_ghost = tools.print_cartesians_pos_carts_symbols(js.ZA, js.RA, only_results=True, el_attach=":")
        mb_ghost = tools.print_cartesians_pos_carts_symbols(js.ZB, js.RB, only_results=True, el_attach=":")
        dimer = tools.print_cartesians_pos_carts_symbols(Z, R, only_results=True)
        job_dir = f"data/{js.DB}/{js.DB}_{js.sys_ind}/{sub_dir}"
        os.makedirs(job_dir, exist_ok=True)
        labels = ["dimer", "mA", "mB"]
        geoms = [dimer, ma + mb_ghost, mb + ma_ghost]
        for l, c, g in zip(labels, js.charges, geoms):
            c = " ".join([str(i) for i in c])
            os.makedirs(f"{job_dir}/{l}", exist_ok=True)
            parralel = ""
            if openmpi_threads != 1 and openmpi_threads < 8:
                parralel = f"PAL{openmpi_threads}"
            elif openmpi_threads >= 8:
                parralel = f"\n%PAL NPROCS {openmpi_threads}END\n"
                # added scf block for converging s22_19

            inp = f"""! {lt} {parralel}
%mdci  TCutPNO    {TCutPNO:.3e} # default 3.33e-7
       TCutPairs  {TCutPairs:.3e}    # default 1e-4
       TCutMKN    {TCutMKN:.3e}    # default 1e-3
       end
%scf
MaxIter 150
directresetfreq 10 # Default value is 15. A value of 1 (very expensive) is sometimes required. A value between 1 and 15 may be more cost-effective.
end
! Bohrs
*xyz {c}
{g}*"""
            with open(f"{job_dir}/{l}/{l}.inp", "w") as f:
                f.write(inp)
            e = run_orca_input(f"{job_dir}/{l}", l)
            lt_es.append(e)
        ie = (lt_es[0] - lt_es[1] - lt_es[2])
        lt_es = [ie, *lt_es]
        energies.append(np.array(lt_es))
    return energies

