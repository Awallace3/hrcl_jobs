import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .jobspec import mp_js, grimme_js, saptdft_js, mp_mon_js
from . import jobspec
from periodictable import elements
import psi4
from psi4 import oeprop
from qcelemental import constants
import json
from qm_tools_aw import tools
import qcelemental as qcel
from pprint import pprint as pp

"""
/theoryfs2/ds/amwalla3/miniconda3/envs/psi4mpi4py_qcng/lib/python3.8/site-packages/psi4/driver/driver_nbody.py
"""


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_pt_dict():
    """
    create_pt_dict creates dictionary for string elements to atomic number.
    """
    el_dc = {}
    for el in elements:
        el_dc[el.number] = el.symbol
    return el_dc


def psi4_vac_mp(
    mem: str,
    level_theory: str,
    mol: str,
    max_iter: int = 200,
) -> np.array:
    """
    psi4_vac_mp
    """
    psi4.core.be_quiet()
    psi4.set_memory(mem)
    mol = psi4.geometry(mol)
    opts = {
        "scf_type": "df",
        "d_convergence": 10,
        "e_convergence": 10,
        "MAXITER": max_iter,
    }
    psi4.set_options(opts)
    e, wfn = psi4.energy(level_theory, molecule=mol, return_wfn=True)
    oeprop(wfn, "MBIS_VOLUME_RATIOS")

    charges = np.array(wfn.variable("MBIS CHARGES"))
    dipoles = np.array(wfn.variable("MBIS DIPOLES"))
    n_at = len(charges)

    widths = np.array(wfn.variable("MBIS VALENCE WIDTHS"))

    try:
        vol_ratio = np.array(wfn.variable("MBIS VOLUME RATIOS"))
    except KeyError:
        vol_ratio = np.ones(1)
    quad = np.array(wfn.variable("MBIS QUADRUPOLES"))
    # print(quad)
    quad = [q[np.triu_indices(3)] for q in quad]
    # print(quad)
    quadrupoles = np.array(quad)
    vac_multipoles = np.concatenate([charges, dipoles, quadrupoles], axis=1)
    return vac_multipoles, charges, widths, vol_ratio


def psi4_env_mp(
    mem: str,
    level_theory: str,
    mol: str,
    env_coords: np.array,
    env_charges: np.array,
    max_iter: int = 200,
    # TODO: add extra parameters to match make_env_inps.py
) -> np.array:
    """
    psi4_env_mp

    ext_p_block = f"env_coords = np.array({list_RBs}) / psi_bohr2angstroms\nenv_charges = np.expand_dims(np.array({list(vac_q_Bs[i])}), axis=-1)\nexternal_potentials = np.concatenate([env_charges, env_coords], axis=1).reshape((-1,4))"

    # convert environment to bohr
    """

    env_coords *= constants.conversion_factor("angstroms", "bohr")
    external_potential = np.concatenate([env_charges, env_coords], axis=1).reshape(
        (-1, 4)
    )
    psi4.core.be_quiet()
    psi4.set_memory(mem)
    mol = psi4.geometry(mol)
    opts = {
        "scf_type": "df",
        "d_convergence": 10,
        "e_convergence": 10,
        "MAXITER": max_iter,
    }
    psi4.set_options(opts)
    e, wfn = psi4.energy(
        level_theory,
        molecule=mol,
        return_wfn=True,
        external_potentials=external_potential,
    )
    oeprop(wfn, "MBIS_VOLUME_RATIOS")
    charges = np.array(wfn.variable("MBIS CHARGES"))
    dipoles = np.array(wfn.variable("MBIS DIPOLES"))

    widths = np.array(wfn.variable("MBIS VALENCE WIDTHS"))
    try:
        vol_ratio = np.array(wfn.variable("MBIS VOLUME RATIOS"))
    except KeyError:
        vol_ratio = np.ones(1)

    n_at = len(charges)
    quad = np.array(wfn.variable("MBIS QUADRUPOLES"))
    quad = [q[np.triu_indices(3)] for q in quad]
    quadrupoles = np.array(quad)
    vac_multipoles = np.concatenate([charges, dipoles, quadrupoles], axis=1)
    # psi4.core.clean()
    return vac_multipoles, widths, vol_ratio


def prep_mol_full(RA, RB, ZA, ZB, TQA, TQB, EA, EB):
    """prep_mol creates molecular geometry for psi4 form js vars"""
    R = np.concatenate((RA, RB), axis=0)
    Z = np.concatenate((ZA, ZB), axis=0)
    TQ = TQA + TQB
    E = np.concatenate((EA, EB), axis=0)
    n_atoms = len(R)  # + len(RBs[i])
    n_e = np.sum(Z) + TQ
    spin = float(abs(n_e % 2)) / 2
    mult = int(2 * spin + 1)
    mol = f"{int(TQ)} {mult}\n"
    for j, coord in enumerate(R):
        mol += f"{E[j]} {coord[0]} {coord[1]} {coord[2]}\n"
    mol += "symmetry c1\nno_reorient\nno_com"
    return mol


def prep_mol(R, Z, TQ, E, ending_tags=True):
    """prep_mol creates molecular geometry for psi4 form js vars"""
    n_atoms = len(R)  # + len(RBs[i])
    n_e = np.sum(Z) + TQ
    spin = float(abs(n_e % 2)) / 2
    mult = int(2 * spin + 1)
    mol = f"{int(TQ)} {mult}\n"
    for j, coord in enumerate(R):
        mol += f"{E[j]} {coord[0]} {coord[1]} {coord[2]}\n"
    if ending_tags:
        mol += "symmetry c1\nno_reorient\nno_com"
    return mol


def run_mp_js_job_only_dimer_mp_only(js: mp_js, el_dc=create_pt_dict()):
    """
    returns [
        vac_multipole_AB,
    ]
    """
    level_theory = js.level_theory[0]
    EA = np.array([el_dc[i] for i in js.ZA])
    EB = np.array([el_dc[i] for i in js.ZB])

    mol_d = prep_mol_full(js.RA, js.RB, js.ZA, js.ZB, js.TQA, js.TQB, EA, EB)
    vac_multipole_AB, charges_AB, vac_widths_AB, vac_vol_rat_AB = psi4_vac_mp(
        js.mem, level_theory, mol_d
    )
    output = [
        vac_multipole_AB,
    ]
    return output


def run_mp_js_job_only_dimer(js: mp_js, el_dc=create_pt_dict()):
    """
    returns [
        vac_multipole_AB,
        vac_widths_AB,
        vac_vol_rat_AB,
    ]
    """
    level_theory = js.level_theory[0]
    EA = np.array([el_dc[i] for i in js.ZA])
    EB = np.array([el_dc[i] for i in js.ZB])

    # TODO:  do for dimer together - figure out how to write the mol_AB
    mol_d = prep_mol_full(js.RA, js.RB, js.ZA, js.ZB, js.TQA, js.TQB, EA, EB)
    vac_multipole_AB, charges_AB, vac_widths_AB, vac_vol_rat_AB = psi4_vac_mp(
        js.mem, level_theory, mol_d
    )
    output = [
        vac_multipole_AB,
        vac_widths_AB,
        vac_vol_rat_AB,
    ]
    return output


def run_mp_mon_js(js: mp_mon_js, el_dc=create_pt_dict()):
    level_theory = js.level_theory[0]
    E = np.array([el_dc[i] for i in js.Z])
    mol_A = prep_mol(js.R, js.Z, js.TQ, E)
    vac_multipole_AB, charges_AB, vac_widths_AB, vac_vol_rat_AB = psi4_vac_mp(
        js.mem, level_theory, mol_A
    )
    output = [
        vac_multipole_AB,
    ]
    return output


def run_mp_js_job_vac_only(js: mp_js, el_dc=create_pt_dict()) -> np.array:
    """
    create_mp_js_job turns mp_js object into a psi4 job and runs it
    """
    level_theory = js.level_theory[0]
    EA = np.array([el_dc[i] for i in js.ZA])
    EB = np.array([el_dc[i] for i in js.ZB])

    mol_d = prep_mol_full(js.RA, js.RB, js.ZA, js.ZB, js.TQA, js.TQB, EA, EB)
    # print(f"mol_d: {mol_d}")
    vac_multipole_AB, charges_AB, vac_widths_AB, vac_vol_rat_AB = psi4_vac_mp(
        js.mem, level_theory, mol_d
    )

    mol_A = prep_mol(js.RA, js.ZA, js.TQA, EA)
    # print(f"mol_A: {mol_A}")
    vac_multipole_A, charges_A, vac_widths_A, vac_vol_rat_A = psi4_vac_mp(
        js.mem, level_theory, mol_A
    )

    mol_B = prep_mol(js.RB, js.ZB, js.TQB, EB)
    # print(f"mol_B: {mol_B}")
    vac_multipole_B, charges_B, vac_widths_B, vac_vol_rat_B = psi4_vac_mp(
        js.mem, level_theory, mol_B
    )

    env_multipole_A, env_widths_A, env_vol_rat_A = psi4_env_mp(
        js.mem, level_theory, mol_A, js.RB, charges_B
    )
    env_multipole_B, env_widths_B, env_vol_rat_B = psi4_env_mp(
        js.mem, level_theory, mol_B, js.RA, charges_A
    )
    output = [
        vac_multipole_A,
        vac_multipole_B,
        vac_multipole_AB,
    ]
    return output


def run_mp_js_job(js: mp_js, el_dc=create_pt_dict()) -> np.array:
    """
    create_mp_js_job turns mp_js object into a psi4 job and runs it
    """
    level_theory = js.level_theory[0]
    EA = np.array([el_dc[i] for i in js.ZA])
    EB = np.array([el_dc[i] for i in js.ZB])

    # TODO:  do for dimer together - figure out how to write the mol_AB
    mol_d = prep_mol_full(js.RA, js.RB, js.ZA, js.ZB, js.TQA, js.TQB, EA, EB)
    vac_multipole_AB, charges_AB, vac_widths_AB, vac_vol_rat_AB = psi4_vac_mp(
        js.mem, level_theory, mol_d
    )

    mol_A = prep_mol(js.RA, js.ZA, js.TQA, EA)
    vac_multipole_A, charges_A, vac_widths_A, vac_vol_rat_A = psi4_vac_mp(
        js.mem, level_theory, mol_A
    )

    mol_B = prep_mol(js.RB, js.ZB, js.TQB, EB)
    vac_multipole_B, charges_B, vac_widths_B, vac_vol_rat_B = psi4_vac_mp(
        js.mem, level_theory, mol_B
    )

    env_multipole_A, env_widths_A, env_vol_rat_A = psi4_env_mp(
        js.mem, level_theory, mol_A, js.RB, charges_B
    )
    env_multipole_B, env_widths_B, env_vol_rat_B = psi4_env_mp(
        js.mem, level_theory, mol_B, js.RA, charges_A
    )
    output = [
        vac_multipole_A,
        vac_multipole_B,
        env_multipole_A,
        env_multipole_B,
        vac_widths_A,
        vac_widths_B,
        vac_vol_rat_A,
        vac_vol_rat_B,
        vac_multipole_AB,
        vac_widths_AB,
        vac_vol_rat_AB,
    ]
    # output_columns=[
    #     "vac_multipole_A",
    #     "vac_multipole_B",
    #     "environment_multipole_A",
    #     "environment_multipole_B",
    #     "vac_widths_A",
    #     "vac_widths_B",
    #     "vac_vol_rat_A",
    #     "vac_vol_rat_B",
    #   "vac_multipole_AB",
    #   "vac_widths_AB",
    #   "vac_vol_rat_AB",
    # ],
    return output


def run_mp_js_job_test(js: mp_js, el_dc=create_pt_dict()) -> np.array:
    """
    create_mp_js_job turns mp_js object into a psi4 job and runs it
    """
    # collecting vacuum multipoles for monA and monB
    EA = np.array([el_dc[i] for i in js.ZA])
    # EB = np.array([el_dc[i] for i in js.ZB])
    return [np.full((5), js.rowid) for i in range(6)]


def run_mp_js_grimme(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    ies = run_psi4_sapt0(
        ma,
        mb,
        ppm=js.mem,
        level_theory=js.level_theory,
    )
    return ies


def run_mp_js_grimme_no_cp(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    ies = run_psi4_sapt0(ma, mb, ppm=js.mem, level_theory=js.level_theory, cp=False)
    return ies


def run_mp_js_grimme_fsapt(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)

    ies_parts = run_psi4_fsapt(ma, mb, ppm=js.mem, level_theory=js.level_theory)
    return ies_parts


def run_mp_js_grimme_no_df(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)

    ies = run_psi4_sapt0(
        # ma, mb, ppm=js.mem, level_theory=js.level_theory, cp=True, scf_type="direct"
        ma,
        mb,
        ppm=js.mem,
        level_theory=js.level_theory,
        cp=True,
        scf_type="pk"
        # ma, mb, ppm=js.mem, level_theory=js.level_theory, cp=True, scf_type="cd"
    )
    return ies


def run_psi4_fsapt(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["hf/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    d_convergence: int = 4,
    scf_type="df",
    rank: int = 2,
) -> []:
    """ """
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}\n--\n{B_cm[0]} {B_cm[1]}\n{B}"
    es_parts = []
    l = level_theory[0]
    mol = psi4.geometry(geom)
    psi4.set_memory(ppm)
    psi4.set_options(
        {
            "d_convergence": d_convergence,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": scf_type,
            "FISAPT_FSAPT_FILEPATH": "hello",
            "FISAPT_DO_FSAPT_DISP": True,
        }
    )
    e, wfn = psi4.energy(l, return_wfn=True)
    print(wfn.matrices())
    """
    # L4748
    # return ref_wfn
    return fisapt_wfn
    # /theoryfs2/ds/amwalla3/miniconda3/envs/psi4mpi4py/lib/python3.9/site-packages/psi4/driver/procrouting/proc.py
    """
    mat = wfn.matrices()
    elst = mat["Elst_AB"].np
    exch = mat["Exch_AB"].np
    indab_ab = mat["IndAB_AB"].np
    indba_ab = mat["IndBA_AB"].np
    disp = mat["Disp_AB"].np
    print("ELST:", elst)
    print("ELST type:", type(elst))
    ie = e
    print(psi4.core.variables())
    ie *= constants.conversion_factor("hartree", "kcal / mol")
    pieces = [ie, elst, exch, indab_ab, indba_ab, disp]
    es_parts.append(pieces)
    psi4.core.clean()
    return es_parts


def run_psi4_sapt0(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["hf/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    cp: bool = True,
    d_convergence: int = 4,
    scf_type="df",
) -> []:
    """ """
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}\n--\n{B_cm[0]} {B_cm[1]}\n{B}"
    es = []

    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "d_convergence": d_convergence,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": scf_type,
                # "cholesky_tolerance": 1e-6 # default about 1e-4
                # check psi4/src/read_options
            }
        )
        # psi4.core.be_quiet()
        if cp:
            e = psi4.energy(l, bsse_type="cp")
            ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        else:
            e = psi4.energy(l, bsse_type="nocp")
            ie = psi4.core.variable("NOCP-CORRECTED INTERACTION ENERGY")
        ie *= constants.conversion_factor("hartree", "kcal / mol")
        es.append(ie)
        psi4.core.clean()
    return es


def run_saptdft_components(js: jobspec.saptdft_js) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    geom = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    es = []
    for l in js.extra_info["level_theory"]:
        mol = psi4.geometry(geom)
        js.extra_info["options"]["sapt_dft_grac_shift_a"] = js.grac_shift_a
        js.extra_info["options"]["sapt_dft_grac_shift_b"] = js.grac_shift_b
        handle_hrcl_extra_info_options(js, l)

        try:
            e = psi4.energy(f"{l}")
            e *= constants.conversion_factor("hartree", "kcal / mol")
            ie = psi4.core.variable("SAPT(DFT) TOTAL ENERGY")
            ELST = psi4.core.variable("SAPT ELST ENERGY")
            EXCH = psi4.core.variable("SAPT EXCH ENERGY")
            IND = psi4.core.variable("SAPT IND ENERGY")
            DISP = psi4.core.variable("SAPT DISP ENERGY")
            mult = constants.conversion_factor("hartree", "kcal / mol")
            out_energies = np.array([ie, ELST, EXCH, IND, DISP]) * mult
        except Exception as e:
            print("Exception:", e)
            out_energies = None
        es.append(out_energies)
        handle_hrcl_psi4_cleanup(js, l)
    return es


def run_psi4_saptdft(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["hf/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    d_convergence: int = 4,
    scf_type="df",
    sapt_dft_grac_shift_a: float = 1e-16,
    sapt_dft_grac_shift_b: float = 1e-16,
) -> []:
    """ """
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}\n--\n{B_cm[0]} {B_cm[1]}\n{B}"
    # print(geom)
    es = []

    for l in level_theory:
        m, bs = l.split("/")
        # print(m, bs)
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "reference": "rhf",
                "basis": "aug-cc-pVDZ",
                "sapt_dft_grac_shift_a": sapt_dft_grac_shift_a,
                "sapt_dft_grac_shift_b": sapt_dft_grac_shift_b,
                "SAPT_DFT_FUNCTIONAL": m,
            }
        )
        # psi4.core.be_quiet()
        e = psi4.energy("sapt(dft)")
        print(e)
        total = psi4.core.variable("SAPT(DFT) TOTAL ENERGY")
        ELST = psi4.core.variable("SAPT ELST ENERGY")
        EXCH = psi4.core.variable("SAPT EXCH ENERGY")
        IND = psi4.core.variable("SAPT IND ENERGY")
        DISP = psi4.core.variable("SAPT DISP ENERGY")
        mult = constants.conversion_factor("hartree", "kcal / mol")
        out_energies = np.array([total, ELST, EXCH, IND, DISP]) * mult
        es.append(out_energies)
        psi4.core.clean()
    return es


def run_saptdft(js: saptdft_js) -> np.array:
    """
    run_saptdft computes scaling factor and uses to run SAPT-DFT
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    shift_a = run_dft_neutral_cation(
        ma, charges=js.charges[1], ppm=js.mem, level_theory=js.level_theory
    )
    # shift_b = run_dft_neutral_cation(
    #     mb, charges=js.charges[2], ppm=js.mem, level_theory=js.level_theory
    # )
    # shift_a.extend(shift_b)
    # ies = run_psi4_saptdft(
    #     ma,
    #     mb,
    #     ppm=js.mem,
    #     level_theory=js.level_theory,
    #     sapt_dft_grac_shift_a=shift_a[-1],
    #     sapt_dft_grac_shift_b=shift_b[-1],
    # )
    # shift_a.extend(ies)
    return shift_a


# def run_saptdft_grac_shift(js: jobspec.saptdft_mon_grac_js):
#     mn = []
#     for i in js.monNs:
#         mn.append(js.geometry[i, :])
#     mn = tools.np_carts_to_string(mn)
#     shift_n = run_dft_neutral_cation(mn,
#                                      charges=js.charges[1],
#                                      ppm=js.mem,
#                                      level_theory=js.level_theory)
#     return shift_n


def run_saptdft(js: saptdft_js) -> np.array:
    """
    run_saptdft computes scaling factor and uses to run SAPT-DFT
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    shift_a = run_dft_neutral_cation(
        ma, charges=js.charges[1], ppm=js.mem, level_theory=js.level_theory
    )
    shift_b = run_dft_neutral_cation(
        mb, charges=js.charges[2], ppm=js.mem, level_theory=js.level_theory
    )
    ies = run_psi4_saptdft(
        ma,
        mb,
        ppm=js.mem,
        level_theory=js.level_theory,
        sapt_dft_grac_shift_a=shift_a[-1],
        sapt_dft_grac_shift_b=shift_b[-1],
    )
    shift_a.extend(shift_b)
    shift_a.extend(ies)
    return shift_a


def run_psi4_sapt0(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["hf/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    cp: bool = True,
    d_convergence: int = 4,
    scf_type="df",
) -> []:
    """ """
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}\n--\n{B_cm[0]} {B_cm[1]}\n{B}"
    es = []

    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "d_convergence": d_convergence,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": scf_type,
                # "cholesky_tolerance": 1e-6 # default about 1e-4
                # check psi4/src/read_options
            }
        )
        # psi4.core.be_quiet()
        if cp:
            e = psi4.energy(l, bsse_type="cp")
            ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        else:
            e = psi4.energy(l, bsse_type="nocp")
            ie = psi4.core.variable("NOCP-CORRECTED INTERACTION ENERGY")
        ie *= constants.conversion_factor("hartree", "kcal / mol")
        es.append(ie)
        psi4.core.clean()
    return es


def run_saptdft_no_grac(js: saptdft_js) -> np.array:
    """
    run_saptdft_qtp computes scaling factor and uses to run SAPT-DFT
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    ies = run_psi4_saptdft(ma, mb, ppm=js.mem, level_theory=js.level_theory)
    return ies


def run_dft_neutral_cation(
    M, charges, ppm, level_theory, d_convergence="8"
) -> np.array:
    """
    run_dft_neutral_cation
    """
    geom_neutral = f"{charges[0]} {charges[1]}\n{M}"
    geom_cation = f"{charges[0]+1} {charges[1]+1}\n{M}"
    out = []
    psi4.core.be_quiet()
    for l in level_theory:
        m, bs = l.split("/")
        mol = psi4.geometry(geom_neutral)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "reference": "uhf",
            }
        )
        e_neutral, wfn_n = psi4.energy(l, return_wfn=True)
        e_neutral = e_neutral
        occ_neutral = wfn_n.epsilon_a_subset(basis="SO", subset="OCC").to_array(
            dense=True
        )
        HOMO = np.amax(occ_neutral)
        mol = psi4.geometry(geom_cation)
        e_cation, wfn = psi4.energy(l, return_wfn=True)
        grac = e_cation - e_neutral + HOMO
        print(f"{e_cation = } {l}\n{e_neutral = } {l}\n{HOMO = } {l}")
        print(f"{grac = }")
        out.append(e_neutral)
        out.append(e_cation)
        out.append(HOMO)
        out.append(grac)
        psi4.core.clean()
    return out


def run_dft_neutral_cation_qca(
    M, charges, ppm, level_theory, d_convergence="8"
) -> np.array:
    """
    run_dft_neutral_cation
    """
    geom_neutral = f"{charges[0]} {charges[1]}\n{M}"
    geom_cation = f"{charges[0]+1} {charges[1]+1}\n{M}"
    out = []
    psi4.core.be_quiet()
    for l in level_theory:
        m, bs = l.split("/")
        mol = psi4.geometry(geom_neutral)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "reference": "uhf",
            }
        )
        e_neutral, wfn_n = psi4.energy(l, return_wfn=True)
        e_neutral = e_neutral
        occ_neutral = wfn_n.epsilon_a_subset(basis="SO", subset="OCC").to_array(
            dense=True
        )
        HOMO = np.amax(occ_neutral)
        mol = psi4.geometry(geom_cation)
        e_cation, wfn = psi4.energy(l, return_wfn=True)
        grac = e_cation - e_neutral + HOMO
        print(f"{e_cation = } {l}\n{e_neutral = } {l}\n{HOMO = } {l}")
        print(f"{grac = }")
        out.append(e_neutral)
        out.append(e_cation)
        out.append(HOMO)
        out.append(grac)
        psi4.core.clean()
    return out


def run_mp_js_dimer_energy(js: mp_js) -> []:
    """
    run_bsse_js runs js for bsse energies
    """
    el_dc = create_pt_dict()
    EA = np.array([el_dc[i] for i in js.ZA])
    EB = np.array([el_dc[i] for i in js.ZB])
    mol_A = prep_mol(js.RA, js.ZA, js.TQA, EA, ending_tags=False)
    mol_B = prep_mol(js.RB, js.ZB, js.TQB, EB, ending_tags=False)
    es = run_psi4_dimer_energy(mol_A, mol_B, js.mem, js.level_theory)
    return es


def run_psi4_dimer_energy(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["sapt0/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    d_convergence: int = 4,
    scf_type="df",
) -> []:
    """ """
    psi4.core.be_quiet()
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A}\n--\n{B}"
    es = []
    mult = constants.conversion_factor("hartree", "kcal / mol")
    for l in level_theory:
        m, bs = l.split("/")
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "d_convergence": d_convergence,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": scf_type,
                "basis": bs,
            }
        )
        e = psi4.energy(m)
        # e = psi4.energy(m, bsse_type="cp")
        # ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        # ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        if "sapt" in m.lower():
            ELST = psi4.core.variable("SAPT ELST ENERGY")
            EXCH = psi4.core.variable("SAPT EXCH ENERGY")
            IND = psi4.core.variable("SAPT IND ENERGY")
            DISP = psi4.core.variable("SAPT DISP ENERGY")
            out_energies = np.array([e, ELST, EXCH, IND, DISP]) * mult
            es.append(out_energies)
        else:
            es.append(np.array([e * mult]))
        psi4.core.clean()
    return es


def run_sapt0_components(js: jobspec.sapt0_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    generate_outputs = "out" in js.extra_info.keys()
    geom = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    es = []
    for l in js.extra_info["level_theory"]:
        handle_hrcl_extra_info_options(js, l)
        mol = psi4.geometry(geom)
        e = psi4.energy(f"{l}")
        e *= constants.conversion_factor("hartree", "kcal / mol")
        ELST = psi4.core.variable("SAPT ELST ENERGY")
        EXCH = psi4.core.variable("SAPT EXCH ENERGY")
        IND = psi4.core.variable("SAPT IND ENERGY")
        DISP = psi4.core.variable("SAPT DISP ENERGY")
        ie = sum([ELST, EXCH, IND, DISP])
        mult = constants.conversion_factor("hartree", "kcal / mol")
        out_energies = np.array([ie, ELST, EXCH, IND, DISP]) * mult
        es.append(out_energies)
        handle_hrcl_psi4_cleanup(js, l)
    return es


def run_mp_js_grimme_components(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    ies = run_psi4_sapt0_components(
        ma,
        mb,
        ppm=js.mem,
        level_theory=js.level_theory,
    )
    return ies


def run_psi4_sapt0_components(
    A: str,
    B: str,
    ppm: str = "4 gb",
    level_theory: [] = ["hf/cc-pvdz"],
    charge_mult: np.array = np.array([[0, 1], [0, 1], [0, 1]]),
    d_convergence: int = 4,
    scf_type="df",
) -> []:
    A_cm = charge_mult[1, :]
    B_cm = charge_mult[2, :]
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}\n--\n{B_cm[0]} {B_cm[1]}\n{B}"
    es = []
    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "d_convergence": d_convergence,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": scf_type,
                # "cholesky_tolerance": 1e-6 # default about 1e-4
                # check psi4/src/read_options
            }
        )
        psi4.core.be_quiet()
        e = psi4.energy(f"{l}")

        e *= constants.conversion_factor("hartree", "kcal / mol")
        # print(psi4.core.variables())

        ELST = psi4.core.variable("SAPT ELST ENERGY")
        EXCH = psi4.core.variable("SAPT EXCH ENERGY")
        IND = psi4.core.variable("SAPT IND ENERGY")
        DISP = psi4.core.variable("SAPT DISP ENERGY")
        ie = sum([ELST, EXCH, IND, DISP])
        mult = constants.conversion_factor("hartree", "kcal / mol")
        out_energies = np.array([ie, ELST, EXCH, IND, DISP]) * mult
        # es.append(ie)
        es.append(out_energies)
        psi4.core.clean()
    return es


def run_psi4_dimer_ie(js: jobspec.psi4_dimer_js):
    """
    xtra = {"level_theory": ["pbe0/aug-cc-pVDZ"], "options": options}
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)
    charges = [[0, 1] for i in range(3)]
    geom = f"{charges[1][0]} {charges[1][1]}\n{ma}\n"
    geom += f"--\n{charges[2][0]} {charges[2][1]}\n{mb}"

    out = []
    psi4.core.be_quiet()
    level_theory = js.extra_info["level_theory"]
    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(js.mem)
        psi4.set_options(js.extra_info["options"])
        if js.extra_info["bsse_type"] == "cp":
            e = psi4.energy(l, bsse_type="cp")
            ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        elif js.extra_info["bsse_type"] == "nocp":
            e = psi4.energy(l, bsse_type="nocp")
            ie = psi4.core.variable("NOCP-CORRECTED INTERACTION ENERGY")
        else:
            print("bsse_type must be cp or nocp")
            raise ValueError()
        ie *= constants.conversion_factor("hartree", "kcal / mol")
        print(ie)
        out.append(ie)
    return out


def run_psi4_dimer_ie_output_files(js: jobspec.psi4_dimer_js):
    """
    xtra = {"level_theory": ["pbe0/aug-cc-pVDZ"], "options": options}
    """
    psi4.core.be_quiet()
    charges = [[0, 1] for i in range(3)]
    geom = tools.generate_p4input_from_df(js.geometry, charges, js.monAs, js.monBs)
    out = []
    job_dir = js.extra_info["out"]["path"]
    level_theory = js.extra_info["level_theory"]
    for l in level_theory:
        clean_name = (
            l.replace("/", "_").replace("-", "_").replace("(", "_").replace(")", "_")
        )
        job_dir += f"/{js.id_label}/{clean_name}_{js.extra_info['bsse_type']}{js.extra_info['out']['version']}"
        os.makedirs(job_dir, exist_ok=True)
        psi4.set_output_file(f"{job_dir}/psi4.out", False, loglevel=10)
        # psi4.print_out("")
        mol = psi4.geometry(geom)
        psi4.set_memory(js.mem)
        psi4.set_options(js.extra_info["options"])
        if js.extra_info["bsse_type"] == "cp":
            e = psi4.energy(l, bsse_type="cp")
            vs = psi4.core.variables()
            ie = vs["CP-CORRECTED INTERACTION ENERGY"]
            dimer = vs["1_((1, 2), (1, 2))"]
            monA = vs["1_((1,), (1,))"]
            monB = vs["1_((2,), (2,))"]
            cp_correction = vs["CP-CORRECTED INTERACTION ENERGY THROUGH 2-BODY"]
        elif js.extra_info["bsse_type"] == "nocp":
            e = psi4.energy(l, bsse_type="nocp")
            vs = psi4.core.variables()
            ie = vs["NOCP-CORRECTED INTERACTION ENERGY"]
            dimer = vs["1_((1, 2), (1, 2))"]
            monA = vs["1_((1,), (1,))"]
            monB = vs["1_((2,), (2,))"]
            cp_correction = vs["NOCP-CORRECTED INTERACTION ENERGY THROUGH 2-BODY"]
        else:
            print("bsse_type must be cp or nocp")
            raise ValueError()
        with open(f"{job_dir}/psi4_vars.json", "w") as f:
            f.write(json.dumps(psi4.core.variables(), indent=4))
        ie *= constants.conversion_factor("hartree", "kcal / mol")
        print(ie)
        ies = [ie, dimer, monA, monB, cp_correction]
        out.extend(ies)
    return out


def generate_job_dir(js, l, sub_job):
    job_dir = js.extra_info["out"]["path"]
    clean_name = (
        l.replace("/", "_").replace("-", "_").replace("(", "_").replace(")", "_")
    )
    job_dir += f"/{js.id_label}/{clean_name}_{js.extra_info['out']['version']}"
    if "out" in js.extra_info.keys() and "sub_path" in js.extra_info["out"].keys():
        job_dir += f"/{js.extra_info['out']['sub_path']}"
    if sub_job != 0:
        job_dir += f"/{sub_job}"
    return job_dir


def handle_hrcl_extra_info_options(js, l, sub_job=0):
    psi4.set_memory(js.mem)
    psi4.set_options(js.extra_info["options"])
    generate_outputs = "out" in js.extra_info.keys()
    set_scratch = "scratch" in js.extra_info.keys()
    if set_scratch:
        psi4.core.IOManager.shared_object().set_default_path(
            os.path.abspath(os.path.expanduser(js.extra_info["scratch"]["path"]))
        )
    if generate_outputs:
        job_dir = generate_job_dir(js, l, sub_job)
        os.makedirs(job_dir, exist_ok=True)
        psi4.set_output_file(f"{job_dir}/psi4.out", False, loglevel=10)
        psi4.core.print_out(f"{js}")
    else:
        psi4.core.be_quiet()
    if "num_threads" in js.extra_info.keys():
        psi4.set_num_threads(js.extra_info["num_threads"])
    return


def handle_hrcl_psi4_cleanup(js, l, sub_job=0, psi4_clean_all=True, wfn=None):
    generate_outputs = "out" in js.extra_info.keys()
    if generate_outputs:
        job_dir = generate_job_dir(js, l, sub_job)
        out_json = f"{job_dir}/{sub_job}_vars.json"
        print(f"{out_json = }")
        if os.path.exists(job_dir):
            with open(out_json, "w") as f:
                out = psi4.core.variables()
                if wfn is not None:
                    tmp = {}
                    for k, v in wfn.variables().items():
                        if isinstance(v, psi4.core.Matrix):
                            tmp[k] = v.to_array(dense=True)
                        else:
                            tmp[k] = v
                    out["wfn"] = tmp
                json_dump = json.dumps(out, indent=4, cls=NumpyEncoder)
                f.write(json_dump)

    if psi4_clean_all:
        psi4.core.clean_options()
        psi4.core.clean_variables()
        psi4.core.clean_timers()
        psi4.core.clean()
    return


def run_saptdft_grac_shift(js: jobspec.saptdft_mon_grac_js, print_level=1):
    """
    xtra = {"level_theory": ["pbe0/aug-cc-pVDZ"], "charge_index": 1, "options": options}
    """
    mn, out = [], []
    for i in js.monNs:
        mn.append(js.geometry[i, :])
    mn = tools.np_carts_to_string(mn)
    charges = js.charges[js.extra_info["charge_index"]]
    geom_neutral = f"{charges[0]} {charges[1]}\n{mn}"
    geom_cation = f"{charges[0]+1} {charges[1]+1}\n{mn}"
    for l in js.extra_info["level_theory"]:
        # Neutral monomer energy
        sub_job = "neutral"
        try:
            handle_hrcl_extra_info_options(js, l, sub_job)
            psi4.geometry(geom_neutral)
            E_neutral, wfn_n = psi4.energy(l, return_wfn=True)
            occ_neutral = wfn_n.epsilon_a_subset(basis="SO", subset="OCC").to_array(
                dense=True
            )
            HOMO = np.amax(occ_neutral)
            handle_hrcl_psi4_cleanup(js, l, sub_job)

            # Cation monomer energy
            sub_job = "cation"
            # Used to read in neutral density as guess for cation, investigate if breaks
            # js.extra_info["options"]["scf__guess"] = "read"
            handle_hrcl_extra_info_options(js, l, sub_job)
            psi4.geometry(geom_cation)
            E_cation, wfn_c = psi4.energy(l, return_wfn=True)
            grac = E_cation - E_neutral + HOMO
            if grac >= 1 or grac <= 0:
                print(f"{grac = }")
                raise Exception("Grac appears wrong. Not inserting into DB.")
            if print_level < 3:
                print(f"{E_cation = } {E_neutral = } {HOMO = } {grac = }")
            out.append(E_neutral)
            out.append(E_cation)
            out.append(HOMO)
            out.append(grac)
            handle_hrcl_psi4_cleanup(js, l, sub_job)
        except (psi4.SCFConvergenceError, Exception) as e:
            out.append(None)
            out.append(None)
            out.append(None)
            out.append(None)
            print(e)
            handle_hrcl_psi4_cleanup(js, l, sub_job)
    return out


def run_saptdft_sapt_2p3_s_inf(js: jobspec.saptdft_js) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    geom = js.psi4_input
    es = []
    mol = psi4.geometry(geom)
    l = "sapt_dft_2p3"
    try:
        sub_job = "sapt_dft"
        handle_hrcl_extra_info_options(js, l, sub_job)
        e1 = psi4.energy("sapt(dft)")
        es.append(psi4.core.variable("EXCH-IND20,R (S^INF)"))
        es.append(psi4.core.variable("SAPT EXCH-DISP20(S^INF) ENERGY"))
        handle_hrcl_psi4_cleanup(js, l, sub_job)

        sub_job = "sapt_2p3"
        handle_hrcl_extra_info_options(js, l, sub_job)
        e2 = psi4.energy("sapt2+3")
        es.append(psi4.core.variable("SAPT EXCH-IND30(S^INF) ENERGY"))
        handle_hrcl_psi4_cleanup(js, l, sub_job)
    except Exception as e:
        print("Exception:", e)
        out_energies = None
        handle_hrcl_psi4_cleanup(js, l, sub_job)
        return [None, None, None]
    return es


def run_MBIS_mbs(js: jobspec.sapt0_js, print_energies=False) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    es = []
    geom_d = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    geom_A, geom_B = geom_d.split("--")
    geom_A += "\nunits angstrom"
    # print(geom_d, geom_A, geom_B, sep="\n\n")

    for l in js.extra_info["level_theory"]:
        try:
            sub_job = "MBIS_dimer"
            mol_d = psi4.geometry(geom_d)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_d, wfn_d = psi4.energy(l, return_wfn=True)
            if print_energies:
                print("dimer energy", e_d)
            oeprop(wfn_d, "MBIS_VOLUME_RATIOS")
            vol_ratio_d = np.array(wfn_d.variable("MBIS VOLUME RATIOS"))
            handle_hrcl_psi4_cleanup(js, l, sub_job)

            sub_job = "MBIS_monA"
            psi4.geometry(geom_A)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_a, wfn_a = psi4.energy(l, return_wfn=True)
            if print_energies:
                print("monA energy", e_a)
            oeprop(wfn_a, "MBIS_VOLUME_RATIOS")
            vol_ratio_a = np.array(wfn_a.variable("MBIS VOLUME RATIOS"))
            handle_hrcl_psi4_cleanup(js, l, sub_job)

            sub_job = "MBIS_monB"
            psi4.geometry(geom_B)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_b, wfn_b = psi4.energy(l, return_wfn=True)
            if print_energies:
                print("monB energy", e_b)
            oeprop(wfn_b, "MBIS_VOLUME_RATIOS")
            vol_ratio_b = np.array(wfn_b.variable("MBIS VOLUME RATIOS"))
            handle_hrcl_psi4_cleanup(js, l, sub_job)
            es.append(vol_ratio_d)
            es.append(vol_ratio_a)
            es.append(vol_ratio_b)

        except Exception as e:
            print("Exception:", e)
            out_energies = None
            handle_hrcl_psi4_cleanup(js, l, sub_job)
            for i in range(7):
                es.append(None)
    return es


def MBIS_props_from_wfn(wfn):
    oeprop(wfn, "MBIS_VOLUME_RATIOS")
    charges = np.array(wfn.variable("MBIS CHARGES"))
    dipoles = np.array(wfn.variable("MBIS DIPOLES"))
    widths = np.array(wfn.variable("MBIS VALENCE WIDTHS"))
    try:
        vol_ratio = np.array(wfn.variable("MBIS VOLUME RATIOS"))
    except KeyError:
        vol_ratio = np.ones(1)
    quad = np.array(wfn.variable("MBIS QUADRUPOLES"))
    quad = [q[np.triu_indices(3)] for q in quad]
    quadrupoles = np.array(quad)
    multipoles = np.concatenate([charges, dipoles, quadrupoles], axis=1)
    radial_2 = np.array(wfn.variable("MBIS RADIAL MOMENTS <R^2>"))
    radial_3 = np.array(wfn.variable("MBIS RADIAL MOMENTS <R^3>"))
    radial_4 = np.array(wfn.variable("MBIS RADIAL MOMENTS <R^4>"))
    return wfn, multipoles, widths, vol_ratio, radial_2, radial_3, radial_4


def MBIS_population(geometry: np.array, multipoles: np.ndarray):
    population = np.zeros(len(geometry))
    for i, m in enumerate(multipoles):
        population[i] = geometry[i, 0] - m[0]
    return population


def run_MBIS(js: jobspec.sapt0_js, print_energies=False) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    es = []
    geom_d = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    geom_A, geom_B = geom_d.split("--")
    geom_A += "\nunits angstrom"

    for l in js.extra_info["level_theory"]:
        try:
            sub_job = "MBIS_dimer"
            mol_d = psi4.geometry(geom_d)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_d, wfn_d = psi4.energy(l, return_wfn=True)
            (
                wfn_d,
                multipoles_d,
                widths_d,
                vol_ratio_d,
                radial_2_d,
                radial_3_d,
                radial_4_d,
            ) = MBIS_props_from_wfn(wfn_d)
            population_d = MBIS_population(js.geometry, multipoles_d)
            handle_hrcl_psi4_cleanup(js, l, sub_job, wfn=wfn_d)

            sub_job = "MBIS_monA"
            psi4.geometry(geom_A)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_a, wfn_a = psi4.energy(l, return_wfn=True)
            (
                wfn_a,
                multipoles_a,
                widths_a,
                vol_ratio_a,
                radial_2_a,
                radial_3_a,
                radial_4_a,
            ) = MBIS_props_from_wfn(wfn_a)
            population_a = MBIS_population(js.geometry[js.monAs], multipoles_a)
            handle_hrcl_psi4_cleanup(js, l, sub_job, wfn=wfn_a)

            sub_job = "MBIS_monB"
            psi4.geometry(geom_B)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_b, wfn_b = psi4.energy(l, return_wfn=True)
            (
                wfn_b,
                multipoles_b,
                widths_b,
                vol_ratio_b,
                radial_2_b,
                radial_3_b,
                radial_4_b,
            ) = MBIS_props_from_wfn(wfn_b)
            population_b = MBIS_population(js.geometry[js.monBs], multipoles_b)
            handle_hrcl_psi4_cleanup(js, l, sub_job, wfn=wfn_b)
            es.append(multipoles_d)
            es.append(multipoles_a)
            es.append(multipoles_b)
            es.append(widths_d)
            es.append(widths_a)
            es.append(widths_b)
            es.append(vol_ratio_d)
            es.append(vol_ratio_a)
            es.append(vol_ratio_b)
            es.append(radial_2_d)
            es.append(radial_2_a)
            es.append(radial_2_b)
            es.append(radial_3_d)
            es.append(radial_3_a)
            es.append(radial_3_b)
            es.append(radial_4_d)
            es.append(radial_4_a)
            es.append(radial_4_b)
            es.append(population_d)
            es.append(population_a)
            es.append(population_b)

        except Exception as e:
            print("Exception:", e)
            out_energies = None
            handle_hrcl_psi4_cleanup(js, l, sub_job)
            for i in range(21):
                es.append(None)
    return es


def run_MBIS_monomer(js: jobspec.monomer_js, print_energies=False) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    es = []
    monAs = [i for i in range(len(js.geometry))]
    geom = tools.generate_p4input_from_df(
        js.geometry, js.charges, monAs, units="angstrom"
    )

    for l in js.extra_info["level_theory"]:
        try:
            sub_job = "MBIS_mon"
            mol_d = psi4.geometry(geom)
            handle_hrcl_extra_info_options(js, l, sub_job)
            e_d, wfn_d = psi4.energy(l, return_wfn=True)
            (
                wfn_d,
                multipoles_d,
                widths_d,
                vol_ratio_d,
                radial_2,
                radal_3,
                radial_4,
            ) = MBIS_props_from_wfn(wfn_d)
            population = MBIS_population(js.geometry, multipoles_d)
            handle_hrcl_psi4_cleanup(js, l, sub_job, wfn=wfn_d)

            es.append(multipoles_d)
            es.append(widths_d)
            es.append(vol_ratio_d)
            es.append(radial_2)
            es.append(radal_3)
            es.append(radial_4)
            es.append(population)

        except Exception as e:
            print("Exception:", e)
            out_energies = None
            handle_hrcl_psi4_cleanup(js, l, sub_job)
            for i in range(7):
                es.append(None)
    return es


def run_interaction_energy(js: jobspec.sapt0_js) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    geom = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    bsse_type = js.extra_info["bsse_type"]
    ie = [None]
    sub_job = "ie"
    for l in js.extra_info["level_theory"]:
        mol = psi4.geometry(geom)
        handle_hrcl_extra_info_options(js, l, sub_job)
        try:
            e = psi4.energy(f"{l}", bsse_type=bsse_type)
            ie[0] = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        except Exception as e:
            print("Exception:", e)
        handle_hrcl_psi4_cleanup(js, l, sub_job)
    return ie


def run_interaction_energy_cp(js: jobspec.sapt0_js) -> np.array:
    """ """

    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options(
            {
                "d_convergence": d_convergence,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": scf_type,
                # "cholesky_tolerance": 1e-6 # default about 1e-4
                # check psi4/src/read_options
            }
        )
        # psi4.core.be_quiet()
        if cp:
            e = psi4.energy(l)
            ie = psi4.core.variable("CP-CORRECTED INTERACTION ENERGY")
        else:
            e = psi4.energy(l, bsse_type="nocp")
            ie = psi4.core.variable("NOCP-CORRECTED INTERACTION ENERGY")
        ie *= constants.conversion_factor("hartree", "kcal / mol")
        es.append(ie)
        psi4.core.clean()
    return es


def create_psi4_input_file(js: jobspec.sapt0_js) -> np.array:
    generate_outputs = "out" in js.extra_info.keys()
    geom = tools.generate_p4input_from_df(
        js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
    )
    es = []
    if not generate_outputs:
        print("No output file specified. Not generating input files.")
    for l in js.extra_info["level_theory"]:
        sub_job = js.extra_info['options']['pno_convergence']
        job_dir = generate_job_dir(js, l, sub_job)
        os.makedirs(job_dir, exist_ok=True)
        opts = ""
        for k, v in js.extra_info["options"].items():
            opts += f"{k} {v}\n"
        with open(f"{job_dir}/psi4.in", "w") as f:
            f.write(
                f"""
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

memory {js.mem}

molecule dimer {{
{geom}
no_com
no_reorient
}}
set {{
    {opts}
}}

{js.extra_info['function_call']}

with open("vars.json", "w") as f:
     json_dump = json.dumps(psi4.core.variables(), indent=4, cls=NumpyEncoder)
     f.write(json_dump)

        """
            )
    return [None for i in range(len(js.extra_info["level_theory"]))]
