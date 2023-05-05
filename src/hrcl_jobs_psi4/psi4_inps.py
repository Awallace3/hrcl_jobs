import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .jobspec import mp_js, grimme_js, saptdft_js, mp_mon_js
from periodictable import elements
import psi4
from psi4 import oeprop
from qcelemental import constants

# from .tools import np_carts_to_string
from qm_tools_aw.tools import np_carts_to_string
import qcelemental as qcel
from pprint import pprint as pp
"""
/theoryfs2/ds/amwalla3/miniconda3/envs/psi4mpi4py_qcng/lib/python3.8/site-packages/psi4/driver/driver_nbody.py

"""


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
    external_potential = np.concatenate([env_charges, env_coords],
                                        axis=1).reshape((-1, 4))
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
        js.mem, level_theory, mol_d)
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
        js.mem, level_theory, mol_d)
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
        js.mem, level_theory, mol_A)
    output = [
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
        js.mem, level_theory, mol_d)

    mol_A = prep_mol(js.RA, js.ZA, js.TQA, EA)
    vac_multipole_A, charges_A, vac_widths_A, vac_vol_rat_A = psi4_vac_mp(
        js.mem, level_theory, mol_A)

    mol_B = prep_mol(js.RB, js.ZB, js.TQB, EB)
    vac_multipole_B, charges_B, vac_widths_B, vac_vol_rat_B = psi4_vac_mp(
        js.mem, level_theory, mol_B)

    env_multipole_A, env_widths_A, env_vol_rat_A = psi4_env_mp(
        js.mem, level_theory, mol_A, js.RB, charges_B)
    env_multipole_B, env_widths_B, env_vol_rat_B = psi4_env_mp(
        js.mem, level_theory, mol_B, js.RA, charges_A)
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    ies = run_psi4_sapt0(ma,
                         mb,
                         ppm=js.mem,
                         level_theory=js.level_theory,
                         cp=False)
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)

    ies_parts = run_psi4_fsapt(ma,
                               mb,
                               ppm=js.mem,
                               level_theory=js.level_theory)
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)

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
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}--\n{A_cm[0]} {A_cm[1]}\n{B}"
    es_parts = []
    l = level_theory[0]
    mol = psi4.geometry(geom)
    psi4.set_memory(ppm)
    psi4.set_options({
        "d_convergence": d_convergence,
        "freeze_core": "True",
        "guess": "sad",
        "scf_type": scf_type,
        "FISAPT_FSAPT_FILEPATH": "hello",
        "FISAPT_DO_FSAPT_DISP": True,
    })
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
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}--\n{A_cm[0]} {A_cm[1]}\n{B}"
    es = []

    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options({
            "d_convergence": d_convergence,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": scf_type,
            # "cholesky_tolerance": 1e-6 # default about 1e-4
            # check psi4/src/read_options
        })
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
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}--\n{A_cm[0]} {A_cm[1]}\n{B}"
    # print(geom)
    es = []

    for l in level_theory:
        m, bs = l.split("/")
        # print(m, bs)
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options({
            "reference": "rhf",
            "basis": "aug-cc-pVDZ",
            "sapt_dft_grac_shift_a": sapt_dft_grac_shift_a,
            "sapt_dft_grac_shift_b": sapt_dft_grac_shift_b,
            "SAPT_DFT_FUNCTIONAL": m,
        })
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    shift_a = run_dft_neutral_cation(ma,
                                     charges=js.charges[1],
                                     ppm=js.mem,
                                     level_theory=js.level_theory)
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


def run_saptdft(js: saptdft_js) -> np.array:
    """
    run_saptdft computes scaling factor and uses to run SAPT-DFT
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    shift_a = run_dft_neutral_cation(ma,
                                     charges=js.charges[1],
                                     ppm=js.mem,
                                     level_theory=js.level_theory)
    shift_b = run_dft_neutral_cation(mb,
                                     charges=js.charges[2],
                                     ppm=js.mem,
                                     level_theory=js.level_theory)
    shift_a.extend(shift_b)
    ies = run_psi4_saptdft(
        ma,
        mb,
        ppm=js.mem,
        level_theory=js.level_theory,
        sapt_dft_grac_shift_a=shift_a[-1],
        sapt_dft_grac_shift_b=shift_b[-1],
    )
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
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}--\n{A_cm[0]} {A_cm[1]}\n{B}"
    es = []

    for l in level_theory:
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options({
            "d_convergence": d_convergence,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": scf_type,
            # "cholesky_tolerance": 1e-6 # default about 1e-4
            # check psi4/src/read_options
        })
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
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    ies = run_psi4_saptdft(ma, mb, ppm=js.mem, level_theory=js.level_theory)
    return ies


def run_dft_neutral_cation(M,
                           charges,
                           ppm,
                           level_theory,
                           d_convergence="8") -> np.array:
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
        psi4.set_options({
            "reference": "uhf",
        })
        e_neutral, wfn_n = psi4.energy(l, return_wfn=True)
        e_neutral = e_neutral
        occ_neutral = wfn_n.epsilon_a_subset(basis="SO",
                                             subset="OCC").to_array(dense=True)
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


def run_dft_neutral_cation_qca(M,
                               charges,
                               ppm,
                               level_theory,
                               d_convergence="8") -> np.array:
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
        psi4.set_options({
            "reference": "uhf",
        })
        e_neutral, wfn_n = psi4.energy(l, return_wfn=True)
        e_neutral = e_neutral
        occ_neutral = wfn_n.epsilon_a_subset(basis="SO",
                                             subset="OCC").to_array(dense=True)
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
    geom = f"{A}--\n{B}"
    es = []
    mult = constants.conversion_factor("hartree", "kcal / mol")
    for l in level_theory:
        m, bs = l.split("/")
        mol = psi4.geometry(geom)
        psi4.set_memory(ppm)
        psi4.set_options({
            "d_convergence": d_convergence,
            "freeze_core": "True",
            "guess": "sad",
            "scf_type": scf_type,
            "basis": bs,
        })
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

def run_mp_js_grimme_components(js: grimme_js) -> np.array:
    """
    create_mp_js_grimme turns mp_js object into a psi4 job and runs it
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
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
    geom = f"{A_cm[0]} {A_cm[1]}\n{A}--\n{A_cm[0]} {A_cm[1]}\n{B}"
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
