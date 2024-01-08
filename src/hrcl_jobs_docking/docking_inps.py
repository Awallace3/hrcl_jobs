import os
from glob import glob
import subprocess
import pandas as pd
import numpy as np
from dataclasses import dataclass
from . import jobspec
from qcelemental import constants
import json
import qm_tools_aw 
import qcelemental as qcel
from pprint import pprint as pp
import MDAnalysis as mda
from MDAnalysis.exceptions import NoDataError

# docking specific imports
from vina import Vina

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def mda_selection_to_xyz_cm(
    selection,
    multiplicity=1,
) -> (np.ndarray, np.ndarray):
    """
    Gather the xyz coordinates and charge+multiplicity from a selection of
    atoms in a universe. NOTE: multiplicity is currently set to 1.
    """
    selection_xyz = selection.positions
    selection_elements = selection.atoms.elements
    try:
        selection_charge = sum(selection.atoms.formalcharges)
    except NoDataError:
        selection_charge = 0
        print("setting charge to 0")
        pass

    cm = np.array([selection_charge, multiplicity], dtype=np.int32) # TODO: make multiplicity a variable
    selection_elements = [qcel.periodictable.to_Z(i) for i in selection_elements]
    g = np.concatenate((np.reshape(selection_elements, (-1, 1)), selection_xyz), axis=1)
    return g, cm


# def run_sapt0_components(js: jobspec.sapt0_js) -> np.array:
#     """
#     create_mp_js_grimme turns mp_js object into a psi4 job and runs it
#     """
#     generate_outputs = "out" in js.extra_info.keys()
#     geom = tools.generate_p4input_from_df(
#         js.geometry, js.charges, js.monAs, js.monBs, units="angstrom"
#     )
#     es = []
#     for l in js.extra_info["level_theory"]:
#         handle_hrcl_extra_info_options(js, l)
#         mol = psi4.geometry(geom)
#         e = psi4.energy(f"{l}")
#
#         e *= constants.conversion_factor("hartree", "kcal / mol")
#         ELST = psi4.core.variable("SAPT ELST ENERGY")
#         EXCH = psi4.core.variable("SAPT EXCH ENERGY")
#         IND = psi4.core.variable("SAPT IND ENERGY")
#         DISP = psi4.core.variable("SAPT DISP ENERGY")
#         ie = sum([ELST, EXCH, IND, DISP])
#         mult = constants.conversion_factor("hartree", "kcal / mol")
#         out_energies = np.array([ie, ELST, EXCH, IND, DISP]) * mult
#         es.append(out_energies)
#         handle_hrcl_psi4_cleanup(js, l)
#     return es


def run_apnet_discos(js: jobspec.apnet_disco_js) -> []:
    """
    Input columns:
        - PRO_PDB: str
        - LIG_PDB: str
        - WAT_PDB: str
        - OTH_PDB: str
    Output columns:
        - apnet_totl_LIG: float
        - apnet_elst_LIG: float
        - apnet_exch_LIG: float
        - apnet_indu_LIG: float
        - apnet_disp_LIG: float
        - apnet_errors: str
    """
    import apnet
    pro_universe = mda.Universe(js.PRO_PDB)
    lig_universe = mda.Universe(js.LIG_PDB)
    # if wat_pdb is not None:
    #     wat_universe = mda.Universe(js.WAT_PDB)
    # if oth_pdb is not None:
    #     oth_universe = mda.Universe(js.OTH_PDB)
    # pro_xyz, pro_cm = mda_selection_to_xyz_cm(
    #     pro_universe.select_atoms("protein and altloc A")
    # )
    # lig_xyz, lig_cm = mda_selection_to_xyz_cm(
    #     lig_universe.select_atoms("not protein")
    # )
    try:
        pro_xyz, pro_cm = mda_selection_to_xyz_cm(
            pro_universe.select_atoms("protein and not altloc B")
        )
        if len(pro_xyz[:, 0]) == 0:
            pro_xyz, pro_cm = mda_selection_to_xyz_cm(
                pro_universe.select_atoms("protein")
            )
        lig_xyz, lig_cm = mda_selection_to_xyz_cm(
            lig_universe.select_atoms("not protein")
        )
    except Exception as e:
        e = "Could not read PDB"
        return [None, None, None, None, None, str(e)]
    mon_a = qm_tools_aw.tools.print_cartesians_pos_carts_symbols(
        pro_xyz[:, 0], pro_xyz[:, 1:], only_results=True
    )
    mon_b = qm_tools_aw.tools.print_cartesians_pos_carts_symbols(
        lig_xyz[:, 0], lig_xyz[:, 1:], only_results=True
    )
    apnet_error = None
    print(js.PRO_PDB, js.LIG_PDB, len(pro_xyz[:, 0]), len(lig_xyz[:, 0]), sep="\n")
    try:
        geom = f"{pro_cm[0]} {pro_cm[1]}\n{mon_a}--\n{lig_cm[0]} {lig_cm[1]}\n{mon_b}\nunits angstrom\n"
        mol = qcel.models.Molecule.from_data(geom)
    except (Exception, ValueError) as e:
        print(e)
        lig_cm[1] = 2
        geom = f"{pro_cm[0]} {pro_cm[1]}\n{mon_a}--\n{lig_cm[0]} {lig_cm[1]}\n{mon_b}\nunits angstrom\n"
        try:
            mol = qcel.models.Molecule.from_data(geom)
        except (Exception, ValueError) as e:
            apnet_error = e
            print(apnet_error)
            return [None, None, None, None, None, str(e)]
    # prediction, uncertainty = apnet.predict_sapt(dimers=[mol])
    # print(prediction, uncertainty)
    try:
        print(mol)
        prediction, uncertainty = apnet.predict_sapt(dimers=[mol])
        print(prediction, uncertainty)
        prediction = prediction[0]
        update_values = (
            prediction[0],
            prediction[1],
            prediction[2],
            prediction[3],
            prediction[4],
        )
        update_values = [float(i) for i in update_values]
        update_values.append(None)
        print(update_values)
    except Exception as e:
        print(e)
        return [None, None, None, None, None, str(e)]
    return update_values


def get_com(pdbqt_file):
    u = mda.Universe(pdbqt_file)
    com = u.atoms.center_of_mass()
    return com


def run_vina_simple(js: jobspec.vina_js) -> []:
    """
    User must provide the following in js.extra_info:
    - sf_name: str
    - setup_python_files_path: str
    where sf_name is the name of the scoring function ['vina', 'ad4'] and
    setup_python_files_path is path to ligand_preparation.py and
    receptor_preparation.py
    """
    js.extra_info["setup_python_files_path"]
    v = Vina(sf_name=js.extra_info["sf_name"])
    PRO_PDBQT = js.PRO_PDB + "qt"
    LIG_PDBQT = js.LIG_PDB + "qt"
    WAT_PDBQT = js.WAT_PDB + "qt"
    OTH_PDBQT = js.OTH_PDB + "qt"
    v.set_receptor(PRO_PDBQT)
    v.set_ligand_from_file(LIG_PDBQT)
    com = get_com(js.LIG_PDB)
    print(com)
    v.compute_vina_maps(com, box_size=[20, 20, 20])

    # Score the current pose
    energy = v.score()
    print("Score before minimization: %.3f (kcal/mol)" % energy[0])

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    print("Score after minimization : %.3f (kcal/mol)" % energy_minimized[0])
    v.write_pose("ligand_minimized.pdbqt", overwrite=True)

    # Dock the ligand
    v.dock(exhaustiveness=32, n_poses=20)
    vina_out = LIG_PDBQT.replace(".pdbqt", "_out.pdbqt")
    v.write_poses(vina_out, n_poses=10, overwrite=True)
    return [energy]


# TODO: Update this function to return data as a list in the following order (and typing) 
        # "vina_total__LIG": "REAL",
        # "vina_inter__LIG": "REAL",
        # "vina_intra__LIG": "REAL",
        # "vina_torsion__LIG": "REAL",
        # "vina_intra_best_pose__LIG": "REAL",
        # "vina_poses_pdbqt__LIG": "TEXT",
        # "vina_all_poses__LIG": "array",
        # "vina_errors__LIG": "TEXT",
def run_autodock_vina(js: jobspec.autodock_vina_disco_js) -> []:
    """
    User must provide the following in js.extra_info:
    - sf_name: str
    - setup_python_files_path: str
    where sf_name is the name of the scoring function ['vina', 'ad4'] and
    setup_python_files_path is path to ligand_preparation.py and
    receptor_preparation.py
    """
    sf = js.extra_info["scoring_function"]
    v = Vina(sf_name=sf)
    PRO_PDBQT = js.PRO_PDB + "qt"
    LIG_PDBQT = js.LIG_PDB + "qt"
    WAT_PDBQT = js.WAT_PDB + "qt"
    OTH_PDBQT = js.OTH_PDB + "qt"

    # TODO: Add logic for running autodock-vina, return length 7
    return []
