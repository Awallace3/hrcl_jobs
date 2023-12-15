import os
import subprocess
import pandas as pd
import numpy as np
from dataclasses import dataclass
from . import jobspec
from qcelemental import constants
import json
from qm_tools_aw import tools
import qcelemental as qcel
from pprint import pprint as pp

# docking specific imports
from vina import Vina
import MDAnalysis as mda

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
    js.extra_info['setup_python_files_path']
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
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.write_pose('ligand_minimized.pdbqt', overwrite=True)

    # Dock the ligand
    v.dock(exhaustiveness=32, n_poses=20)
    vina_out = LIG_PDBQT.replace(".pdbqt", "_out.pdbqt")
    v.write_poses(vina_out, n_poses=10, overwrite=True)
    return [energy]
