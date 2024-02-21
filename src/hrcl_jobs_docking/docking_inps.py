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
import signal

# docking specific imports
from vina import Vina


class SegFault(Exception):
    pass


def segfault_handler(signum, frame):
    print("segfault")
    raise SegFault("segfault")


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

    cm = np.array(
        [selection_charge, multiplicity], dtype=np.int32
    )  # TODO: make multiplicity a variable
    selection_elements = [qcel.periodictable.to_Z(i) for i in selection_elements]
    g = np.concatenate((np.reshape(selection_elements, (-1, 1)), selection_xyz), axis=1)
    return g, cm


def mda_selection_to_xyz(
    selection,
) -> (np.ndarray, np.ndarray):
    """
    Gather the xyz coordinates and charge+multiplicity from a selection of
    atoms in a universe. NOTE: multiplicity is currently set to 1.
    """
    selection_xyz = selection.positions
    selection_elements = selection.atoms.elements
    selection_elements = [qcel.periodictable.to_Z(i) for i in selection_elements]
    g = np.concatenate((np.reshape(selection_elements, (-1, 1)), selection_xyz), axis=1)
    return g


def compute_memory(memory_per_process):
    """
    Convert memory_per_process to bytes
    NOTE: using the binary definition of (GB -> bytes) meaning 1 GB = 1024^3 bytes

    if an number is passed in, it is assumed to be in GB

    """
    if type(memory_per_process) == str:
        if (
            memory_per_process[-1].upper() == "G"
            or memory_per_process[-2:].upper() == "GB"
        ):
            mem_value = "".join([i for i in memory_per_process if i.isdigit()])
            memory_per_process = float(mem_value) * 1024 * 1024 * 1024
        elif memory_per_process[-1] == "M":
            memory_per_process = float(memory_per_process[:-1]) * 1024 * 1024
    else:
        memory_per_process = float(memory_per_process) * 1024 * 1024 * 1024
    return memory_per_process


def run_apnet_pdbs(js: jobspec.apnet_pdbs_js) -> []:
    """
    Input columns:
        - PRO_PDB: str # ensure already protenated
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
    import tensorflow as tf

    # tf.config.threading.set_intra_op_parallelism_threads(js.extra_info["n_cpus"])
    # tf.config.threading.set_inter_op_parallelism_threads(js.extra_info["n_cpus"])
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    physical_cpus = tf.config.experimental.list_physical_devices("CPU")
    assert physical_cpus, "No CPUs were detected."
    # memory = compute_memory(js.extra_info["mem_per_process"])
    # assert memory, "No memory was detected."

    # tf.config.threading.set_intra_op_parallelism_threads(0)
    # tf.config.threading.set_inter_op_parallelism_threads(0)

    if js.PRO_PDB is None:
        return [None, None, None, None, None, "PRO_PDB is None"]
    if js.LIG_PDB is None:
        return [None, None, None, None, None, "LIG_PDB is None"]

    try:
        pro_universe = mda.Universe(js.PRO_PDB)
        lig_universe = mda.Universe(js.LIG_PDB)
        pro_xyz = mda_selection_to_xyz(
            pro_universe.select_atoms("all and not altloc B")
        )
        atom_count_protein = len(pro_xyz)
        max_atoms = js.extra_info.get("atom_max", None)
        if max_atoms is not None and atom_count_protein > max_atoms:
            return [None, None, None, None, None, "atom_count_protein > atom_max"]
        lig_xyz = mda_selection_to_xyz(lig_universe.atoms)
        pro_cm = [js.PRO_CHARGE, 1]
        lig_cm = [js.LIG_CHARGE, 1]
        print(f"{pro_universe} {js.PRO_CHARGE} {js.PRO_PDB}", f"{lig_universe} {js.LIG_CHARGE} {js.LIG_PDB}", sep="\n")
        mon_a = qm_tools_aw.tools.print_cartesians_pos_carts_symbols(
            pro_xyz[:, 0], pro_xyz[:, 1:], only_results=True
        )
        mon_b = qm_tools_aw.tools.print_cartesians_pos_carts_symbols(
            lig_xyz[:, 0], lig_xyz[:, 1:], only_results=True
        )
        apnet_error = None
    except Exception as e:
        print(e)
        e = "Could not read PDB"
        return [None, None, None, None, None, str(e)]
    try:
        geom = f"{pro_cm[0]} {pro_cm[1]}\n{mon_a}--\n{lig_cm[0]} {lig_cm[1]}\n{mon_b}\nunits angstrom\n"
        mol = qcel.models.Molecule.from_data(geom)
        print(mol)

        prediction, uncertainty = apnet.predict_sapt(dimers=[mol])
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
    except Exception as e:
        print(e)
        return [None, None, None, None, None, str(e)]
    return update_values


def get_com(pdbqt_file):
    u = mda.Universe(pdbqt_file)
    com = u.atoms.center_of_mass().tolist()
    return com


def prepare_ligand4(ligand_filename, outputfilename):
    cmd = f"python3 ~/data/gits/apnet_docking/docking_practice/MGLToolsPckgsPy3/prepare_ligand4.py -l ligand_filename -o outputfilename"
    out = subprocess.run(cmd, shell=True, check=True)


def prepare_receptor4(receptor_filename, outputfilename):
    cmd = f"python3 ~/data/gits/apnet_docking/docking_practice/MGLToolsPckgsPy3/prepare_receptor4.py -r receptor_filename -o outputfilename"
    out = subprocess.run(cmd, shell=True, check=True)


def run_autodock_vina(js: jobspec.autodock_vina_js, verbose=1) -> []:
    """
    User must provide the following in js.extra_info:
    - sf_name: str
    - setup_python_files_path: str
    where sf_name is the name of the scoring function ['vina', 'ad4'] and
    setup_python_files_path is path to ligand_preparation.py and
    receptor_preparation.py
    """
    import docking_tools_amw

    # try:
    if "n_poses" in js.extra_info.keys():
        n_poses = js.extra_info["sf_params"]["n_poses"]
    else:
        n_poses = 10
    if "exhaustiveness" in js.extra_info.keys():
        exhaustiveness = js.extra_info["sf_params"]["exhaustiveness"]
    else:
        exhaustiveness = 32
    if "npts" in js.extra_info.keys():
        npts = js.extra_info["sf_params"]["npts"]
    else:
        npts = [54, 54, 54]
    if "box_size" in js.extra_info.keys():
        box_size = js.extra_info["sf_params"]["box_size"]
    else:
        box_size = [30, 30, 30]
    identifier = js.extra_info.get("identifier", None)
    if identifier is None:
        identifier = "_0"
    else:
        identifier = f"_{identifier}"
    npts_param = f"npts={npts[0]},{npts[1]},{npts[2]}"
    sf_name = js.extra_info["sf_name"]
    v = Vina(sf_name=sf_name, cpu=js.extra_info["n_cpus"], seed=875234)
    # TODO: set_weights for vina and ad4
    # need to run on docked structure single-point energy and collect
    # v.set_weights([])
    if js.PRO_PDB is None:
        return [None, None, None, None, None, None, None, "PRO_PDBQT is None"]
    if js.LIG_PDB is None:
        return [None, None, None, None, None, None, None, "LIG_PDBQT is None"]

    PRO_PDBQT = js.PRO_PDB + "qt"
    LIG_PDBQT = js.LIG_PDB + "qt"
    # WAT_PDBQT = js.WAT_PDB + "qt"
    # OTH_PDBQT = js.OTH_PDB + "qt"
    PRO = PRO_PDBQT.replace(".pdbqt", "")
    def_dir = os.getcwd()
    try:
        if verbose:
            print(PRO_PDBQT, LIG_PDBQT, sep="\n")
        if not os.path.exists(PRO_PDBQT):
            docking_tools_amw.prepare_receptor4.prepare_receptor4(
                receptor_filename=js.PRO_PDB,
                outputfilename=PRO_PDBQT,
                # charges_to_add=None,
            )
        if not os.path.exists(LIG_PDBQT):
            docking_tools_amw.prepare_ligand4.prepare_ligand4(
                ligand_filename=js.LIG_PDB,
                outputfilename=LIG_PDBQT,
                repairs="hydrogen",
            )
        # find the center of the binding pocket, for this dataset that is also the center of mass of the ligand
        com = get_com(js.LIG_PDB)
        # set the ligand
        ad_vina_errors = None
        # if vina or vinardo then set the receptor and computer the vina maps, if autodock then prepare the gpf and autogrid
        # NOTE: receptor must be set before ligand
        if sf_name in ["vina", "vinardo"]:
            print(sf_name)
            v.set_receptor(PRO_PDBQT)
            v.compute_vina_maps(center=com, box_size=box_size)
        elif sf_name == "ad4":
            os.chdir("/".join(PRO.split("/")[:-1]))
            PRO_MAP = PRO.split("/")[-1]
            PRO = PRO_MAP + identifier

            PRO_GPF = f"{PRO}.gpf"
            PRO_GLG = f"{PRO}.glg"
            print(f"{PRO_GPF = }")
            PRO_PDBQT = PRO_PDBQT.split("/")[-1]
            LIG_PDBQT = LIG_PDBQT.split("/")[-1]
            docking_tools_amw.prepare_gpf.prepare_gpf(
                receptor_filename=PRO_PDBQT,
                ligand_filename=LIG_PDBQT,
                output_gpf_filename=PRO_GPF,
                parameters=[npts_param],
                identifier=identifier,
            )
            cmd = f"autogrid4 -p {PRO_GPF} -l {PRO_GLG}"
            os.system(cmd)
            v.load_maps(PRO)
        else:
            ad_vina_errors = "invalid sf_name"
        v.set_ligand_from_file(LIG_PDBQT)

        # docking
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        vina_out = LIG_PDBQT.replace(".pdbqt", f"{js.LIG_NAME.replace(':', '_')}_{js.system}_out.pdb")
        v.write_poses(vina_out, n_poses=n_poses, overwrite=True)
        energies = v.energies(n_poses=n_poses)

        if sf_name == "ad4":
            cmd = f"rm {PRO_GPF} {PRO_GLG} {PRO}.map*"
            os.system(cmd)
            os.chdir(def_dir)

    except Exception as e:
        ad_vina_errors = str(e)
        os.chdir(def_dir)
    if ad_vina_errors == None:
        return [
            energies[0][0],
            energies[0][1],
            energies[0][2],
            energies[0][3],
            energies[0][4],
            vina_out,
            energies,
            ad_vina_errors,
        ]
    else:
        return [None, None, None, None, None, None, None, ad_vina_errors]

def run_autodock_vina_components(js: jobspec.autodock_vina_js, verbose=1) -> []:
    """
    User must provide the following in js.extra_info:
    - sf_name: str
    - setup_python_files_path: str
    where sf_name is the name of the scoring function ['vina', 'ad4'] and
    setup_python_files_path is path to ligand_preparation.py and
    receptor_preparation.py
    """
    import docking_tools_amw

    # try:
    if "n_poses" in js.extra_info.keys():
        n_poses = js.extra_info["sf_params"]["n_poses"]
    else:
        n_poses = 10
    if "exhaustiveness" in js.extra_info.keys():
        exhaustiveness = js.extra_info["sf_params"]["exhaustiveness"]
    else:
        exhaustiveness = 32
    if "npts" in js.extra_info.keys():
        npts = js.extra_info["sf_params"]["npts"]
    else:
        npts = [54, 54, 54]
    if "box_size" in js.extra_info.keys():
        box_size = js.extra_info["sf_params"]["box_size"]
    else:
        box_size = [30, 30, 30]
    identifier = js.extra_info.get("identifier", None)
    if identifier is None:
        identifier = "_0"
    else:
        identifier = f"_{identifier}"
    npts_param = f"npts={npts[0]},{npts[1]},{npts[2]}"
    sf_name = js.extra_info["sf_name"]
    v = Vina(sf_name=sf_name, cpu=js.extra_info["n_cpus"], seed=875234)
    if js.PRO_PDB is None:
        return [None, None, None, None, None, None, None, "PRO_PDBQT is None"]
    if js.LIG_PDB is None:
        return [None, None, None, None, None, None, None, "LIG_PDBQT is None"]

    PRO_PDBQT = js.PRO_PDB + "qt"
    LIG_PDBQT = js.LIG_PDB + "qt"
    # WAT_PDBQT = js.WAT_PDB + "qt"
    # OTH_PDBQT = js.OTH_PDB + "qt"
    PRO = PRO_PDBQT.replace(".pdbqt", "")
    def_dir = os.getcwd()
    try:
        if verbose:
            print(PRO_PDBQT, LIG_PDBQT, sep="\n")
        if not os.path.exists(PRO_PDBQT):
            docking_tools_amw.prepare_receptor4.prepare_receptor4(
                receptor_filename=js.PRO_PDB,
                outputfilename=PRO_PDBQT,
                # charges_to_add=None,
            )
        # if not os.path.exists(LIG_PDBQT):
        if os.path.exists(LIG_PDBQT):
            docking_tools_amw.prepare_ligand4.prepare_ligand4(
                ligand_filename=js.LIG_PDB,
                outputfilename=LIG_PDBQT,
                repairs="",
                verbose=True,
            )
        return
        u = mda.Universe(js.LIG_PDB)
        print(u.atoms.positions)
        print(u.atoms.elements)
        # find the center of the binding pocket, for this dataset that is also the center of mass of the ligand
        com = get_com(js.LIG_PDB)
        # set the ligand
        ad_vina_errors = None
        # if vina or vinardo then set the receptor and computer the vina maps, if autodock then prepare the gpf and autogrid
        # NOTE: receptor must be set before ligand
        if sf_name in ["vina", "vinardo"]:
            print(sf_name)
            v.set_receptor(PRO_PDBQT)
            v.compute_vina_maps(center=com, box_size=box_size)
        elif sf_name == "ad4":
            os.chdir("/".join(PRO.split("/")[:-1]))
            PRO_MAP = PRO.split("/")[-1]
            PRO = PRO_MAP + identifier

            PRO_GPF = f"{PRO}.gpf"
            PRO_GLG = f"{PRO}.glg"
            print(f"{PRO_GPF = }")
            PRO_PDBQT = PRO_PDBQT.split("/")[-1]
            LIG_PDBQT = LIG_PDBQT.split("/")[-1]
            docking_tools_amw.prepare_gpf.prepare_gpf(
                receptor_filename=PRO_PDBQT,
                ligand_filename=LIG_PDBQT,
                output_gpf_filename=PRO_GPF,
                parameters=[npts_param],
                identifier=identifier,
            )
            cmd = f"autogrid4 -p {PRO_GPF} -l {PRO_GLG}"
            os.system(cmd)
            v.load_maps(PRO)
        else:
            ad_vina_errors = "invalid sf_name"
        v.set_ligand_from_file(LIG_PDBQT)

        # docking
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        vina_out = LIG_PDBQT.replace(".pdbqt", f"{js.LIG_NAME.replace(':', '_')}_{js.system}_out.pdb")
        v.write_poses(vina_out, n_poses=n_poses, overwrite=True)
        poses = v.poses(n_poses=1, coordinates_only=True)
        print(f"{poses = }")
        energies = v.energies(n_poses=n_poses)

        weighted_energies = []

        if sf_name == "ad4":
            ad4_weights = [0.1662, 0.1209, 0.1406, 0.1322, 50]
            for n, i in enumerate(ad4_weights):
                new_weighs = [0] * 5
                new_weighs[n] = i
                v.set_weights(new_weighs)
                e = v.energies(n_poses=1)
                print(e)
                weighted_energies.append(e)
            cmd = f"rm {PRO_GPF} {PRO_GLG} {PRO}.map*"
            os.system(cmd)
            os.chdir(def_dir)
        elif sf_name == "vina":
            vina_weights = [-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50, 0.05846]
            for n, i in enumerate(vina_weights):
                new_weights = [0] * 7
                new_weights[n] = i
                print(new_weights)
                v.set_weights(new_weights)
                print(v.info())
                e = v.score()
                print(e)
                weighted_energies.append(e)
        elif sf_name == 'vinardo':
            vinardo_weights = [-0.045, 0.8, -0.035, -0.6, 50, 0.05846]
            for n, i in enumerate(vinardo_weights):
                new_weighs = [0] * 6
                new_weighs[n] = i
                v.set_weights(new_weighs)
                e = v.energies(n_poses=1)
                weighted_energies.append(e)
        print(f"{weighted_energies = }")

    except Exception as e:
        ad_vina_errors = str(e)
        print(ad_vina_errors)
        os.chdir(def_dir)
    if ad_vina_errors == None:
        return [
            energies[0][0],
            energies[0][1],
            energies[0][2],
            energies[0][3],
            energies[0][4],
            vina_out,
            energies,
            ad_vina_errors,
        ]
    else:
        return [None, None, None, None, None, None, None, ad_vina_errors]

# return data as a list in the following order (and typing)
# "vina_total__LIG": "REAL",
# "vina_inter__LIG": "REAL",
# "vina_intra__LIG": "REAL",
# "vina_torsion__LIG": "REAL",
# "vina_intra_best_pose__LIG": "REAL",
# "vina_poses_pdbqt__LIG": "TEXT",
# "vina_all_poses__LIG": "array",
# "vina_errors__LIG": "TEXT",
