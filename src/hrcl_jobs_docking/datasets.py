from dataclasses import dataclass
import numpy as np
import pandas as pd
import hrcl_jobs as hrcl
from . import docking_inps
from . import jobspec

HIVE_PARAMS = {
    "mem_per_process": "60 gb",
    "num_omp_threads": 8,
}


def apnet_disco_dataset(
    db_path,
    table_name,
    col_check="apnet_totl__LIG",
    assay="KD",
    hex=False,
    check_apnet_errors=False,
    extra_info={},
    hive_params={
        "mem_per_process": "24 gb",
        "num_omp_threads": 4,
    },
    parallel=True,
):
    if hex:
        machine = hrcl.utils.machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]
    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f"{rank = } {memory_per_thread = } ")

    output_columns = [col_check]
    suffix = col_check.split("__")[1:]
    for i in ["apnet_elst", "apnet_exch", "apnet_indu", "apnet_disp"]:
        output_columns.append(i + "__" + "__".join(suffix))
    output_columns.append("apnet_errors")
    print(output_columns)

    matches = {
        col_check: ["NULL"],
        "Assay": [assay],
        "PRO_PDB_Hs": ["NOT NULL"],
    }
    if check_apnet_errors:
        matches["apnet_errors"] = ["NULL"]

    con, cur = hrcl.sqlt.establish_connection(db_path)
    query = hrcl.sqlt.query_columns_for_values(
        cur,
        table_name,
        id_names=["id"],
        matches=matches,
    )
    extra_info['n_cpus'] = num_omp_threads

    if not parallel:
        mode = hrcl.serial
    else:
        mode = hrcl.parallel
    mode.ms_sl_extra_info(
        id_list=query,
        db_path=db_path,
        table_name=table_name,
        js_obj=jobspec.apnet_disco_js,
        headers_sql=jobspec.apnet_disco_js_headers(),
        run_js_job=docking_inps.run_apnet_discos,
        extra_info=extra_info,
        ppm=memory_per_thread,
        id_label="id",
        output_columns=output_columns,
        print_insertion=True,
    )
    return


def vina_api_disco_dataset(
    db_path,
    table_name,
    col_check="vina_total__LIG",
    assay="KD",
    hex=False,
    check_apnet_errors=False,
    scoring_function="vina",
    extra_info={
        "sf_params": {
            "exhaustiveness": 32,
            "n_poses": 10,
            "npts": [54, 54, 54],
        },
    },
    hive_params={
        "mem_per_process": "24 gb",
        "num_omp_threads": 4,
    },
    parallel=True,
):
    print("Starting vina docking...")
    if hex:
        machine = hrcl.utils.machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]
    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print(f"{rank = } {memory_per_thread = } ")

    suffix = "".join(col_check.split("__")[1:])
    print(suffix)
    if scoring_function in ["vina", 'vinardo']:
        output_columns = [
            f"{scoring_function}_total__{suffix}",
            f"{scoring_function}_inter__{suffix}",
            f"{scoring_function}_intra__{suffix}",
            f"{scoring_function}_torsion__{suffix}",
            f"{scoring_function}_intra_best_pose__{suffix}",
            f"{scoring_function}_poses_pdbqt__{suffix}",
            f"{scoring_function}_all_poses__{suffix}",
            f"{scoring_function}_errors__{suffix}",
        ]
    elif scoring_function == "ad4":
        output_columns = [
            f"{scoring_function}_total__{suffix}",
            f"{scoring_function}_inter__{suffix}",
            f"{scoring_function}_intra__{suffix}",
            f"{scoring_function}_torsion__{suffix}",
            f"{scoring_function}_minus_intra__{suffix}",
            f"{scoring_function}_poses_pdbqt__{suffix}",
            f"{scoring_function}_all_poses__{suffix}",
            f"{scoring_function}_errors__{suffix}",
        ]
    else:
        print("scoring function not recognized")
        return
    print(f"{output_columns = }")

    matches = {
        col_check: ["NULL"],
        "Assay": [assay],
    }

    if check_apnet_errors:
        matches[f"{scoring_function}_errors_{suffix}"] = ["NULL"]

    print(f"Connecting to {db_path}:{table_name}...")

    con, cur = hrcl.sqlt.establish_connection(db_path)
    query = hrcl.sqlt.query_columns_for_values(
        cur,
        table_name,
        id_names=["id"],
        matches=matches,
    )

    extra_info["sf_name"] = scoring_function
    extra_info['n_cpus'] = num_omp_threads
    # query = [7916 ]
    # print(query)
    # query = [query[0]]
    print(f"Total number of jobs: {len(query)}")

    if not parallel:
        mode = hrcl.serial
    else:
        mode = hrcl.parallel
    mode.ms_sl_extra_info(
        id_list=query,
        db_path=db_path,
        table_name=table_name,
        js_obj=jobspec.autodock_vina_disco_js,
        headers_sql=jobspec.autodock_vina_disco_js_headers(),
        run_js_job=docking_inps.run_autodock_vina,
        extra_info=extra_info,
        ppm=memory_per_thread,
        id_label="id",
        output_columns=output_columns,
        print_insertion=True,
    )
    return
