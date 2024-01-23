import json
import subprocess
import time
import os
from . import sqlt
from . import parallel
import hrcl_jobs_psi4 as hrcl_psi4
import numpy as np
import pandas as pd
from qm_tools_aw import tools
from pprint import pprint as pp
from mpi4py import MPI
from glob import glob

HIVE_PARAMS = {
    "mem_per_process": "60 gb",
    "num_omp_threads": 8,
}


def machine_list_resources(rank_0_one_thread=True) -> []:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    machines = {
        "ds2": parallel.machineResources("ds2", 9, 9, 80),
        "hex6": parallel.machineResources("hex6", 6, 6, 62),
        "hex8": parallel.machineResources("hex8", 6, 6, 62),
        "hex9": parallel.machineResources("hex9", 6, 6, 58),
        "hex11": parallel.machineResources("hex11", 6, 6, 62),
    }
    uname_n = subprocess.check_output("uname -n", shell=True).decode("utf-8").strip()

    machine = machines[uname_n]

    name = machine.name
    cores = machine.cores
    threads = machine.threads
    memory = machine.memory

    unames = comm.allgather(uname_n)
    current_machine_cnt = 0
    start_rank = 0
    start = True
    for n, i in enumerate(unames):
        if start and i == uname_n:
            current_machine_cnt += 1
            end_rank = n
            continue
        elif i == uname_n:
            current_machine_cnt += 1
            start == True
            start_rank = n

    if current_machine_cnt == 0:
        raise ValueError("No machines found")

    on_rank_0 = False
    if uname_n == unames[0]:
        on_rank_0 = True

    if rank_0_one_thread and on_rank_0:
        threads -= 1
        current_machine_cnt -= 1
        memory -= 4

    evenly_divided_omp = threads // current_machine_cnt
    remainder_omp = threads % current_machine_cnt
    if end_rank - rank < remainder_omp:
        omp_threads = evenly_divided_omp + 1
        marked_for_more_mem = True
    else:
        omp_threads = evenly_divided_omp
        marked_for_more_mem = False

    if rank == 0:
        machine.omp_threads = 1
        machine.memory_per_thread = 4
    else:
        machine.omp_threads = omp_threads
        machine.memory_per_thread = int(memory * (omp_threads / threads))
    if rank_0_one_thread and on_rank_0:
        threads += 1
        memory += 4
    comm.barrier()
    print(
        f"rank {rank} is using {machine.name} with {machine.omp_threads} / {threads} and {machine.memory_per_thread}/{machine.memory} GB"
    )
    return machine


def random_percentage_of_array(arr, percentage):
    # Calculate the number of elements to select
    num_elements = int(len(arr) * percentage / 100)

    # Use np.random.choice to select elements without replacement
    selected_elements = np.random.choice(arr, num_elements, replace=False)

    return selected_elements


def compute_MBIS(
    DB_NAME,
    TABLE_NAME,
    col_check="MBIS_hf_adz",
    hex=True,
    hive_params=HIVE_PARAMS,
    TESTING=False,
) -> None:
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]
    method, basis_str = hrcl_psi4.get_level_of_theory(col_check)
    basis = col_check.split("_")[-1]
    table_cols = {
        f"MBIS_{method}_multipoles_d_{basis}": "FLOAT",
        f"MBIS_{method}_multipoles_a_{basis}": "FLOAT",
        f"MBIS_{method}_multipoles_b_{basis}": "FLOAT",
        f"MBIS_{method}_widths_d_{basis}": "array",
        f"MBIS_{method}_widths_a_{basis}": "array",
        f"MBIS_{method}_widths_b_{basis}": "array",
        f"MBIS_{method}_vol_ratio_d_{basis}": "array",
        f"MBIS_{method}_vol_ratio_a_{basis}": "array",
        f"MBIS_{method}_vol_ratio_b_{basis}": "array",
        f"MBIS_{method}_radial_2_d_{basis}": "array",
        f"MBIS_{method}_radial_2_a_{basis}": "array",
        f"MBIS_{method}_radial_2_b_{basis}": "array",
        f"MBIS_{method}_radial_3_d_{basis}": "array",
        f"MBIS_{method}_radial_3_a_{basis}": "array",
        f"MBIS_{method}_radial_3_b_{basis}": "array",
        f"MBIS_{method}_radial_4_d_{basis}": "array",
        f"MBIS_{method}_radial_4_a_{basis}": "array",
        f"MBIS_{method}_radial_4_b_{basis}": "array",
        f"MBIS_{method}_populations_d_{basis}": "array",
        f"MBIS_{method}_populations_a_{basis}": "array",
        f"MBIS_{method}_populations_b_{basis}": "array",
    }
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"{rank = } {memory_per_thread = } ")
    if rank == 0:
        sqlt.create_update_table(DB_NAME, TABLE_NAME, table_cols=table_cols)
    col_check_MBIS = f"MBIS_{method}_widths_a_{basis}"
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]

    con, cur = sqlt.establish_connection(DB_NAME)
    mbis_ids = sqlt.collect_ids_for_parallel(
        DB_NAME,
        TABLE_NAME,
        col_check=[col_check_MBIS, "array"],
        matches={
            col_check_MBIS: ["NULL"],
        },
        ascending=not TESTING,
        sort_column="Geometry",
    )

    options = {
        "basis": basis_str,
        "E_CONVERGENCE": 8,
        "D_CONVERGENCE": 8,
    }
    xtra_mbis = {
        "options": options,
        "num_threads": num_omp_threads,
        "level_theory": [f"{method}/{basis_str}"],
        "out": {
            "path": "schr",
            "version": "1",
        },
    }
    print(xtra_mbis)
    parallel.ms_sl_extra_info(
        id_list=mbis_ids,
        db_path=DB_NAME,
        table_name=TABLE_NAME,
        js_obj=hrcl_psi4.jobspec.sapt0_js,
        headers_sql=hrcl_psi4.jobspec.sapt0_js_headers(),
        run_js_job=hrcl_psi4.psi4_inps.run_MBIS,
        extra_info=xtra_mbis,
        ppm=memory_per_thread,
        id_label="id",
        output_columns=table_cols.keys(),
        print_insertion=True,
    )
    return


def compute_MBIS_atom(
    DB_NAME,
    TABLE_NAME,
    col_check="MBIS_hf_adz",
    hex=True,
    hive_params=HIVE_PARAMS,
    TESTING=False,
) -> None:
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]
    method, basis_str = hrcl_psi4.get_level_of_theory(col_check)
    basis = col_check.split("_")[-1]
    table_cols = {
        f"MBIS_{method}_multipoles_{basis}": "FLOAT",
        f"MBIS_{method}_widths_{basis}": "array",
        f"MBIS_{method}_vol_ratio_{basis}": "array",
        f"MBIS_{method}_radial_2_{basis}": "array",
        f"MBIS_{method}_radial_3_{basis}": "array",
        f"MBIS_{method}_radial_4_{basis}": "array",
        f"MBIS_{method}_populations_{basis}": "array",
    }
    col_check_MBIS = f"MBIS_{method}_radial_2_{basis}"
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"{rank = } {memory_per_thread = } ")
    if rank == 0:
        sqlt.create_update_table(DB_NAME, TABLE_NAME, table_cols=table_cols)
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]

    con, cur = sqlt.establish_connection(DB_NAME)
    mbis_ids = sqlt.collect_ids_for_parallel(
        DB_NAME,
        TABLE_NAME,
        col_check=[col_check_MBIS, "array"],
        matches={
            col_check_MBIS: ["NULL"],
        },
        ascending=not TESTING,
        sort_column="Geometry",
    )

    options = {
        "basis": basis_str,
        "E_CONVERGENCE": 8,
        "D_CONVERGENCE": 8,
    }
    xtra_mbis = {
        "options": options,
        "num_threads": num_omp_threads,
        "level_theory": [f"{method}/{basis_str}"],
        "out": {
            "path": "atomic_props",
            "version": "1",
        },
    }
    print(xtra_mbis)
    parallel.ms_sl_extra_info(
        id_list=mbis_ids,
        db_path=DB_NAME,
        table_name=TABLE_NAME,
        js_obj=hrcl_psi4.jobspec.monomer_js,
        headers_sql=hrcl_psi4.jobspec.monomer_js_headers(),
        run_js_job=hrcl_psi4.psi4_inps.run_MBIS_monomer,
        extra_info=xtra_mbis,
        ppm=memory_per_thread,
        id_label="id",
        output_columns=table_cols.keys(),
        print_insertion=True,
    )
    return


def compute_energy(
    DB_NAME,
    TABLE_NAME,
    col_check="SAPT0_adz",
    hex=True,
    hive_params=HIVE_PARAMS,
    TESTING=False,
    options=None,
    xtra=None,
    output_root="schr",
) -> None:
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]
    method, basis_str = hrcl_psi4.get_level_of_theory(col_check)
    js_obj, js_headers, run_js_job = hrcl_psi4.get_parallel_functions(method)
    table_cols, col_check = hrcl_psi4.get_col_check(method, col_check.split("_")[-1])
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"{rank = } {memory_per_thread = } ")
    if rank == 0:
        sqlt.create_update_table(DB_NAME, TABLE_NAME, table_cols=table_cols)
    if hex:
        machine = machine_list_resources()
        memory_per_thread = f"{machine.memory_per_thread} gb"
        num_omp_threads = machine.omp_threads
    else:
        memory_per_thread = hive_params["mem_per_process"]
        num_omp_threads = hive_params["num_omp_threads"]

    con, cur = sqlt.establish_connection(DB_NAME)
    job_ids = sqlt.collect_ids_for_parallel(
        DB_NAME,
        TABLE_NAME,
        col_check=[col_check, "array"],
        matches={
            col_check: ["NULL"],
        },
        # ascending=not TESTING,
        ascending=TESTING,
        sort_column="Geometry",
    )
    if options is None:
        options = {
            "basis": basis_str,
            "E_CONVERGENCE": 8,
            "D_CONVERGENCE": 8,
        }
    print(f"{options = }")
    if xtra is None:
        xtra= {
            "options": options,
            "num_threads": num_omp_threads,
            "level_theory": [f"{method}/{basis_str}"],
            "out": {
                "path": output_root,
                "version": "1",
            },
        }
    print(f"{xtra = }")
    parallel.ms_sl_extra_info(
        id_list=job_ids,
        db_path=DB_NAME,
        table_name=TABLE_NAME,
        js_obj=js_obj,
        headers_sql=js_headers(),
        run_js_job=run_js_job,
        extra_info=xtra,
        ppm=memory_per_thread,
        id_label="id",
        output_columns=table_cols.keys(),
        print_insertion=True,
    )
    return
