from .sqlt import (
    establish_connection,
    update_mp_rows,
    update_rows,
    collect_rows_into_js_ls_mp,
    collect_row_specific_into_js_mp,
    collect_ids_into_js_ls,
    collect_id_into_js,
    update_by_id,
    read_output,
)
from mpi4py import MPI
import os
from glob import glob
import time
from .jobspec import example_js
from . import pgsql


def example_run_js_job(js: example_js) -> float:
    """
    example_run_js_job
    """
    v1 = js.val + 1
    v2 = js.val + 2
    return [v1, v2]


class machineResources:
    def __init__(
        self,
        name: str,
        cores: int,
        threads: int,
        memory: int,
        memory_per_core="4 gb",
        omp_threads: int = 2,
    ):  # GB
        self.name = name
        self.cores = cores
        self.threads = threads
        self.memory = memory  # Total Memory
        self.memory_per_thread = memory_per_core
        self.omp_threads = omp_threads




def ms_sl_extra_info_pg(
    pgsql_op: pgsql.pgsql_operations,
    id_list=[0, 50],
    run_js_job=example_run_js_job,
    extra_info={},  # memory requirements should be set here
    js_obj=example_js,
    print_insertion=False,
):
    """
    To use ms_sl_extra_info_pg properly, write your own run_js_job function
    along with an appropriate js_obj dataclass. This function assumes
    that you have created a postgresql database and have a connection

    To run with multiple procs, use following example
    ```bash
    mpiexec -n 2 python3 -u main.py
    ```
    """
    if id_list is not None and len(id_list) == 0:
        print("No ids to run")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    print("rank", rank)
    if rank == 0:
        jobs = len(id_list)
        start = time.time()
        first = True
        conn, cur = pgsql_op.connect_db()
        cur_rows = []
        id_list_first = id_list[: n_procs - 1]
        if len(id_list) == 1 or n_procs == 2:
            js = pgsql_op.job_query(conn, id_list[0], js_obj, extra_info)
            r = [js]
        else:
            r = pgsql_op.job_query(conn, id_list_first, js_obj, extra_info)

        for n, js in enumerate(r):
            n = n + 1
            req = comm.isend(js, dest=n, tag=2)
            req.wait()
            print(f"{n} / {jobs}")

        diff = len(r) - (n_procs - 1)
        if diff < 0:
            print(
                f"WARNING: n_procs ({n_procs}) > len(id_list) ({len(id_list)}), reducing n_procs to {len(r) + 1}"
            )
            for n in range(len(r) + 1, n_procs):
                req = comm.isend(0, dest=n, tag=2)
                req.wait()
            n_procs = len(r) + 1

        id_list_extra = id_list[len(r) :]
        for n, active_ind in enumerate(id_list_extra):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            target_proc = output.pop()
            id_value = output.pop()
            js = pgsql_op.job_query(conn, active_ind, js_obj, extra_info)
            comm.send(js, dest=target_proc, tag=2)
            i1 = time.time()
            print(f"MAIN: {n + n_procs - 1} / {jobs}")
            p = glob("psi.*.clean")
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            pgsql_op.update(conn, output, id_value)
            i2 = time.time() - i1
            insertion_str = ""
            if print_insertion:
                insertion_str = f", output={output}"
            print(f"\nMAIN: id {id_value} inserted{insertion_str}\n")
        print("\nMAIN CLEANING UP PROCESSES\n")
        for n in range(n_procs - 1):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            p = glob("psi.*.clean")
            target_proc = output.pop()
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            id_value = output.pop()
            pgsql_op.update(conn, output, id_value)
            comm.send(0, dest=target_proc, tag=2)
            insertion_str = ""
            if print_insertion:
                insertion_str = f", output={output}"
            print(f"\nMAIN: id {id_value} inserted{insertion_str}\n")
        print("\nCOMPLETED MAIN\n")
    else:
        js = 1
        req = comm.irecv(source=0, tag=2)
        js = req.wait()
        if js == 0:
            print(f"rank: {rank} TERMINATING")
            return
        if isinstance(js, list):
            print(f"rank: {rank}, js.main_id: {js.id_label}")
            raise ValueError("js is a list")
        print(f"rank: {rank}")
        s = time.time()
        js.extra_info = extra_info
        output = run_js_job(js)
        output.append(js.id_label)
        output.append(rank)
        comm.send(output, dest=0, tag=2)
        print(f"rank: {rank} TOOK {time.time() - s} seconds")
        while js != 0:
            s = time.time()
            js = comm.recv(source=0, tag=2)
            if js != 0:
                print(f"rank: {rank}")
                js.extra_info = extra_info
                output = run_js_job(js)
                output.append(js.id_label)
                output.append(rank)
                comm.send(output, dest=0, tag=2)
                print(f"rank: {rank} spent {time.time() - s} seconds")
        print(rank, "TERMINATING")
        return


def ms_sl_extra_info(
    id_list=[0, 50],
    db_path="db/dimers_all.db",
    run_js_job=example_run_js_job,
    update_func=update_by_id,
    extra_info={},
    headers_sql=["main_id", "id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    js_obj=example_js,
    ppm="4gb",
    table_name="main",
    id_label="id",
    output_columns=[
        "env_multipole_A",
        "env_multipole_B",
        "vac_widths_A",
        "vac_widths_B",
        "vac_vol_rat_A",
        "vac_vol_rat_B",
    ],
    print_insertion=False,
):
    """
    To use ms_sl properly, write your own run_js_job function along with an
    appropriate js_obj dataclass.

    This was designed to work with psi4 jobs using the python api; however,
    any user defined function for workers will work.

    To run with multiple procs, use following example
    ```bash
    mpiexec -n 2 python3 -u main.py
    ```
    """
    if len(id_list) == 0:
        print("No ids to run")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    print("rank", rank)
    if rank == 0:
        jobs = len(id_list)
        start = time.time()
        first = True
        con, cur = establish_connection(db_p=db_path)
        cur_rows = []
        id_list_first = id_list[: n_procs - 1]
        if len(id_list) == 1 or n_procs == 2:
            js = collect_id_into_js(
                cur,
                headers_sql,
                ppm,
                extra_info,
                js_obj,
                id_list[0],
                id_label,
                table_name,
            )
            r = [js]

        else:
            r = collect_ids_into_js_ls(
                cur,
                headers_sql,
                ppm,
                extra_info,
                js_obj,
                id_list_first,
                id_label,
                table_name,
            )

        for n, js in enumerate(r):
            n = n + 1
            req = comm.isend(js, dest=n, tag=2)
            req.wait()
            print(f"{n} / {jobs}")

        diff = len(r) - (n_procs - 1)
        if diff < 0:
            print(
                f"WARNING: n_procs ({n_procs}) > len(id_list) ({len(id_list)}), reducing n_procs to {len(r) + 1}"
            )
            for n in range(len(r) + 1, n_procs):
                req = comm.isend(0, dest=n, tag=2)
                req.wait()
            n_procs = len(r) + 1

        id_list_extra = id_list[len(r) :]
        # active_ind = jobs + n_procs - 1
        # while active_ind <= jobs:
        for n, active_ind in enumerate(id_list_extra):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            target_proc = output.pop()
            id_value = output.pop()
            js = collect_id_into_js(
                cur,
                headers_sql,
                ppm,
                extra_info,
                js_obj,
                active_ind,
                id_label,
                table_name,
            )
            comm.send(js, dest=target_proc, tag=2)
            i1 = time.time()
            print(f"MAIN: {n + n_procs - 1} / {jobs}")
            p = glob("psi.*.clean")
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            update_func(
                con,
                cur,
                output,
                id_value=id_value,
                id_label=id_label,
                table=table_name,
                output_columns=output_columns,
            )
            i2 = time.time() - i1
            insertion_str = ""
            if print_insertion:
                insertion_str = f", output={output}"
            print(f"\nMAIN: id {id_value} inserted{insertion_str}\n")
        print("\nMAIN CLEANING UP PROCESSES\n")
        for n in range(n_procs - 1):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            p = glob("psi.*.clean")
            target_proc = output.pop()
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            id_value = output.pop()
            update_func(
                con,
                cur,
                output,
                id_label=id_label,
                id_value=id_value,
                table=table_name,
                output_columns=output_columns,
            )
            comm.send(0, dest=target_proc, tag=2)
            insertion_str = ""
            if print_insertion:
                insertion_str = f", output={output}"
            print(f"\nMAIN: id {id_value} inserted{insertion_str}\n")
        print("\nCOMPLETED MAIN\n")

    else:
        js = 1
        req = comm.irecv(source=0, tag=2)
        js = req.wait()
        if js == 0:
            print(f"rank: {rank} TERMINATING")
            return
        print(f"rank: {rank}, js.main_id: {js.id_label}")
        s = time.time()
        js.extra_info = extra_info
        js.mem = ppm
        output = run_js_job(js)
        output.append(js.id_label)
        output.append(rank)
        comm.send(output, dest=0, tag=2)
        print(f"rank: {rank} TOOK {time.time() - s} seconds")
        while js != 0:
            # print(f"rank: {rank} is BLOCKED")
            s = time.time()
            js = comm.recv(source=0, tag=2)
            if js != 0:
                print(f"rank: {rank}, js.main_id: {js.id_label}")
                js.extra_info = extra_info
                js.mem = ppm
                output = run_js_job(js)
                output.append(js.id_label)
                output.append(rank)
                comm.send(output, dest=0, tag=2)
                print(f"rank: {rank} spent {time.time() - s} seconds on {js.id_label}")
        print(rank, "TERMINATING")
        return


# READS n_procs from comm now
def ms_sl(
    id_list=[0, 50],
    db_path="db/dimers_all.db",
    collect_ids_into_js_ls=collect_ids_into_js_ls,
    collect_id_into_js=collect_id_into_js,
    run_js_job=example_run_js_job,
    update_func=update_by_id,
    headers_sql=["main_id", "id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    level_theory=["hf/aug-cc-pV(D+d)Z"],
    js_obj=example_js,
    ppm="4gb",
    table="main",
    id_label="main_id",
    output_columns=[
        "env_multipole_A",
        "env_multipole_B",
        "vac_widths_A",
        "vac_widths_B",
        "vac_vol_rat_A",
        "vac_vol_rat_B",
    ],
):
    """
    To use ms_sl properly, write your own run_js_job function along with an
    appropriate js_obj dataclass.

    This was designed to work with psi4 jobs using the python api; however,
    any user defined function for workers will work.

    To run with multiple procs, use following example
    ```bash
    mpiexec -n 2 python3 -u main.py
    ```
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()
    print("rank", rank)
    if rank == 0:
        jobs = len(id_list)
        start = time.time()
        first = True
        con, cur = establish_connection(db_p=db_path)
        cur_rows = []
        id_list_first = id_list[: n_procs - 1]
        if len(id_list) == 1 or n_procs == 2:
            js = collect_id_into_js(
                cur,
                mem=ppm,
                headers=headers_sql,
                extra_info=level_theory,
                dataclass_obj=js_obj,
                id_value=id_list[0],
                id_label=id_label,
                table=table,
            )
            r = [js]

        else:
            r = collect_ids_into_js_ls(
                cur,
                mem=ppm,
                headers=headers_sql,
                extra_info=level_theory,
                dataclass_obj=js_obj,
                id_list=id_list_first,
                id_label=id_label,
                table=table,
            )
        for n, js in enumerate(r):
            n = n + 1
            req = comm.isend(js, dest=n, tag=2)
            req.wait()
            print(f"{n} / {jobs}")
        id_list_extra = id_list[len(r) :]
        # active_ind = jobs + n_procs - 1
        # while active_ind <= jobs:
        for n, active_ind in enumerate(id_list_extra):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            target_proc = output.pop()
            id_value = output.pop()
            js = collect_id_into_js(
                cur,
                mem=ppm,
                headers=headers_sql,
                extra_info=level_theory,
                dataclass_obj=js_obj,
                id_value=active_ind,
                id_label=id_label,
                table=table,
            )
            comm.send(js, dest=target_proc, tag=2)
            i1 = time.time()
            print(f"main thread: {n + n_procs - 1} / {jobs}")
            p = glob("psi.*.clean")
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            update_func(
                con,
                cur,
                output,
                id_label=id_label,
                id_value=id_value,
                table=table,
                output_columns=output_columns,
            )
            i2 = time.time() - i1
            print(f"main thread inserted: {id_value} in {i2} seconds")
        print("MAIN CLEANING UP PROCESSES")
        for n in range(n_procs - 1):
            output = comm.recv(source=MPI.ANY_SOURCE, tag=2)
            p = glob("psi.*.clean")
            target_proc = output.pop()
            for i in p:
                if os.path.exists(i):
                    try:
                        os.remove(i)
                    except FileNotFoundError:
                        continue
            id_value = output.pop()
            update_func(
                con,
                cur,
                output,
                id_label=id_label,
                id_value=id_value,
                table=table,
                output_columns=output_columns,
            )
            comm.send(0, dest=target_proc, tag=2)
        print((time.time() - start) / 60, "Minutes")
        print("COMPLETED MAIN\nDisplaying first two entries in db...")
        # read_output(db_path, id_list=[id_list[0]], id_label=id_label)

    else:
        start = True
        js = 1
        req = comm.irecv(source=0, tag=2)
        js = req.wait()
        print(f"rank: {rank}, js.main_id: {js.id_label}")
        output = run_js_job(js)
        output.append(js.id_label)
        output.append(rank)
        s = time.time()
        comm.send(output, dest=0, tag=2)
        while js != 0:
            # print(f"rank: {rank} is BLOCKED")
            js = comm.recv(source=0, tag=2)
            print(f"rank: {rank} WAITED {time.time() - s} seconds")
            if js != 0:
                print(f"rank: {rank}, js.main_id: {js.id_label}")
                output = run_js_job(js)
                output.append(js.id_label)
                output.append(rank)
                s = time.time()
                comm.send(output, dest=0, tag=2)
        print(rank, "TERMINATING")
        return
