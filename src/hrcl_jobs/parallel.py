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


def example_run_js_job(js: example_js) -> float:
    """
    example_run_js_job
    """
    v1 = js.val + 1
    v2 = js.val + 2
    return [v1, v2]

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


def ms_sl_extra_info(
    id_list=[0, 50],
    db_path="db/dimers_all.db",
    collect_ids_into_js_ls=collect_ids_into_js_ls,
    collect_id_into_js=collect_id_into_js,
    run_js_job=example_run_js_job,
    update_func=update_by_id,
    extra_info=[],
    headers_sql=["main_id", "id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
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
                extra_info=extra_info,
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
                extra_info=extra_info,
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
                extra_info=extra_info,
                dataclass_obj=js_obj,
                id_value=active_ind,
                id_label=id_label,
                table=table,
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
                id_label=id_label,
                id_value=id_value,
                table=table,
                output_columns=output_columns,
            )
            i2 = time.time() - i1
            print(f"\nMAIN: id {id_value} inserted\n")
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
                table=table,
                output_columns=output_columns,
            )
            comm.send(0, dest=target_proc, tag=2)
        print("\nCOMPLETED MAIN\n")
        # read_output(db_path, id_list=[id_list[0]], id_label=id_label)

    else:
        start = True
        js = 1
        req = comm.irecv(source=0, tag=2)
        js = req.wait()
        print(f"rank: {rank}, js.main_id: {js.id_label}")
        s = time.time()
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
                output = run_js_job(js)
                output.append(js.id_label)
                output.append(rank)
                comm.send(output, dest=0, tag=2)
                print(f"rank: {rank} TOOK {time.time() - s} seconds")
        print(rank, "TERMINATING")
        return
