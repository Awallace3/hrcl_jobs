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
    extra_info["n_cpus"] = num_omp_threads

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


def example_queriers():
    cmd_ids = f"""
        SELECT sf.{scoring_function}_id FROM {schema_name}.{table_name} sf
        JOIN {schema_name}.protein_ligand__{table_name} plsf
            ON plsf.pl_id = sf.{scoring_function}_id
        JOIN {schema_name}.protein_ligand pl
            ON plsf.pl_id = pl.pl_id
            WHERE pl.assay = (%s)
            AND {col_check} IS NOT NULL
        ;
    """
    cur.execute(cmd_ids, (assay,))
    query = cur.fetchall()
    id = query[:2]
    print(id)
    for i in query:
        id = i[0]
        cmd_inputs = f"""
        SELECT sf.{scoring_function}_id, pl.pro_pdb, pl.lig_pdb, pl.wat_pdb, pl.oth_pdb FROM {schema_name}.{table_name} sf
            JOIN {schema_name}.protein_ligand__{table_name} plsf
                ON plsf.pl_id = sf.{scoring_function}_id
            JOIN {schema_name}.protein_ligand pl
                ON plsf.pl_id = pl.pl_id
            WHERE sf.{scoring_function}_id = (%s)
                ;
        """
        cur.execute(cmd_inputs, (id,))
        query = cur.fetchall()
        js = jobspec.autodock_vina_disco_js(
            *query[0],
            extra_info,
            "",
        )
        print(js)
        out = docking_inps.run_autodock_vina(js)
        print(out)
        cmd_insert = f"""
        INSERT INTO {schema_name}.{table_name} sf
        ({", ".join(output_columns)}) VALUES 
        ({", ".join(['%s'] * len(output_columns))})
        WHERE {scoring_function}_id = (%s)
        ;
        """
        print(cmd_insert)
        cur.execute(cmd_insert, (*out, id))


def vina_api_disco_dataset(
    # db_path,
    psqldb_url=hrcl.pgsql.psqldb,
    schema_name="bmoad",
    table_name="vina",
    col_check="vina_total",
    assay="Kd",
    hex=False,
    check_errors=True,
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
    """
    Assumes postgresql database with schema_name:
    פּ {schema_name} (11)
         ad4
         apnet
         experiment
         protein_ligand
         protein_ligand__ad4
         protein_ligand__apnet
         protein_ligand__vina
         protein_ligand__vinardo
         vina
         vinardo
    - consult db.py to generate the database
    """
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

    extra_info["sf_name"] = scoring_function
    extra_info["n_cpus"] = num_omp_threads

    if scoring_function in ["vina", "vinardo"]:
        output_columns = [
            f"{scoring_function}_total",
            f"{scoring_function}_inter",
            f"{scoring_function}_intra",
            f"{scoring_function}_torsion",
            f"{scoring_function}_intra_best_pose",
            f"{scoring_function}_poses_pdbqt",
            f"{scoring_function}_all_poses",
            f"{scoring_function}_errors",
        ]
    elif scoring_function == "ad4":
        output_columns = [
            f"{scoring_function}_total",
            f"{scoring_function}_inter",
            f"{scoring_function}_intra",
            f"{scoring_function}_torsion",
            f"{scoring_function}_minus_intra",
            f"{scoring_function}_poses_pdbqt",
            f"{scoring_function}_all_poses",
            f"{scoring_function}_errors",
        ]
    else:
        print("scoring function not recognized")
        return
    print(f"{output_columns = }")

    matches = {
        col_check: ["NULL"],
        "Assay": [assay],
    }

    if check_errors:
        matches[f"{scoring_function}_errors"] = ["NULL"]

    allowed_columns = [
        f"{scoring_function}_total",
        f"{scoring_function}_inter",
        f"{scoring_function}_intra",
        f"{scoring_function}_torsion",
        f"{scoring_function}_intra_best_pose",
        f"{scoring_function}_poses_pdbqt",
        f"{scoring_function}_all_poses",
        f"{scoring_function}_errors",
        f"{scoring_function}_poses_pdbqt",
    ]

    allowed_table_names = [
        "vina",
        "vinardo",
        "ad4",
        # expand to include columns
    ]
    allowed_schemas = [
        "disco_docking",
        'bmoad',
    ]
    # con, cur = hrcl.pgsql.establish_connection(psqldb_info)
    con, cur = hrcl.pgsql.connect(psqldb_url)
    if table_name not in allowed_table_names:
        print(f"table_name must be one of {allowed_table_names}")
        return
    if schema_name not in allowed_schemas:
        print(f"schema_name must be one of {allowed_schemas}")
        return
    if col_check not in allowed_columns:
        print(f"col_check must be one of {allowed_columns}")
        return
    set_columns = ", ".join([f"{i} = %s" for i in output_columns])

    pgsql_op = hrcl.pgsql.pgsql_operations(
        pgsql_url=psqldb_url,
        table_name=table_name,
        schema_name=schema_name,
        init_query_cmd=f"""
        SELECT sf.{scoring_function}_id FROM {schema_name}.{table_name} sf
        JOIN {schema_name}.protein_ligand__{table_name} plsf
            ON plsf.{scoring_function}_id = sf.{scoring_function}_id
        JOIN {schema_name}.protein_ligand pl
            ON plsf.pl_id = pl.pl_id
            WHERE pl.assay = ('{assay}')
            AND {col_check} IS NULL
        ;
    """,
        job_query_cmd=f"""
        SELECT sf.{scoring_function}_id, pl.pro_pdb, pl.lig_pdb, pl.wat_pdb, pl.oth_pdb FROM {schema_name}.{table_name} sf
            JOIN {schema_name}.protein_ligand__{table_name} plsf
                ON plsf.pl_id = sf.{scoring_function}_id
            JOIN {schema_name}.protein_ligand pl
                ON plsf.pl_id = pl.pl_id
            WHERE sf.{scoring_function}_id = (%s)
                ;
        """,
        update_cmd=f"""
        UPDATE {schema_name}.{table_name} sf 
        SET {set_columns}
        WHERE {scoring_function}_id = (%s)
        ;
        """
    )
    query = pgsql_op.init_query(con, assay)
    print(query)

    print(f"Total number of jobs: {len(query)}")

    if not parallel:
        mode = hrcl.serial
    else:
        mode = hrcl.parallel
    mode.ms_sl_extra_info_pg(
        pgsql_op=pgsql_op,
        id_list=query,
        js_obj=jobspec.autodock_vina_disco_js,
        run_js_job=docking_inps.run_autodock_vina,
        extra_info=extra_info,
        id_label="id",
        print_insertion=True,
    )
    return
