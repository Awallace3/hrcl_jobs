import numpy as np
import pandas as pd
import hrcl_jobs as hrcl
from . import docking_inps
from . import jobspec


def pgsql_op_ad4_vina_apnet(
    psqldb_url, table_name, schema_name, scoring_function, assay, col_check, set_columns, system: str
):
    system_pieces = system.split("_")
    pro_name = system_pieces[0]
    print(pro_name, system)
    apnet_charge = "pro_charge"
    s = f"AND sf.system = ('{system}')"
    if pro_name == "proteinHs" and system.endswith("PQR"):
        pro_pdb_col = f"proteinhs_pdb"
        apnet_charge = pro_pdb_col.replace("pdb", "charge")
    elif pro_name.lower() == "proteinhs":
        pro_pdb_col = f"pro_pdb_hs"
    elif pro_name == 'proteinHsWater':
        pro_pdb_col = "proteinhswater_pdb"
        apnet_charge = pro_pdb_col.replace("pdb", "charge")
    elif pro_name == 'proteinHsWaterOther':
        pro_pdb_col = "proteinhswaterother_pdb"
        apnet_charge = pro_pdb_col.replace("pdb", "charge")
    elif pro_name == 'proteinHsOther':
        pro_pdb_col = "proteinhsother_pdb"
        apnet_charge = pro_pdb_col.replace("pdb", "charge")
    else:
        raise ValueError(f"system {system} not recognized")
    print(f"Using {pro_pdb_col} column for pl query...")


    if scoring_function in ['vina', 'vinardo', 'ad4']:
        init_query_cmd=f"""
        SELECT sf.{scoring_function}_id FROM {schema_name}.{table_name} sf
        JOIN {schema_name}.protein_ligand__{table_name} plsf
            ON plsf.{scoring_function}_id = sf.{scoring_function}_id
        JOIN {schema_name}.protein_ligand pl
            ON plsf.pl_id = pl.pl_id
        WHERE pl.assay = ('{assay}')
            {s}
            AND {col_check} IS NULL
            AND pl.{pro_pdb_col} IS NOT NULL
            AND pl.lig_pdb IS NOT NULL
            ;
        """
        job_query_cmd = f"""
        SELECT sf.{scoring_function}_id, pl.{pro_pdb_col}, pl.lig_pdb, pl.lig_name, sf.system FROM {schema_name}.{table_name} sf
            JOIN {schema_name}.protein_ligand__{table_name} plsf
                ON plsf.{scoring_function}_id = sf.{scoring_function}_id
            JOIN {schema_name}.protein_ligand pl
                ON plsf.pl_id = pl.pl_id
            WHERE sf.{scoring_function}_id = %s;
        """ 
    elif scoring_function == 'apnet':
        init_query_cmd=f"""
        SELECT sf.{scoring_function}_id FROM {schema_name}.{table_name} sf
            JOIN {schema_name}.protein_ligand__{table_name} plsf
                ON plsf.{scoring_function}_id = sf.{scoring_function}_id
            JOIN {schema_name}.protein_ligand pl
                ON plsf.pl_id = pl.pl_id
            WHERE pl.assay = ('{assay}')
                AND {col_check} IS NULL 
                AND pl.{apnet_charge} IS NOT NULL
                AND pl.lig_charge IS NOT NULL
                AND pl.{pro_pdb_col} IS NOT NULL
                AND pl.lig_pdb IS NOT NULL
                {s}
                AND sf.apnet_errors is NULL
                ;
        """
        job_query_cmd = f"""
        SELECT sf.{scoring_function}_id, pl.{pro_pdb_col}, pl.lig_pdb, pl.{apnet_charge}, pl.lig_charge, sf.system
            FROM {schema_name}.{table_name} sf
            JOIN {schema_name}.protein_ligand__{table_name} plsf
                ON plsf.{scoring_function}_id = sf.{scoring_function}_id
            JOIN {schema_name}.protein_ligand pl
                ON plsf.pl_id = pl.pl_id
            WHERE sf.{scoring_function}_id = %s
        """ 
    else:
        raise ValueError(f"scoring function {scoring_function} not recognized")

    return hrcl.pgsql.pgsql_operations(
        pgsql_url=psqldb_url,
        table_name=table_name,
        schema_name=schema_name,
        init_query_cmd=init_query_cmd,
        job_query_cmd=job_query_cmd,
        update_cmd=f"""
        UPDATE {schema_name}.{table_name} sf 
        SET {set_columns}
        WHERE {scoring_function}_id = %s
        ;
        """,
    )


def dataset_ad4_vina_apnet(
    psqldb_url=hrcl.pgsql.psqldb,
    schema_name="bmoad",
    table_name="vina",
    col_check="vina_total",
    assay="Kd",
    system="proteinHs_ligand",
    hex=False,
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
    extra_info["mem_per_process"] = memory_per_thread
    extra_info["n_cpus"] = num_omp_threads

    # JOBS
    if scoring_function in ["vina", "vinardo"]:
        output_columns = [
            f"{scoring_function}_total",
            f"{scoring_function}_inter",
            f"{scoring_function}_intra",
            f"{scoring_function}_torsion",
            f"{scoring_function}_intra_best_pose",
            f"{scoring_function}_poses",
            f"{scoring_function}_all_poses",
            f"{scoring_function}_errors",
        ]
        js_obj = jobspec.autodock_vina_js
        run_js_job = docking_inps.run_autodock_vina
    elif scoring_function == "ad4":
        output_columns = [
            f"{scoring_function}_total",
            f"{scoring_function}_inter",
            f"{scoring_function}_intra",
            f"{scoring_function}_torsion",
            f"{scoring_function}_minus_intra",
            f"{scoring_function}_poses",
            f"{scoring_function}_all_poses",
            f"{scoring_function}_errors",
        ]
        js_obj = jobspec.autodock_vina_js
        run_js_job = docking_inps.run_autodock_vina
    elif scoring_function == "apnet":
        output_columns = [
            f"{scoring_function}_total",
            f"{scoring_function}_elst",
            f"{scoring_function}_exch",
            f"{scoring_function}_indu",
            f"{scoring_function}_disp",
            f"{scoring_function}_errors",
        ]
        extra_info['atom_max'] = 15000
        js_obj = jobspec.apnet_pdbs_js
        run_js_job = docking_inps.run_apnet_pdbs
    else:
        print("scoring function not recognized")
        return
    print(f"{output_columns = }")
    allowed_table_names = [
        "vina",
        "vinardo",
        "ad4",
        "apnet",
    ]
    allowed_schemas = [
        "disco_docking",
        "bmoad",
    ]
    if not parallel or rank == 0:
        con, cur = hrcl.pgsql.connect(psqldb_url)
    if table_name not in allowed_table_names:
        print(f"table_name must be one of {allowed_table_names}")
        return
    if schema_name not in allowed_schemas:
        print(f"schema_name must be one of {allowed_schemas}")
        return
    set_columns = ", ".join([f"{i} = %s" for i in output_columns])

    if not parallel:
        mode = hrcl.serial
        pgsql_op = pgsql_op_ad4_vina_apnet(
            psqldb_url, table_name, schema_name, scoring_function, assay,
            col_check, set_columns, system
        )
        extra_info['identifier'] = 0
        query = pgsql_op.init_query(con, assay)
        query = [i[0] for i in query]
        print(f"Total number of jobs: {len(query)}, printing first 10")
        print(query[:10])
    else:
        mode = hrcl.parallel
        extra_info['identifier'] = rank
        if rank == 0:
            pgsql_op = pgsql_op_ad4_vina_apnet(
                psqldb_url, table_name, schema_name, scoring_function, assay,
                col_check, set_columns, system
            )
            query = pgsql_op.init_query(con, assay)
            query = [i[0] for i in query]
            print(f"Total number of jobs: {len(query)}, printing first 10")
            print(query[:10])
        else:
            pgsql_op = None
            query = None

    mode.ms_sl_extra_info_pg(
        pgsql_op=pgsql_op,
        id_list=query,
        js_obj=js_obj,
        run_js_job=run_js_job,
        extra_info=extra_info,
        print_insertion=True,
    )
    return
