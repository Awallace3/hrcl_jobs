import sqlite3 as sql
import pandas as pd
from .parallel import ms_sl
from .sqlt import (
    establish_connection,
    convert_df_into_sql,
    convert_df_into_sql_mp,
    collect_rows_into_js_ls_mp,
    create_new_induction_table,
    subset_df,
    collect_rows_index_range,
    select_table_by_column_match,
)
from qm_tools_aw.tools import np_carts_to_string, string_carts_to_np

from mpi4py import MPI
import numpy as np
from .s22 import s22_testing


def tasks() -> None:
    """
    tasks is a playground
    """
    db_p = "db/dimers_all.db"
    con, cur = establish_connection(db_p)
    v = select_table_by_column_match(cur, selection="*", column="rowid", value=1)
    print(v)


def setup_mp(test=False):
    db_p = "db/dimers_all.db"
    convert_df_into_sql_mp("1600K_train_dimers-fixed.pkl", db_p)
    con, cur = establish_connection(db_p)
    if test:
        rows = collect_rows_into_js_ls_mp(cur, mem="4gb", v_range=[2, 5])
        for n, i in enumerate(rows):
            print(n, i)
    return


def setup_mp(
    df_p="data/1600K_train_dimers-fixed.pkl",
    db_p="db/dimers_all.db",
    test=False,
):
    convert_df_into_sql_mp(df_p, db_p)
    con, cur = establish_connection(db_p)
    if test:
        rows = collect_rows_into_js_ls_mp(cur, mem="4gb", v_range=[2, 5])
        for n, i in enumerate(rows):
            print(n, i)
    return


def setup_grimme(test=True) -> None:
    """
    setup_grimme
    """
    input_columns = {
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "Benchmark": "FLOAT",
        "C6s": "array",
        "C6_A": "array",
        "C6_B": "array",
    }
    output_columns = {
        "HF_dz": "FLOAT",
        "HF_jdz": "FLOAT",
        "HF_dz_no_cp": "FLOAT",
        "HF_jdz_no_cp": "FLOAT",
        "HF_adz": "FLOAT",
        "HF_tz": "FLOAT",
        "HF_qz": "FLOAT",
        "HF_qz_no_df": "FLOAT",
        "HF_qz_conv_e_4": "FLOAT",
        "main_id": "INTEGER PRIMARY KEY",
    }
    convert_df_into_sql(
        df_p="./data/grimme_fitset.pkl",
        db_p="db/grimme_fitset.db",
        input_columns=input_columns,
        output_columns=output_columns,
    )
    # con, cur = establish_connection(db_p)
    # if test:
    #     rows = collect_rows_into_js_ls_mp(cur, mem="4gb", v_range=[2, 5])
    #     for n, i in enumerate(rows):
    #         print(n, i)
    return


def test_db_s22(db_p="db/test.db") -> None:
    """
    test_db for saptdft
    """
    input_columns = {
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "charges": "array",
    }
    output_columns = {
        "main_id": "INTEGER PRIMARY KEY",
        "SAPTDFT_adz": "FLOAT",
        "pbe0_adz": "FLOAT",
        "pbe0_adz_cation": "FLOAT",
        "SAPTDFT_jdz": "FLOAT",
        "pbe0_jdz": "FLOAT",
        "pbe0_jdz_cation": "FLOAT",
        "SAPTDFT_atz": "FLOAT",
        "pbe0_atz": "FLOAT",
        "pbe0_atz_cation": "FLOAT",
    }
    geom = np.array(
        [
            [6, -1.551007, -0.114520, 0.000000],
            [1, -1.934259, 0.762503, 0.000000],
            [1, -0.599677, 0.040712, 0.000000],
            [6, 1.350625, 0.111469, 0.000000],
            [1, 1.680398, -0.373741, -0.758561],
            [1, 1.680398, -0.373741, 0.758561],
        ]
    )
    monAs = np.array([0, 1, 2])
    monBs = np.array([3, 4, 5])
    charges = np.array([[0, 1], [0, 1], [0, 1]])
    df = pd.DataFrame(
        {
            "Geometry": [geom],
            "monAs": [monAs],
            "monBs": [monBs],
            "charges": [charges],
        }
    )
    df.to_pickle("data/test.pkl")

    convert_df_into_sql(
        df_p="data/test.pkl",
        db_p=db_p,
        input_columns=input_columns,
        output_columns=output_columns,
    )
    return


def test_db(db_p="db/test.db") -> None:
    """
    test_db for saptdft
    """
    input_columns = {
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "charges": "array",
    }
    output_columns = {
        "main_id": "INTEGER PRIMARY KEY",
        "SAPTDFT_adz": "FLOAT",
        "pbe0_adz": "FLOAT",
        "pbe0_adz_cation": "FLOAT",
        "SAPTDFT_jdz": "FLOAT",
        "pbe0_jdz": "FLOAT",
        "pbe0_jdz_cation": "FLOAT",
        "SAPTDFT_atz": "FLOAT",
        "pbe0_atz": "FLOAT",
        "pbe0_atz_cation": "FLOAT",
    }
    geom = np.array(
        [
            [8, -1.551007, -0.114520, 0.000000],
            [1, -1.934259, 0.762503, 0.000000],
            [1, -0.599677, 0.040712, 0.000000],
            [8, 1.350625, 0.111469, 0.000000],
            [1, 1.680398, -0.373741, -0.758561],
            [1, 1.680398, -0.373741, 0.758561],
        ]
    )
    monAs = np.array([0, 1, 2])
    monBs = np.array([3, 4, 5])
    charges = np.array([[0, 1], [0, 1], [0, 1]])
    df = pd.DataFrame(
        {
            "Geometry": [geom],
            "monAs": [monAs],
            "monBs": [monBs],
            "charges": [charges],
        }
    )
    df.to_pickle("data/test.pkl")

    convert_df_into_sql(
        df_p="data/test.pkl",
        db_p=db_p,
        input_columns=input_columns,
        output_columns=output_columns,
    )
    return


def test_db_2(db_p="db/test2.db") -> None:
    """
    test_db for saptdft
    """
    geos = s22_testing()
    geoms = [string_carts_to_np(i) for i in geos]
    d = {
        "monAs": [],
        "monBs": [],
        "Geometry": [],
        "charges": [],
    }
    for i in geoms:
        d["Geometry"].append(i[0])
        d["charges"].append(i[1])
        d["monAs"].append(i[2])
        d["monBs"].append(i[3])
    df = pd.DataFrame(d)
    df.to_pickle("data/test2.pkl")

    input_columns = {
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "charges": "array",
    }
    output_columns = {
        "main_id": "INTEGER PRIMARY KEY",
        "SAPTDFT_adz": "FLOAT",
        "pbe0_adz": "FLOAT",
        "pbe0_adz_cation": "FLOAT",
        "SAPTDFT_jdz": "FLOAT",
        "pbe0_jdz": "FLOAT",
        "pbe0_jdz_cation": "FLOAT",
        "SAPTDFT_atz": "FLOAT",
        "pbe0_atz": "FLOAT",
        "pbe0_atz_cation": "FLOAT",
    }
    convert_df_into_sql(
        df_p="data/test2.pkl",
        db_p=db_p,
        input_columns=input_columns,
        output_columns=output_columns,
    )
    return


def setup_schriber(df_p="data/sr3.pkl", db_p="db/schr.db") -> None:
    """
    setup_schriber
    """
    input_columns = {
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "Benchmark": "FLOAT",
        "C6s": "array",
        "C6_A": "array",
        "C6_B": "array",
        "DB": "text",
        "System": "text",
        "System #": "INTEGER",
        "Benchmark": "FLOAT",
        "SAPT0": "FLOAT",
        "SAPT": "FLOAT",
        "Disp20": "FLOAT",
        "SAPT DISP ENERGY": "FLOAT",
        "SAPT DISP20 ENERGY": "FLOAT",
        "D3Data": "array",
        "Geometry": "array",
        "monAs": "array",
        "monBs": "array",
        "charges": "array",
        "C6s": "array",
        "C6_A": "array",
        "C6_B": "array",
        "C8s": "array",
        "C8_A": "array",
        "C8_B": "array",
        "disp_d": "FLOAT",
        "disp_a": "FLOAT",
        "disp_b": "FLOAT",
        "HF_dz": "FLOAT",
        "HF_jdz": "FLOAT",
        "HF_adz": "FLOAT",
        "HF_tz": "FLOAT",
        "HF_jdz_dftd4": "FLOAT",
        "HF_atz": "FLOAT",
        "HF_jtz": "FLOAT",
    }
    output_columns = {
        "main_id": "INTEGER PRIMARY KEY",
        "SAPTDFT_adz": "FLOAT",
        "pbe0_adz": "FLOAT",
        "pbe0_adz_cation": "FLOAT",
        "SAPTDFT_jdz": "FLOAT",
        "pbe0_jdz": "FLOAT",
        "pbe0_jdz_cation": "FLOAT",
        "SAPTDFT_atz": "FLOAT",
        "pbe0_atz": "FLOAT",
        "pbe0_atz_cation": "FLOAT",
        "pbe0_adz_A": "FLOAT",
        "pbe0_adz_cation_A": "FLOAT",
        "pbe0_adz_grac_A": "FLOAT",
        "pbe0_adz_HOMO_A": "FLOAT",
        "pbe0_adz_B": "FLOAT",
        "pbe0_adz_cation_B": "FLOAT",
        "pbe0_adz_grac_B": "FLOAT",
        "pbe0_adz_HOMO_B": "FLOAT",
        "pbe0_aqz_A": "FLOAT",
        "pbe0_aqz_cation_A": "FLOAT",
        "pbe0_aqz_grac_A": "FLOAT",
        "pbe0_aqz_HOMO_A": "FLOAT",
        "pbe0_aqz_B": "FLOAT",
        "pbe0_aqz_cation_B": "FLOAT",
        "pbe0_aqz_grac_B": "FLOAT",
        "pbe0_aqz_HOMO_B": "FLOAT",
    }
    convert_df_into_sql(
        df_p=df_p,
        db_p=db_p,
        input_columns=input_columns,
        output_columns=output_columns,
    )
    return


def read_example_output(db_path="db/dimers.db", row_range=[0, 1]) -> None:
    """
    read_example_output reads sql row by rowid to verify update
    """
    con, cur = establish_connection(db_path)
    rows = collect_rows_index_range(cur, row_range)
    failures = []
    for n, i in enumerate(rows):
        i = list(i)
        rowid = row_range[0] + n + 1
        print("rowid:", rowid, i[-1])
        if i[-1] == float(0):
            failures.append(rowid)
    print(len(failures))
    print(failures)
    return
