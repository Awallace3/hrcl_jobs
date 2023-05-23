import sqlite3 as sql
import pandas as pd
import numpy as np
import io
import zlib  # default compression is 6 on a ( 0-9 ) scale
from dataclasses import dataclass
from .jobspec import mp_js
import os


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sql.Binary(zlib.compress(out.read()))
    # return sql.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(zlib.decompress(out.read()))
    return np.load(out)


# Converts np.array to TEXT when inserting
sql.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sql.register_converter("array", convert_array)
# sql.register_converter("ARRAY", convert_array)


def establish_connection(
    db_p="db/dimers.db",
) -> (object, object):
    """
    establish_connection
    """
    try:
        con = sql.connect(db_p, detect_types=sql.PARSE_DECLTYPES)
        cur = con.cursor()
        return con, cur
    except sql.OperationalError:
        print("Error with db path. Ensure all directories exist.")
        return None, None


def new_table(
    db_path="db/test.db",
    table_name="new",
    table={
        "id": "PRIMARY KEY",
        "R": "array",
        "out": "array",
    },
):
    conn, cur = establish_connection(db_path)
    headers = ",\n".join([f"{k} {v}" for k, v in table.items()])
    if not conn:
        return
    table_format = f""" CREATE TABLE {table_name} (
            {headers}
            );"""
    return create_table(conn, table_format)


def create_table(conn, create_table_sql):
    """create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sql.OperationalError as e:
        # print("TABLE ALREADY EXISTS. SKIPPING GENERATION\n")
        print(e)
        return False
    return True


def table_add_columns(con: sql.Connection, table_name:str, table_dict: dict,) -> bool:
    """
    table_add_columns insert columns into a table.
    """
    cur = con.execute(f"PRAGMA table_info({table_name});")
    desc = cur.fetchall()
    existing_table = {}
    for i in desc:
        existing_table[i[1]] = i[2]
    for k, v in table_dict.items():
        if k not in existing_table.keys():
            print(f"Adding column {k} to {table_name}")
            con.execute(f"ALTER TABLE {table_name} ADD COLUMN {k} {v};")
    return True



def create_new_db(
    db_name="db/main.db",
    table_name="main",
    table={
        "Dimer": "TEXT",
        "R": "FLOAT",
        "CBS": "FLOAT",
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "Benchmark": "FLOAT",
        "C6s": "array",
        "C6_A": "array",
        "C6_B": "array",
        "HF_dz": "FLOAT",
        "HF_tz": "FLOAT",
    },
) -> None:
    """
    create_new_db creates a new db and creates a new table
    main_id integer PRIMARY KEY
    """
    headers = ",\n".join([f"{k} {v}" for k, v in table.items()])
    con, cur = establish_connection(db_name)
    if not con:
        return
    table_format = f""" CREATE TABLE IF NOT EXISTS {table_name} (
            {headers}
            );"""
    print(table_format)
    create_table(con, table_format)
    return


def sql_np_test(
    # df
) -> None:
    """
    convert_df_into_sql
    """
    con = sql.connect("db/dimers.db", detect_types=sql.PARSE_DECLTYPES)
    cur = con.cursor()
    x = np.arange(18).reshape(3, 6)
    cur.execute("DROP TABLE IF EXISTS test2")
    cur.execute(
        "create table test2 (id INTEGER PRIMARY KEY AUTOINCREMENT, arr array)"
    )
    cur.execute("insert into test2 (arr) values (?)", (x,))
    cur.execute("select arr from test2")
    data = cur.fetchone()[0]
    return data


def insert_new_row(
    cur,
    con,
    table="main",
    insertion=["main_id", "geometry"],
    values=(None, 0),
) -> None:
    """
    insert_new_row inserts row into a table.
    insertion = "insert into main(main_id, geometry) values (?)",
    values = (x,)
    """
    q = ",".join(["?" for i in range(len(insertion))])
    ins = ",".join([i for i in insertion])
    cmd = f"insert into {table}({ins}) values ({q})"
    cur.execute(cmd, values)
    con.commit()
    return


def create_new_induction_table() -> None:
    """
    create_new_induction_table
    """
    con, cur = establish_connection()
    cur.execute(
        f"""
CREATE TABLE induction2 (
        db_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dimerpair TEXT NOT NULL,
        id INTEGER NOT NULL,
        RA array,
        RB array,
        ZA array,
        ZB array,
        TQA FLOAT NOT NULL,
        TQB FLOAT NOT NULL,
        Ind_aug FLOAT NOT NULL,
        electric_field_A array,
        electric_field_B array,
        vac_multipole_A array,
        vac_multipole_B array,
        environment_multipole_A array,
        environment_multipole_B array);
"""
    )
    con.commit()
    return


def subset_df(index_split=range(10000, 10050)):
    df = pd.read_pickle("1600K_train_dimers-fixed.pkl")
    df = df.iloc[index_split]
    df.to_pickle("10k_10050_dimers.pkl")
    return


def convert_df_into_sql(
    df_p="df.pkl",
    db_p="db/sql.db",
    input_columns={
        "Dimer": "TEXT",
        "R": "FLOAT",
        "CBS": "FLOAT",
        "monAs": "array",
        "monBs": "array",
        "Geometry": "array",
        "Benchmark": "FLOAT",
        "C6s": "array",
        "C6_A": "array",
        "C6_B": "array",
        "HF_dz": "FLOAT",
        "HF_tz": "FLOAT",
    },
    output_columns={
        "HF_dz": "FLOAT",
        "HF_jdz": "FLOAT",
        "HF_adz": "FLOAT",
        "HF_tz": "FLOAT",
        "main_id": "INTEGER PRIMARY KEY",
    },
) -> None:
    """
    convert_df_into_sql builds db from pd.DataFrame
    """
    con, cur = establish_connection(db_p=db_p)
    df = pd.read_pickle(df_p)
    # df = df[df['DB'] == "S22by7"]
    # df = df.drop_duplicates(subset="System #", inplace=False)
    print(f"{len(df)=}")
    print(type(output_columns))
    print(input_columns)

    for k, v in output_columns.items():
        if v.lower() == "float":
            df[k] = [float(0) for i in range(len(df))]
        elif v.lower() == "array":
            df[k] = [np.zeros(2) for i in range(len(df))]
        elif v.lower() == "text":
            df[k] = ["" for i in range(len(df))]
        elif v.lower() == "integer primary key":
            df[k] = [i for i in range(len(df))]
        else:
            raise TypeError()
        input_columns[k] = v
    df = df[[i for i in input_columns.keys()]]
    # print(input_columns)
    print(df.columns)
    print(df)
    print("built df...")
    df.to_sql(
        "main",
        con,
        if_exists="replace",
        index=False,
        dtype=input_columns,
    )
    return


def convert_df_into_sql_mp(df_p="5_dimers.pkl", db_p="db/dimers.db") -> None:
    """
    convert_df_into_sql
    """
    con, cur = establish_connection(db_p=db_p)
    df = pd.read_pickle(df_p)
    del df["multipoles_A"]
    del df["multipoles_B"]
    df["main_id"] = [i for i in range(len(df))]
    # df["electric_field_A"] = [np.zeros(5) for i in range(len(df))]
    # df["electric_field_B"] = [np.zeros(5) for i in range(len(df))]
    df["vac_multipole_A"] = [np.zeros(5) for i in range(len(df))]
    df["vac_multipole_B"] = [np.zeros(5) for i in range(len(df))]
    df["environment_multipole_A"] = [np.zeros(5) for i in range(len(df))]
    df["environment_multipole_B"] = [np.zeros(5) for i in range(len(df))]
    df["vac_widths_A"] = [np.zeros(5) for i in range(len(df))]
    df["vac_widths_B"] = [np.zeros(5) for i in range(len(df))]
    df["vac_vol_rat_A"] = [np.zeros(5) for i in range(len(df))]
    df["vac_vol_rat_B"] = [np.zeros(5) for i in range(len(df))]
    print("built extra columns...")
    print(df.columns.values)
    df.to_sql(
        "main",
        con,
        if_exists="replace",
        index=False,
        chunksize=50,
        method="multi",
        dtype={
            "main_id": "INTEGER PRIMARY KEY",
            "dimerpair": "TEXT ",
            "RA": "array",
            "RB": "array",
            "ZA": "array",
            "ZB": "array",
            "TQA": "FLOAT",
            "TQB": "FLOAT",
            "id": "INTEGER",
            "Total_jun": "FLOAT",
            "Elst_jun": "FLOAT",
            "Exch_jun": "FLOAT",
            "Ind_jun": "FLOAT",
            "Disp_jun": "FLOAT",
            "Total_aug": "FLOAT",
            "Elst_aug": "FLOAT",
            "Exch_aug": "FLOAT",
            "Ind_aug": "FLOAT",
            "Disp_aug": "FLOAT",
            "Ind_aug": "FLOAT",
            "electric_field_A": "array",
            "electric_field_B": "array",
            "vac_multipole_A": "array",
            "vac_multipole_B": "array",
            "environment_multipole_A": "array",
            "environment_multipole_B": "array",
            "vac_widths_A": "array",
            "vac_widths_B": "array",
            "vac_vol_rat_A": "array",
            "vac_vol_rat_B": "array",
        },
    )
    return


def update_by_id(
    conn,
    cursor,
    output,
    id_value: int = 0,
    id_label: str = "main_id",
    table="main",
    output_columns=[
        "electric_field_A",
        "electric_field_B",
        "vac_multipole_A",
        "vac_multipole_B",
        "environment_multipole_A",
        "environment_multipole_B",
    ],
) -> None:
    """
    update_mp_rows
    """
    headers = ",\n".join([f"{i} = ?" for i in output_columns])
    cmd = f"""
        UPDATE {table}
        SET
            {headers}
        WHERE
            {id_label}=?;
    """
    # print(cmd)
    # print("OUTPUT:", (*tuple(output), id_value))
    cursor.execute(
        cmd,
        (*tuple(output), id_value),
    )
    conn.commit()
    return


def update_rows(
    conn,
    cursor,
    output,
    col_val,
    col_match="rowid",
    table="main",
    output_columns=[
        "electric_field_A",
        "electric_field_B",
        "vac_multipole_A",
        "vac_multipole_B",
        "environment_multipole_A",
        "environment_multipole_B",
    ],
) -> None:
    """
    update_mp_rows
    """
    headers = ",\n".join([f"{i} = ?" for i in output_columns])
    print("headers", headers)
    print("output", output)
    print(
        (*tuple(output), col_val),
    )
    cmd = f"""
        UPDATE {table}
        SET
            {headers}
        WHERE
            {col_match}=?;
    """
    cursor.execute(
        cmd,
        (*tuple(output), col_val),
    )
    conn.commit()
    return


def update_mp_rows(
    conn,
    cursor,
    output,
    rowid,
) -> None:
    """
    update_mp_rows
    """
    cursor.execute(
        f"""
UPDATE induction
SET
        electric_field_A = ?,
        electric_field_B = ?,
        vac_multipole_A = ?,
        vac_multipole_B = ?,
        environment_multipole_A = ?,
        environment_multipole_B = ?
    WHERE
        rowid = ?;
""",
        (*tuple(output), rowid),
    )
    conn.commit()
    return


def collect_row_specific_into_js_mp(
    cursor: object,
    headers: [] = ["rowid", "id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    mem: str = "4gb",
    extra_info: [] = "hf/aug-cc-pV(D+d)Z",
    dataclass_obj: mp_js = mp_js,
    rowid: int = 0,
    table="main",
) -> []:
    """
    collects a specific row for a table with requested headers.
    The headers list must match the dataclass_obj's fields by positional order.
    """
    cols = ", ".join(headers)
    sql_cmd = f"""SELECT {cols} FROM {table} WHERE {table}.rowid = {rowid};"""
    cursor.execute(sql_cmd)
    js = dataclass_obj(*(cursor.fetchall()[0]), extra_info, mem=mem)
    return js


def collect_id_into_js(
    cursor: object,
    headers: [] = ["main_id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    mem: str = "4gb",
    extra_info: [] = ["hf/aug-cc-pV(D+d)Z"],
    dataclass_obj: mp_js = mp_js,
    id_value: int = 0,
    id_label: str = "main_id",
    table="main",
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    cols = ", ".join(headers)
    sql_cmd = f"""SELECT {cols} FROM {table} WHERE {id_label} = {id_value};
"""
    cursor.execute(sql_cmd)
    v = cursor.fetchone()
    try:
        js = dataclass_obj(
            *v,
            extra_info,
            mem=mem,
        )
    except (TypeError):
        print(
            "ERROR:\n\tEXITING FROM collect_id_into_js\n\n\tCheck that id is valid!\n"
        )
        os.sys.exit()
    return js


def return_id_list(cur, column, table_name, id_name="id", values=[0]) -> [int]:
    """
    return_id_list queries db for matches with column and returns id
    """
    op = "="
    if len(values) == 1:
        if values[0] == "NULL":
            op = " IS "
        sql_cmd = f"""SELECT {id_name} FROM {table_name} WHERE {column}{op}{values[0]};"""
    else:
        sql_cmd = f"""SELECT {id_name} FROM {table_name} WHERE {column} IN {tuple(values)};"""
    cur.execute(sql_cmd)
    id_list = [i for i, *_ in cur.fetchall()]
    return id_list

def query_clean_match(m):
    """
    query_clean_match
    """
    if type(m[0]) == str:
        m = [f'"{i}"' for i in m]
    return m


def query_columns_for_values(cur, table_name, id_names=["id"], matches={
    "id": [0],
    }) -> [int]:
    """
    return_id_list queries db for matches with column and returns id
    """
    if type(id_names) == str:
        id_name = id_names
    else:
        id_name = ", ".join(id_names)
    if len(matches) > 0:
        where_match = []
        for k, v in matches.items():
            v = query_clean_match(v)
            if len(v) == 1:
                if v[0] == '"NULL"':
                    m = f"{k} IS NULL"
                else:
                    m = f'{k}=={v[0]}'
            else:
                m = f"{k} IN {tuple(v)}"
            where_match.append(m)
        wm = " AND ".join(where_match)
        sql_cmd = f"""SELECT {id_name} FROM {table_name} WHERE {wm};"""
    else:
        sql_cmd = f"""SELECT {id_name} FROM {table_name};"""
    print(sql_cmd)
    cur.execute(sql_cmd)
    val_list = [i for i in cur.fetchall()]
    for i in range(len(val_list)):
        if len(val_list[i]) == 1:
            val_list[i] = val_list[i][0]
    return val_list


def return_id_list_full_table(cur, table_name, id_name="id") -> [int]:
    """
    return_id_list queries db for matches with column and returns id
    """
    sql_cmd = f"""SELECT {id_name} FROM {table_name};"""
    cur.execute(sql_cmd)
    id_list = [i for i, *_ in cur.fetchall()]
    return id_list


def collect_ids_into_ls(
    cursor: object,
    id_list: [] = [0, 1],
    id_label: str = "main_id",
    table="main",
    outputs="*",
) -> []:
    """
    collects ids into list
    """
    sql_cmd = f"""SELECT {outputs} FROM {table} WHERE {table}.{id_label} IN {tuple(id_list)};"""
    cursor.execute(sql_cmd)
    js_ls = [i for i in cursor.fetchall()]
    return js_ls


def collect_ids_into_js_ls(
    cursor: object,
    headers: [] = ["main_id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    mem: str = "4gb",
    extra_info: [] = ["hf/aug-cc-pV(D+d)Z"],
    dataclass_obj: mp_js = mp_js,
    id_list: [] = [0, 1],
    id_label: str = "main_id",
    table="main",
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    cols = ", ".join(headers)
    sql_cmd = f"""SELECT {cols} FROM {table} WHERE {table}.{id_label} IN {tuple(id_list)};"""
    cursor.execute(sql_cmd)
    js_ls = [
        dataclass_obj(
            *i,
            extra_info,
            mem=mem,
        )
        for i in cursor.fetchall()
    ]
    return js_ls


def collect_all_table_values_into_ls(
    cursor: object,
    headers: [] = ["id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    id_list: [] = [0, 1],
    id_label: str = "id",
    table="tcase",
    process_func=None,
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    cols = ", ".join(headers)
    sql_cmd = f"""SELECT {cols} FROM {table};"""
    cursor.execute(sql_cmd)
    if process_func:
        ls = [process_func(i) for i in cursor.fetchall()]
    else:
        ls = [i for i in cursor.fetchall()]
    return ls


def collect_ids_into_ls(
    cursor: object,
    headers: [] = ["id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    id_list: [] = [0, 1],
    id_label: str = "id",
    table="tcase",
    process_func=None,
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    cols = ", ".join(headers)
    sql_cmd = (
        f"""SELECT {cols} FROM {table} WHERE {id_label} IN {tuple(id_list)};"""
    )
    cursor.execute(sql_cmd)
    if process_func:
        ls = [process_func(i) for i in cursor.fetchall()]
    else:
        ls = [i for i in cursor.fetchall()]
    return ls


def collect_rows_into_js_ls_mp(
    cursor: object,
    headers: [] = ["main_id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    mem: str = "4gb",
    extra_info: [] = ["hf/aug-cc-pV(D+d)Z"],
    dataclass_obj: mp_js = mp_js,
    v_range: [] = [0, 2],
    table="main",
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    cols = ", ".join(headers)
    sql_cmd = f"""SELECT {cols} FROM {table} WHERE {table}.rowid
    BETWEEN {v_range[0]} AND {v_range[1]}
"""
    cursor.execute(sql_cmd)
    js_ls = [
        dataclass_obj(
            *i,
            extra_info,
            mem=mem,
        )
        for i in cursor.fetchall()
    ]
    return js_ls


def collect_rows_index_range(
    cursor: object,
    v_range: [] = [0, 2],
    table="main",
) -> []:
    """
    collect_rows_index_range collects a range of rows for a table with requested headers.
    """
    sql_cmd = f"""SELECT * FROM {table} WHERE {table}.rowid
    BETWEEN {v_range[0]} AND {v_range[1]}
"""
    cursor.execute(sql_cmd)
    return cursor.fetchall()


def select_table_by_column_match(
    cursor: object,
    selection="*",
    column="id",
    value="'CCCxNXdO-1_acc-carbonyl_CCxdOXNc1nccynHY1_H-Narom_37_0.73_175_29_132_25_93'",
    table="main",
):
    sql_cmd = (
        f"""SELECT {selection} FROM {table} WHERE {table}.{column}={value};"""
    )
    cursor.execute(sql_cmd)
    return cursor.fetchall()


def table_to_df_pkl(
    db_p="db/dimers_all.db",
    table="main",
    df_p="data/dimers_10k.pkl",
    id_list=[],
    id_label="main_id",
) -> None:
    """
    table_to_df_pkl
    """
    con, cur = establish_connection(db_p)
    if id_list:
        cmd = f"""SELECT * FROM {table} WHERE {table}.{id_label} IN {tuple(id_list)};"""
        print(cmd)
        df = pd.read_sql_query(cmd, con)
    else:
        df = pd.read_sql_query(f"SELECT * from {table};", con)
    print(df)
    df.to_pickle(df_p)
    print(df.columns)
    return


def table_to_df_csv(
    db_p="db/dimers_all.db",
    table="main",
    df_csv="data/dimers_10k.csv",
    id_list=[],
    id_label="main_id",
) -> None:
    """
    table_to_df_csv
    """
    con, cur = establish_connection(db_p)
    if id_list:
        cmd = f"""SELECT * FROM {table} WHERE {table}.{id_label} IN {tuple(id_list)};"""
        print(cmd)
        df = pd.read_sql_query(cmd, con)
    else:
        df = pd.read_sql_query(f"SELECT * from {table};", con)
    print(df)
    df.to_csv(df_csv)
    print(df.columns)
    return



def read_example_output(db_path="db/dimers.db", row_range=[0, 1]) -> None:
    """
    read_example_output reads sql row by rowid to verify update
    """
    con, cur = establish_connection(db_path)
    rows = collect_rows_index_range(cur, row_range)
    for n, i in enumerate(rows):
        print(n, list(i))
    return


def read_output(
    db_path="db/dimers.db",
    id_list=[0, 1],
    id_label: str = "main_id",
    table: str = "main",
    outputs="*",
) -> None:
    """
    read_example_output reads sql row by rowid to verify update
    """
    con, cur = establish_connection(db_path)
    r = collect_ids_into_ls(
        cur, id_list=id_list, id_label=id_label, table=table, outputs=outputs
    )
    for n, i in enumerate(r):
        print(f"main_id: {id_list[n]}, \t{list(i)[-5:-1]}")
    return


def delete_rows_by_search(
    con,
    table_name: str,
    column: str,
    values: [],
) -> None:
    """
    delete_rows_by_id
    """
    cur = con.cursor()
    if len(values) == 1:
        sql_cmd = f"""DELETE FROM {table_name} WHERE {column}=={values[0]};"""
        print(sql_cmd)
    else:
        sql_cmd = (
            f"""DELETE FROM {table_name} WHERE {column} IN {tuple(values)};"""
        )
    cur.execute(sql_cmd)
    con.commit()
    return
