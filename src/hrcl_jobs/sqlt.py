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
    print(headers)
    if not conn:
        return
    table_format = f""" CREATE TABLE {table_name} (
            {headers}
            );"""
    print(table_format)
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


def table_add_columns(
    con: sql.Connection,
    table_name: str,
    table_dict: dict,
    debug: bool = True,
) -> bool:
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


def drop_table(db_name="db/main.db", table_name="main") -> None:
    """
    drop_table drops a table from a db
    """
    con, cur = establish_connection(db_name)
    if not con:
        return
    cur.execute(f"DROP TABLE {table_name}")
    con.commit()
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
    cur.execute("create table test2 (id INTEGER PRIMARY KEY AUTOINCREMENT, arr array)")
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


def update_column_value(
    con,
    cur,
    table_name,
    column_name,
    value,
    matches={},
    joiner="AND",
    verbose=1,
):
    """
    update_column_value
    """
    cmd = f"UPDATE {table_name} SET {column_name} = {value}"
    cmd = handle_sql_matches(matches, cmd)
    if verbose:
        print(cmd)
    try:
        cur.execute(cmd)
        con.commit()
    except (Exception) as e:
        print(e)
        return False
    return True


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
    table_name="main",
    overwrite=False,
) -> None:
    """
    convert_df_into_sql builds db from pd.DataFrame
    """
    if os.path.exists(db_p) and not overwrite:
        print(f"{db_p} already exists!")
        return
    con, cur = establish_connection(db_p=db_p)

    df = pd.read_pickle(df_p)
    # df = df[df['DB'] == "S22by7"]
    # df = df.drop_duplicates(subset="System #", inplace=False)
    print(f"{len(df)=}")
    print(type(output_columns))
    print(input_columns)

    for k, v in output_columns.items():
        if v.lower() == "float" or v.lower() == "real" or v.lower() == "integer":
            df[k] = [pd.NA for i in range(len(df))]
        elif v.lower() == "array":
            df[k] = [pd.NA for i in range(len(df))]
        elif v.lower() == "text":
            df[k] = [pd.NA for i in range(len(df))]
        elif v.lower() == "integer primary key":
            df[k] = [i for i in range(len(df))]
        else:
            print(f"Ensure that {v} is a valid SQL type.")
            df[k] = [None for i in range(len(df))]
        input_columns[k] = v
    df = df[[i for i in input_columns.keys()]]
    # print(input_columns)
    print(df.columns)
    print(df)
    print("built df...")
    df.to_sql(
        table_name,
        con,
        if_exists="replace",
        index=False,
        dtype=input_columns,
    )
    print("built sql...")
    for k, v in output_columns.items():
        if v.lower() == "array":
            print(v)
            update_column_value(con, cur, table_name, k, "NULL")
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
    insert_none=True,
) -> None:
    """
    update_mp_rows
    """
    if len(output_columns) != len(output):
        print("OUTPUT_COLUMNS AND OUTPUT MUST BE THE SAME LENGTH!", id_value)
        return
    headers = ",\n".join([f"{i} = ?" for i in output_columns])
    cmd = f"""
        UPDATE {table}
        SET
            {headers}
        WHERE
            {id_label}=?;
    """
    if not insert_none:
        for i in output:
            if i is None:
                print(f"None in output, skipping {id_value}: {output = }...")
                return
    cursor.execute(
        cmd,
        (*tuple(output), id_value),
    )
    conn.commit()
    return


def update_rows(
    conn,
    cursor,
    output: list,
    col_val: int,  # could be string as well
    col_match="id",
    table="main",
    output_columns=[
        "electric_field_A",
        "electric_field_B",
        "vac_multipole_A",
        "vac_multipole_B",
        "environment_multipole_A",
        "environment_multipole_B",
    ],
    verbose=1,
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
            {col_match}=?;
    """
    if verbose:
        print("HEADERS = ", headers)
        print("OUTPUT = ", output)
        print(
            (*tuple(output), col_val),
        )
        print(cmd)

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
    *args,
) -> []:
    """
    collect_rows collects a range of rows for a table with requested headers.
    The headers list must match the dataclass_obj's fields.
    """
    if callable(headers):
        headers = headers()

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
            *args,
        )
    except TypeError as e:
        print(e)
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
        sql_cmd = (
            f"""SELECT {id_name} FROM {table_name} WHERE {column}{op}{values[0]};"""
        )
    else:
        sql_cmd = (
            f"""SELECT {id_name} FROM {table_name} WHERE {column} IN {tuple(values)};"""
        )
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


def handle_sql_matches(matches, sql_cmd, joiner="AND"):
    if len(matches) > 0:
        where_match = []
        for k, v in matches.items():
            v = query_clean_match(v)
            if len(v) == 1:
                if v[0] == '"NULL"':
                    m = f"{k} IS NULL"
                elif v[0] == '"NOT NULL"':
                    m = f"{k} IS NOT NULL"
                elif type(v[0]) == str and "!" in v[0]:
                    m = f"{k} NOT LIKE '{v[0][2:-1]}'"
                else:
                    m = f"{k}=={v[0]}"
            else:
                m = f"{k} IN {tuple(v)}"
            where_match.append(m)
        wm = f" {joiner} ".join(where_match)
        sql_cmd = f"""{sql_cmd} WHERE {wm};"""
    return sql_cmd


def sqlt_execute(
    cur,
    table_name,
    action="SELECT",
    extra_action="WHERE",
    cols=["id"],
    matches={
        # "id": [0],
    },
    joiner="AND",
) -> [int]:
    """
    return_id_list queries db for matches with column and returns id
    """
    conn = cur.connection
    if type(cols) == str:
        cols = cols
    else:
        cols = ", ".join(cols)
    if len(matches) > 0:
        where_match = []
        for k, v in matches.items():
            v = query_clean_match(v)
            if len(v) == 1:
                if v[0] == '"NULL"':
                    m = f"{k} IS NULL"
                elif v[0] == '"NOT NULL"':
                    m = f"{k} IS NOT NULL"
                elif type(v[0]) == str and "!" in v[0]:
                    # m = f"{k}!='{v[0][1:-1]}'"
                    m = f"{k} NOT LIKE '{v[0][2:-1]}'"
                else:
                    m = f"{k}=={v[0]}"
            else:
                if type(v[0]) == str:
                    v = ", ".join([f"'{i[1:-1]}'" for i in v])
                    m = f"{k} IN ({v})"
                else:
                    m = f"{k} IN {tuple(v)}"

            where_match.append(m)
        wm = f" {joiner} ".join(where_match)
        sql_cmd = f"""{action} {cols} FROM {table_name} {extra_action} {wm};"""
    else:
        sql_cmd = f"""{action} {cols} FROM {table_name};"""
    print(sql_cmd)
    cur.execute(sql_cmd)
    val_list = [i for i in cur.fetchall()]
    for i in range(len(val_list)):
        if len(val_list[i]) == 1:
            val_list[i] = val_list[i][0]
    return val_list


def query_distinct_columns(cur, table_name, col) -> []:
    """
    query_distinct_columns gets a list of distinct values for a column
    """
    sql_cmd = f"SELECT count({col}), {col} AS CountOf FROM {table_name} GROUP BY {col};"
    # print(sql_cmd)
    cur.execute(sql_cmd)
    val_list = [[i, j] for i, j in cur.fetchall()]
    return val_list


def query_columns_for_values(
    cur,
    table_name,
    id_names=["id"],
    matches={
        "id": [0],
    },
) -> [int]:
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
                elif v[0] == '"NOT NULL"':
                    m = f"{k} IS NOT NULL"
                else:
                    m = f"{k}=={v[0]}"
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
    sql_cmd = (
        f"""SELECT {cols} FROM {table} WHERE {table}.{id_label} IN {tuple(id_list)};"""
    )
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
    if len(id_list) > 1:
        sql_cmd = (
            f"""SELECT {cols} FROM {table} WHERE {id_label} IN {tuple(id_list)};"""
        )
    else:
        sql_cmd = f"""SELECT {cols} FROM {table} WHERE {id_label}=={id_list[0]};"""
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
    sql_cmd = f"""SELECT {selection} FROM {table} WHERE {table}.{column}={value};"""
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
    print(f"db_p: {db_p}")
    print(f"df_p: {df_p}")
    print(f"table: {table}")
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
        sql_cmd = f"""DELETE FROM {table_name} WHERE {column} IN {tuple(values)};"""
    cur.execute(sql_cmd)
    con.commit()
    return


# TODO: test function
def create_update_table(
    db_path: str,
    table_name: str,
    table_cols: dict,
    data: dict = {},
    debug = True,
) -> bool:
    """
    create_update_table will either create a new table
    or update the columns if more columns are added according to
    the table_cols schema
    """

    table_not_exists = new_table(db_path, table_name, table_cols)
    con, cur = establish_connection(db_path)
    table_add_columns(con, table_name, table_cols)
    print("table exists", table_not_exists)
    if table_not_exists:
        if len(data) > 0:
            print("Updating Table...")
            insertion, vals = [], []
            v_len = len(data[list(data.keys())[0]])
            for k, v in data.items():
                insertion.append(k)
                vals.append(v)
                if len(v) != v_len:
                    print("ERROR: data length mismatch", k, len(v), v_len)
                    return False

            vals = tuple(vals)
            cnt = 0
            for r in zip(*vals):
                insert_new_row(cur, con, table_name, insertion, r)
                cnt += 1
            print(f"Inserted {cnt} rows into {table_name}")
    elif len(data) > 0:
        ids = return_id_list_full_table(cur, table_name, "id")
        print(ids)
        for k, v in data.items():
            if k not in table_cols.keys():
                print(f"ERROR: {k} not in {table_name} columns")
                return False
            else:
                print(f"Updating {k}...")
                for i, j in zip(ids, v):
                    print(k, i, j)
                    update_by_id(
                        con,
                        cur,
                        output=[j],
                        id_value=i,
                        id_label="id",
                        table=table_name,
                        output_columns=[k],
                    )
    return True


def collect_ids_for_parallel(
    DB_NAME: str,
    TABLE_NAME: str,
    col_check: dict = ["SAPT0_adz", "array"],  # ["name", "type"]
    sort_column: str = "Geometry",
    matches: dict = {"SAPT0_adz": "NULL"},
    id_value: str = "id",
    joiner: str = "AND",
    ascending: bool = True,
) -> []:
    """
    collect_ids_for_parallel creates column if doesn't exist and matches NULL entries for ID list
    """
    con, cur = establish_connection(DB_NAME)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    id_list = []
    if rank == 0:
        create_update_table(
            DB_NAME,
            TABLE_NAME,
            table_cols={col_check[0]: col_check[1]},
            data={},
        )
        query = sqlt_execute(
            cur,
            TABLE_NAME,
            cols=[
                id_value,
                sort_column,
            ],
            matches=matches,
            joiner=joiner,
        )
        query = [(i[0], len(i[1])) for i in query]
        if ascending is not None:
            query = sorted(query, key=lambda x: x[1], reverse=ascending)
        id_list = [i[0] for i in query]
        print(f"MAIN: {len(id_list)} computations to run")
    id_list = comm.bcast(id_list, root=0)
    return id_list


def merge_db_cols(
    db1={
        "db_path": "db/schr.db",
        "table_name": "main",
    },
    db2={
        "db_path": "db/schr_HIVE.db",
        "table_name": "main",
        "col_names": [
            "SAPT0_jtz",
            # "SAPT0_atz",
        ],
    },
    overwrite=True,
):
    print(f"db1: {db1['db_path']}")
    print(f"db2: {db2['db_path']}")
    con1, cur1 = establish_connection(db1["db_path"])
    con2, cur2 = establish_connection(db2["db_path"])
    for i in db2["col_names"]:
        print(i)
        q1 = sqlt_execute(
            cur1,
            db1["table_name"],
            cols=[
                "id",
            ],
            matches={
                i: ["NOT NULL"],
            },
        )
        print(f"{len(q1)} rows in {db1['table_name']} with {i} not null")
        if overwrite or len(q1) == 0:
            print("Updating...")
            q2 = sqlt_execute(
                cur2,
                db2["table_name"],
                cols=[
                    "id",
                    i,
                ],
                matches={
                    i: ["NOT NULL"],
                },
            )
            update = []
            for j in q2:
                update.append({"id": j[0], "value": j[1]})
            sql_cmd = f"UPDATE {db1['table_name']} SET {i}=:value WHERE rowId=:id"
            print(sql_cmd)
            con1.executemany(sql_cmd, update)
            # print(list(con1.execute(f"select {i} from {db1['table_name']}")))
            con1.commit()
    return

def rename_column(conn, table_name, old_column_name, new_column_name):
    sql_cmd = f"""ALTER TABLE {table_name} RENAME COLUMN {old_column_name} TO {new_column_name};"""
    print(sql_cmd)
    c = conn.cursor()
    c.execute(sql_cmd)
    conn.commit()
    return sql_cmd
