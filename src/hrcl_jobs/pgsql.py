import pandas as pd
import numpy as np
import psycopg2
from pprint import pprint as pp
from dataclasses import dataclass
from .jobspec import example_js
from . import sqlt

@dataclass
class example_js:
    id_label: int
    val: float
    extra_info: {}
    mem: str = None

@dataclass
class psqldb:
    dbname: str
    user: str
    password: str
    host: str

def connect(url):
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    return conn, cur


def establish_connection(dbinfo=psqldb, user=None, password=None, host=None):
    if isinstance(dbinfo, psqldb):
        dbname, user, password, host = (
            dbinfo.dbname,
            dbinfo.user,
            dbinfo.password,
            dbinfo.host,
        )
    elif isinstance(dbinfo, str) and (user is None or password is None or host is None):
        print("Error: Must provide user, password, and host")
        raise ValueError
    else:
        dbname = dbinfo

    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
    )
    cur = conn.cursor()
    return conn, cur


def schema_information(conn, cur, schema_name):
    cur.execute(
        f"""
        SELECT 
            table_name 
        FROM 
            information_schema.tables
        WHERE 
            table_schema = '{schema_name}';
        """
    )
    result = cur.fetchall()
    result = pd.DataFrame(result, columns=["table_name"])
    print(f"Schema information for {schema_name}")
    pp(result)
    return result


def table_information(conn, cur, schema_name, table_name):
    cur.execute(
        f"""
        SELECT 
            column_name, 
            data_type
        FROM 
            information_schema.columns
        WHERE 
            table_schema = '{schema_name}' 
            AND table_name   = '{table_name}';
        """
    )
    result = cur.fetchall()
    result = pd.DataFrame(result, columns=["column_name", "data_type"])
    print(f"Schema information for {schema_name}.{table_name}")
    pp(result)
    return result


def create_junction_table(conn, cur, schema_name, t1_name, t1_id, t2_name, t2_id):
    # Now we need to create a junction table to connect the two tables
    new_table_name = f"{t1_name}__{t2_name}"
    try:
        cur.execute(
            f"""
        CREATE TABLE {schema_name}.{new_table_name}(
            id SERIAL PRIMARY KEY,
            {t1_id} INTEGER REFERENCES {schema_name}.{t1_name}({t1_id}) ON DELETE CASCADE,
            {t2_id} INTEGER REFERENCES {schema_name}.{t2_name}({t2_id}) ON DELETE CASCADE
        );
        """
        )
        print(f"Creating junction table {schema_name}.{new_table_name}")
    except psycopg2.Error:
        print(f"Table {schema_name}.{new_table_name} exists")
        conn.rollback()
    conn.commit()
    return


def update_by_id(
    conn,
    cursor,
    output,
    id_value: int = 0,
    id_label: str = "id",
    table="main",
    output_columns=[
        "HF_adz",
        "MP2_adz",
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


def collect_id_into_js(
    cursor: object,
    headers: [] = ["main_id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    extra_info: [] = ["hf/aug-cc-pV(D+d)Z"],
    dataclass_obj: example_js = example_js,
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
            *args,
        )
    except TypeError as e:
        print(e)
        os.sys.exit()
    return js


def collect_ids_into_js_ls(
    cursor: object,
    headers: [] = ["main_id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    extra_info: [] = ["hf/aug-cc-pV(D+d)Z"],
    dataclass_obj: example_js = example_js,
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
        )
        for i in cursor.fetchall()
    ]
    return js_ls


def query_columns_for_values(
    cur,
    schema_name,
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
            v = sqlt.query_clean_match(v)
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
        sql_cmd = f"""SELECT {id_name} FROM {schema_name}.{table_name} WHERE {wm};"""
    else:
        sql_cmd = f"""SELECT {id_name} FROM {schema_name}.{table_name};"""
    print(sql_cmd)
    cur.execute(sql_cmd)
    val_list = [i for i in cur.fetchall()]
    for i in range(len(val_list)):
        if len(val_list[i]) == 1:
            val_list[i] = val_list[i][0]
    return val_list
