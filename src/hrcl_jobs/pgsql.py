import pandas as pd
import numpy as np
import psycopg2
from pprint import pprint as pp
from dataclasses import dataclass
from .jobspec import example_js
from . import sqlt
import os
import subprocess
import urllib


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

class jobQuery(Exception):
    def __init__(self, msg):
        self.msg = msg

class pgsql_operations:
    def __init__(
        self,
        pgsql_url: str,
        table_name: str,
        schema_name: str,
        init_query_cmd,
        job_query_cmd: str,
        update_cmd: str,
    ):
        self.pgsql_url = pgsql_url
        self.table_name = table_name
        self.schema_name = schema_name
        self.init_query_cmd = init_query_cmd
        self.job_query_cmd = job_query_cmd
        self.update_cmd = update_cmd
        print(f"pgsql_operations initialized for {schema_name}.{table_name}")
        print(f"init_query_cmd:\n    {init_query_cmd}")
        print(f"job_query_cmd:\n    {job_query_cmd}")
        print(f"update_cmd:\n    {update_cmd}")

    def init_query(self, conn, where_value):
        cur = conn.cursor()
        print(self.init_query_cmd)
        cur.execute(self.init_query_cmd, (where_value,))
        return cur.fetchall()

    def job_query(self, conn, id, js_obj, extra_info={}):
        cur = conn.cursor()
        if isinstance(id, list):
            cmd = self.job_query_cmd.replace("= %s", f" IN {tuple(id)}")
            cur.execute(cmd)
        else:
            cur.execute(self.job_query_cmd, (id,))
        js_ls = [
            js_obj(
                *i,
                extra_info,
            )
            for i in cur.fetchall()
        ]
        if len(js_ls) == 0:
            raise jobQuery(f"No jobs found for {id}")
        elif len(js_ls) == 1:
            return js_ls[0]
        elif isinstance(id, list) and len(js_ls) != len(id):
            raise jobQuery(f"No jobs found for {id}")
        return js_ls

    def update(self, conn, output, id_value):
        for i in range(len(output)):
            if isinstance(output[i], np.ndarray):
                output[i] = output[i].tolist()
        cur = conn.cursor()
        cur.execute(self.update_cmd, (*output, id_value))
        conn.commit()
        return
    
    def connect_db(self):
        conn, cur = connect(self.pgsql_url)
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


def execute_sql_file(filename, connection):
    # Read the SQL file
    with open(filename, "r") as file:
        sql_script = file.read()

    # Execute the SQL commands
    with connection.cursor() as cursor:
        cursor.execute(sql_script)
        connection.commit()
    return


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


def connect_to_db(pw_source="file", dbname="disco", ip_db=None, port=5432):
    print(f"Connecting to {dbname} with {pw_source}")
    user_path_expand = os.path.expanduser(pw_source)
    with open(user_path_expand, "r") as f:
        data = f.readlines()
    user = data[0].strip()
    pw = urllib.parse.quote_plus(data[1].strip())
    # ip_db = "128.61.254.12:5432"
    if ip_db is None:
        cmd = "hostname -I | awk '{print $1}'"
        ip_db = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        ip_db = f"{ip_db}:{port}"
    else:
        ip_db = f"{ip_db}:{port}"

    psqldb_local = psqldb(
        dbname=dbname,
        user=user,
        host=ip_db,
        password=pw,
    )
    pg_url = (
        f"postgresql://{psqldb_local.user}:{psqldb_local.password}@{psqldb_local.host}/{psqldb_local.dbname}"
    )
    print(pg_url)
    return pg_url
