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
import io
import zlib  # default compression is 6 on a ( 0-9 ) scale

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(zlib.decompress(out.read()))
    return np.load(out)

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
        id_label: str = "id",
    ):
        self.pgsql_url = pgsql_url
        self.table_name = table_name
        self.schema_name = schema_name
        self.init_query_cmd = init_query_cmd
        self.job_query_cmd = job_query_cmd
        self.update_cmd = update_cmd
        self.id_label = id_label
        print(f"pgsql_operations initialized for {schema_name}.{table_name}")
        print(f"init_query_cmd:\n    {init_query_cmd}")
        print(f"job_query_cmd:\n    {job_query_cmd}")
        print(f"update_cmd:\n    {update_cmd}")

    def init_query(self, conn, where_value=None, sort_by=None, ascending=True, ids_only=True):
        cur = conn.cursor()
        print(self.init_query_cmd)
        if where_value is None:
            cur.execute(self.init_query_cmd)
        else:
            cur.execute(self.init_query_cmd, (where_value,))
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        if sort_by is not None:
            df = df.sort_values(by=sort_by, ascending=ascending)
        if ids_only:
            return df[self.id_label].tolist()
        return df

    def job_query(self, conn, id, js_obj, extra_info={}):
        cur = conn.cursor()
        if isinstance(id, list):
            cmd = self.job_query_cmd.replace("= %s", f" IN {tuple(id)}")
            cur.execute(cmd)
        else:
            cur.execute(self.job_query_cmd, (id,))

        if "mem" in js_obj.__annotations__ and "mem_per_process" in extra_info.keys():
            js_ls = [
                js_obj(
                    *i,
                    extra_info,
                    extra_info["mem_per_process"],
                )
                for i in cur.fetchall()
            ]
        else:
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
        try:
            for i in range(len(output)):
                if isinstance(output[i], np.ndarray):
                    output[i] = output[i].tolist()
            if len(output) < 1:
                print("No output to update")
                return 
            if len(output) != len(self.update_cmd.split('%s')) - 2:
                print(f"Output and update_cmd do not match: {len(output) = } {len(self.update_cmd.split('%s')) - 2 = }")
                return
            cur = conn.cursor()
            cur.execute(self.update_cmd, (*output, id_value))
            conn.commit()
        except (Exception) as e:
            import traceback
            print(f"Error updating {id_value}: {e}")
            print(traceback.format_exc())
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
    except psycopg2.Error as e:
        print("ERROR:", e)
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


def connect_to_db(pw_source="file", dbname="disco", ip_db=None, port=5432, return_con=False):
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
    if return_con:
        conn, _ = connect(pg_url)
        print(conn)
        return conn
    return pg_url

def identify_array_dtype(array):
    """
    Identify the correct PostgreSQL type for numpy arrays.
    """
    if issubclass(array.dtype.type, np.integer):
        return 'INTEGER'
    elif issubclass(array.dtype.type, np.float_):
        return 'DOUBLE PRECISION'
    else:
        raise ValueError(f"Array with unsupported data type {array.dtype}.")

def get_postgresql_column_types(df):
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'DOUBLE PRECISION',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
    }

    column_types = {}
    
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == 'object':
            if df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                # Determine the dimensionality
                first_valid_value = df[col][df[col].apply(lambda x: isinstance(x, np.ndarray))].iloc[0]

                array_dtype = identify_array_dtype(first_valid_value)
                if first_valid_value.ndim == 1:
                    column_types[col] = f'{array_dtype}[]'
                elif first_valid_value.ndim == 2:
                    column_types[col] = f'{array_dtype}[][]'
                else:
                    raise ValueError(f"Numpy array with more than 2 dimensions found in column '{col}'")
            elif isinstance(df.iloc[0][col], bool):
                column_types[col] = 'BOOLEAN'
            elif is_float_series(df[col]):
                column_types[col] = 'DOUBLE PRECISION'
            else:
                column_types[col] = 'TEXT'
        else:
            col_str = str(col_dtype)
            column_types[col] = type_mapping.get(col_str, 'TEXT')

    return column_types

def is_float_series(series):
    """
    Check if a series of dtype "object" actually contains float values.
    """
    try:
        # Convert series to float and check if all values can be converted.
        series.astype(float)
        return True
    except ValueError:
        return False


def format_value_for_sql(value):
    """
    Format individual value for SQL insertion.
    """
    if isinstance(value, bytes):
        print(value)
        value = convert_array(value)
    if isinstance(value, np.ndarray):
        # Convert ndarray to PostgreSQL array literal; nump.ndarray already prints in correct nested format if more dimensional.
        return f'{{ {str(value.tolist())[1:-1]} }}'
    elif pd.isna(value):
        return 'NULL'
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (np.integer, int)):
        return str(value)
    elif isinstance(value, (np.floating, float)):
        return str(value)
    elif isinstance(value, bool):
        return str(value).upper()
    elif isinstance(value, pd.Timestamp):
        return f"'{value}'"
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

def write_sql_file(df, table_name='my_table', file_name='output.sql'):
    col_types = get_postgresql_column_types(df)
    final_col_types = {}
    for col, v in col_types.items():
        if " " in col:
            col = f'"{col}"'
        final_col_types[col] = v

    columns_definition = ",\n  ".join([f'{col} {dtype}' for col, dtype in final_col_types.items()])
    
    create_table_stmt = f'CREATE TABLE {table_name} (\n  {columns_definition}\n);'
    
    insert_into_stmts = []
    for i, row in df.iterrows():
        values = ", ".join([format_value_for_sql(value) for value in row])
        insert_into_stmt = f'INSERT INTO {table_name} VALUES ({values});'
        insert_into_stmts.append(insert_into_stmt)
    
    sql_statements = f'{create_table_stmt}\n\n' + "\n".join(insert_into_stmts)
    
    with open(file_name, 'w') as file:
        file.write(sql_statements)

def convert_to_sql(df, table_name, file_name=None, debug=False, schema=None):
    if schema is not None:
        table_name = f'{schema}.{table_name}'
    create_table_statement = f"DROP TABLE IF EXISTS {table_name};\nCREATE TABLE {table_name} (\n"
    insert_into_statement = f"INSERT INTO {table_name} VALUES "
    insert_values = []
    
    column_definitions = []
    column_types = get_postgresql_column_types(df)
    for col, dtype in column_types.items():
        # if " " in col:
        col = f'"{col}"'
        column_definitions.append(f"{col} {dtype}")
    print("CREATE TABLE statement:")
    print(",\n".join(column_definitions))

    create_table_statement += ",\n".join(column_definitions)
    create_table_statement += "\n);"

    for index, row in df.iterrows():
        values = []
        for col, value in row.items():
            if value is None or isinstance(value, bytes):
                values.append("NULL")
            elif isinstance(value, list):
                value = "'" + str(value).replace('[', '{').replace(']', '}') + "'" # Convert list to PostgreSQL array format
                values.append(value)
            elif isinstance(value, np.ndarray):
                # value = str(value).replace('[', '{').replace(']', '}') # Convert list to PostgreSQL array format
                value = f"'{{ {str(value.tolist())[1:-1].replace('[', '{').replace(']', '}')} }}'"
                values.append(value)
            elif isinstance(value, str):
                values.append("'" + value.replace("'", "''") + "'") # Escape single quotes in strings
            elif isinstance(value, bool):
                values.append(str(value).upper())
            else:
                values.append(str(value))
            if index==0 and debug:
                print(col, value)
        insert_values.append("(" + ", ".join(values) + ")")
        if index == 0 and debug:
            tmp = ",\n".join(insert_values) + ";"
            print(tmp)
    insert_into_statement += ",\n".join(insert_values) + ";"
    
    sql_schema = create_table_statement + "\n\n" + insert_into_statement
    if file_name is not None:
        with open(file_name, 'w') as f:
            f.write(sql_schema)
    return sql_schema

def generate_sql_file_info(df, table_name, foreign_keys=None, file_name=None, debug=False, schema=None):
    table_name_start = table_name
    if schema is not None:
        table_name = f'{schema}.{table_name}'
    create_table_statement = f"DROP TABLE IF EXISTS {table_name};\nCREATE TABLE {table_name} (\n"
    insert_into_statement = f"INSERT INTO {table_name} VALUES "
    insert_values = []
    column_definitions = []
    column_definitions.append(f"id_{table_name_start} SERIAL PRIMARY KEY")
    column_types = get_postgresql_column_types(df)
    for col, dtype in column_types.items():
        if " " in col:
            col = f'"{col}"'
        column_definitions.append(f"{col} {dtype}")
    if foreign_keys is not None:
        for fk in foreign_keys:
            column_definitions.append(f"FOREIGN KEY ({fk['main_col_ref']}) REFERENCES {fk['table']}({fk['reference']})")
    create_table_statement += ",\n".join(column_definitions)
    create_table_statement += "\n);"
    for index, row in df.iterrows():
        values = []
        for col, value in row.items():
            if value is None or isinstance(value, bytes):
                values.append("NULL")
            elif isinstance(value, list):
                value = "'" + str(value).replace('[', '{').replace(']', '}') + "'" # Convert list to PostgreSQL array format
                values.append(value)
            elif isinstance(value, np.ndarray):
                # value = str(value).replace('[', '{').replace(']', '}') # Convert list to PostgreSQL array format
                value = f"'{{ {str(value.tolist())[1:-1].replace('[', '{').replace(']', '}')} }}'"
                values.append(value)
            elif isinstance(value, str):
                values.append("'" + value.replace("'", "''") + "'") # Escape single quotes in strings
            else:
                values.append(str(value))
            if index==0 and debug:
                print(col, value)
        insert_values.append("(" + ", ".join(values) + ")")
        if index == 0 and debug:
            tmp = ",\n".join(insert_values) + ";"
            print(tmp)
    insert_into_statement += ",\n".join(insert_values) + ";"
    return create_table_statement, insert_into_statement

def convert_to_sql_multiple_dfs(main_df, dfs, table_name, file_name=None, debug=False, schema=None):
    foriegn_keys = []
    table_creations = ""
    all_insertions = ""
    for k, v in dfs.items():
        df = v['df']
        extra_table = f"{schema}_{k}"
        table_info, insertions = generate_sql_file_info(df, extra_table, file_name=None, debug=False, schema=schema)
        foriegn_keys.append({"main_col_ref": f"{extra_table}_fk", "table": extra_table, "reference": v['ref_col']})
        table_creations += table_info + "\n\n"
        all_insertions += insertions + "\n\n"
        # need to create foreign tables first but insertaions of foreign tables last

    main_table_info, main_insertions = generate_sql_file_info(main_df, table_name, foriegn_keys, file_name=None, debug=False, schema=schema)
    table_creations += main_table_info + "\n\n"
    # all_insertions = main_insertions + "\n\n" + all_insertions
    all_insertions += main_insertions  + "\n\n"
    sql_schema = table_creations + all_insertions
    if file_name is not None:
        with open(file_name, 'w') as f:
            f.write(sql_schema)
    return sql_schema

def table_add_columns(
    con: object,
    table_name: str,
    schema_name: str,
    table_dict: dict,
    debug: bool = True,
    print_only: bool = False,
) -> bool:
    """
    table_add_columns insert columns into a table.
    NOTE: don't use schema.table_name, but just table_name for cmd.
    """
    if schema_name is not None and not schema_name.endswith('.'):
        schema_name += "."
    cur = con.cursor()
    cmd = f"select column_name, data_type from information_schema.columns where table_name='{table_name}'"
    cur.execute(cmd)
    desc = cur.fetchall()
    existing_table = {}
    for i in desc:
        existing_table[i[0]] = i[1]
    if debug:
        print(f"{table_name = }")
        print(f"{cmd = }")
        print(f"{existing_table=}")
        pp(existing_table.keys())
    for k, v in table_dict.items():
        if k.replace('"', "") not in existing_table.keys():
            # if debug:
            if print_only:
                print(f"ALTER TABLE {schema_name}{table_name} ADD COLUMN {k} {v};")
                continue
            print(f"Adding column {k} to {schema_name}{table_name}")
            cur.execute(f"ALTER TABLE {schema_name}{table_name} ADD COLUMN {k} {v};")
            con.commit()
    return True

