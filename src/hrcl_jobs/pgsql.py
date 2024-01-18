import pandas as pd
import numpy as np
import psycopg2
from pprint import pprint as pp


def establish_connection(dbname, user, password, host):
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
