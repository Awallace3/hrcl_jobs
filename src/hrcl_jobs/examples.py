import hrcl_jobs_orca
from hrcl_jobs import sqlt
from hrcl_jobs.parallel import ms_sl


def main():
    """
    Runs s22 db jobs with DLPNO-CCSD(T)
    """
    # Build sql db
    db_path = "db/test.db"
    table_name = "main"
    table_cols = {
        "id": "INTEGER PRIMARY KEY",
        "RA": "array",
        "ZA": "array",
        "TQA": "array",
        "RB": "array",
        "ZB": "array",
        "TQB": "array",
        "mp2_adz": "float",
    }
    sqlt.new_table(db_path, table_name, table_cols)

    # Run jobs
    id_list = [i for i in range(10)]
    ms_sl(
        id_list=id_list,
        db_path=db_path,
        run_js_job=example_run_js_job,
        headers_sql=["id", "RA", "ZA", "TQA"],
        level_theory=["mp2/aug-cc-pvdz"],
        js_obj=example_js,
        ppm="4gb",
        table="main",
        id_label="id",
        output_columns=[
            "output_column_name",
        ],
    )
    return


if __name__ == "__main__":
    main()
