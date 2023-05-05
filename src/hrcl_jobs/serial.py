from .sqlt import (
    establish_connection,
    update_mp_rows,
    update_rows,
    collect_rows_into_js_ls_mp,
    collect_row_specific_into_js_mp,
    read_example_output,
    collect_id_into_js,
    update_by_id,
)
import os
from glob import glob
import time
from .jobspec import example_js


def example_run_js_job(js: example_js) -> float:
    """
    example_run_js_job
    """
    v1 = js.val + 1
    v2 = js.val + 2
    return [v1, v2]


def ms_sl_serial(
    id_list=[0, 50],
    db_path="db/dimers_all.db",
    collect_id_into_js=collect_id_into_js,
    run_js_job=example_run_js_job,
    update_func=update_by_id,
    headers_sql=["main_id", "id", "RA", "RB", "ZA", "ZB", "TQA", "TQB"],
    level_theory=["hf/aug-cc-pV(D+d)Z"],
    js_obj=example_js,
    ppm="4gb",
    table="main",
    id_label="main_id",
    output_columns=[
        "env_multipole_A",
        "env_multipole_B",
        "vac_widths_A",
        "vac_widths_B",
        "vac_vol_rat_A",
        "vac_vol_rat_B",
    ],
):
    """
    To use ms_sl_serial properly, write your own collect_rows_into_js_ls and
    collect_row_specific_into_js functions to pass as arguements to this
    function. Ensure that collect_rows_into_js_ls returns the correct js for
    your own run_js_job function.

    This is designed to work with psi4 jobs using python api.
    """

    start = time.time()
    first = True
    con, cur = establish_connection(db_p=db_path)
    for n, active_ind in enumerate(id_list):
        js = collect_id_into_js(
            cur,
            mem=ppm,
            headers=headers_sql,
            extra_info=level_theory,
            dataclass_obj=js_obj,
            id_value=active_ind,
            id_label=id_label,
            table=table,
        )
        output = run_js_job(js)
        update_func(
            con,
            cur,
            output,
            id_label=id_label,
            id_value=active_ind,
            table=table,
            output_columns=output_columns,
        )
    print((time.time() - start) / 60, "Minutes")
    print("COMPLETED MAIN")
    return
