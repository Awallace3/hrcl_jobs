import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
import psi4
from qm_tools_aw import tools
import qcelemental as qcel
from pprint import pprint as pp
from . import jobspec


def run_psi4_dimer_ie(js: jobspec.psi4_dimer_js):
    """
    xtra = {"level_theory": ["pbe0/aug-cc-pVDZ"], "options": options}
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = tools.np_carts_to_string(ma)
    mb = tools.np_carts_to_string(mb)

    import qcfractal.interface as qcfi

    kw = qcfi.models.KeywordSet(**{"values": js.extra_info['options']})
    kw_id = client.add_keywords([kw])[0]

    geom = f"{charges[0]} {charges[1]}\n{M}"
    geom_in = qcfi.Molecule.from_data(geom)
    out = []
    psi4.core.be_quiet()
    level_theory = js.extra_info["level_theory"]
    for l in level_theory:
        m, bs = l.split("/")
        r1 = client.add_compute(
            program="psi4",
            method=l,
            basis=bs,
            driver="energy",
            molecule=[geom_in],
            keywords=kw_id,
            protocols={"bsse_type": js.extra_info['bsse_type']},
        )
        try:
            id1 = int(r1.ids[0])
            ret1 = client.query_results(id=r1.ids)[0]
            result1 = ret1.dict()
            if js.extra_info['bsse_type'] == "cp":
                ie = result1["extras"]["qcvars"]["CP-CORRECTED INTERACTION ENERGY"]
            elif js.extra_info['bsse_type'] == "nocp":
                ie = result1["extras"]["qcvars"]["NOCP-CORRECTED INTERACTION ENERGY"]
            else:
                print("bsse_type not recognized")
                raise ValueError
            out.append(ie)
        except (AttributeError, TypeError):
            out.append(None)
        return out


def run_saptdft_grac_shift_qcfi(js: jobspec.saptdft_mon_grac_js):
    """
    xtra = {"level_theory": ["pbe0/aug-cc-pVDZ"], "charge_index": 1, "options": options}
    """
    mn = []
    for i in js.monNs:
        mn.append(js.geometry[i, :])
    mn = tools.np_carts_to_string(mn)
    shift_n = run_dft_neutral_cation_qca_qcng_error(
        js.client,
        mn,
        charges=js.charges[js.extra_info["charge_index"]],
        ppm=js.mem,
        id_label=js.id_label,
        level_theory=js.extra_info["level_theory"],
        mon=js.extra_info["charge_index"],
    )
    return shift_n


def run_dft_neutral_cation_qca_qcng_error(
    client,
    M,
    charges,
    ppm,
    id_label,
    level_theory,
    mon,
    gather_results=True,
    print_file=False,
    options={"reference": "uhf"},
) -> np.array:
    """
    run_dft_neutral_cation
    """
    import qcfractal.interface as qcfi

    kw = qcfi.models.KeywordSet(**{"values": options})
    kw_id = client.add_keywords([kw])[0]

    geom_neutral = f"{charges[0]} {charges[1]}\n{M}"
    geom_cation = f"{charges[0]+1} {charges[1]+1}\n{M}"
    geom_n = qcfi.Molecule.from_data(geom_neutral)
    geom_c = qcfi.Molecule.from_data(geom_cation)
    out = []
    psi4.core.be_quiet()
    for l in level_theory:
        m, bs = l.split("/")
        if print_file:
            create_psi4_input_file_from_args(
                geom_cation, ppm, method=l, options=options
            )
        if gather_results:
            r1 = client.add_compute(
                program="psi4",
                method=l,
                basis=bs,
                driver="energy",
                molecule=[geom_n],
                keywords=kw_id,
                protocols={"wavefunction": "orbitals_and_eigenvalues"},
            )
            r2 = client.add_compute(
                program="psi4",
                method=l,
                basis=bs,
                driver="energy",
                molecule=[geom_c],
                keywords=kw_id,
                protocols={"wavefunction": "orbitals_and_eigenvalues", "stdout": True},
            )
            try:
                id1 = int(r1.ids[0])
                ret1 = client.query_results(id=r1.ids)[0]
                id2 = int(r2.ids[0])
                orbs = ret1.get_wavefunction("eigenvalues_a").flatten()
                orbs_b = ret1.get_wavefunction("eigenvalues_b").flatten()
                result1 = ret1.dict()
                nalpha = result1["properties"]["calcinfo_nalpha"]
                nbeta = result1["properties"]["calcinfo_nbeta"]
                if nalpha >= nbeta:
                    orbs = ret1.get_wavefunction("eigenvalues_a").flatten()
                    HOMO = orbs.flatten()[nalpha - 1]
                else:
                    orbs = ret1.get_wavefunction("eigenvalues_b").flatten()
                    HOMO = orbs.flatten()[nbeta - 1]
                E_n = ret1.dict()["properties"]["return_energy"]
                out.append(E_n)
            except (AttributeError, TypeError):
                fn = f"error_dimers/{id_label}_{mon}_n.in"
                create_psi4_input_file_from_args(
                    geom_neutral, ppm, l, options=options, fn=fn
                )
                out.append(None)
            try:
                ret2 = client.query_results(id=r2.ids)[0]
                E_c = ret2.dict()["properties"]["return_energy"]
                out.append(E_c)
            except (AttributeError, TypeError):
                fn = f"error_dimers/{id_label}_{mon}_c.in"
                create_psi4_input_file_from_args(
                    geom_cation, ppm, l, options=options, fn=fn
                )
                out.append(None)
        else:
            client.add_compute(
                program="psi4",
                method=l,
                basis=bs,
                driver="energy",
                molecule=[geom_n],
                keywords=kw_id,
                protocols={"wavefunction": "orbitals_and_eigenvalues"},
            )
            client.add_compute(
                program="psi4",
                method=l,
                basis=bs,
                driver="energy",
                molecule=[geom_c],
                keywords=kw_id,
                protocols={"wavefunction": "orbitals_and_eigenvalues"},
            )
        return out


def collect_saptdft_mons_individually(js: jobspec.saptdft_mon_grac_js) -> []:
    """
    collect_saptdft_mons_individually collects data for each monomer A or B
    """
    # options = {
    #     "reference": "uhf",
    #     "level_shift": 0.2,
    #     "level_shift_cutoff": 1e-5,
    #     "MAXITER": 100,
    #     "SCF_INITIAL_ACCELERATOR": "ADIIS",
    #     "E_CONVERGENCE": 5,
    #     "D_CONVERGENCE": 5,
    # }
    # options = {
    #     "reference": "uhf",
    #     "level_shift": 0.1,
    #     "level_shift_cutoff": 1e-5,
    #     "MAXITER": 500,
    #     "SCF_INITIAL_ACCELERATOR": "ADIIS",
    #     "E_CONVERGENCE": 5,
    #     "D_CONVERGENCE": 5,
    # }
    # options = {
    #     "reference": "uhf",
    #     "level_shift": 0.1,
    #     "level_shift_cutoff": 1e-5,
    #     "MAXITER": 500,
    #     "SCF_INITIAL_ACCELERATOR": "ADIIS",
    #     "E_CONVERGENCE": 5,
    #     "D_CONVERGENCE": 5,
    # }
    options = {
        "reference": "uhf",
        "level_shift": 0.2,
        "level_shift_cutoff": 1e-4,
        "MAXITER": 300,
        "SCF_INITIAL_ACCELERATOR": "ADIIS",
        "E_CONVERGENCE": 6,
        "D_CONVERGENCE": 6,
    }
    if mon == "A":
        ms = []
        for i in js.monAs:
            ms.append(js.geometry[i, :])
    elif mon == "B":
        ms = []
        for i in js.monBs:
            ms.append(js.geometry[i, :])
    else:
        print("Invalid mon selected: should be 'A' or 'B'")
        return []
    ms = np_carts_to_string(ms)
    out = run_dft_neutral_cation_qca_qcng_error(
        js.client,
        ms,
        charges=js.charges[2],
        ppm=js.mem,
        id_label=js.id_label,
        mon=mon,
        level_theory=js.level_theory,
        gather_results=True,
        options=options,
    )
    return out


def run_saptdft_with_grads(
    client,
    M1,
    M2,
    charges,
    ppm,
    level_theory,
    gather_results=True,
    sapt_dft_grac_shift_a: float = 1e-16,
    sapt_dft_grac_shift_b: float = 1e-16,
    print_file: bool = True,
) -> np.array:
    """
    run_dft_neutral_cation
    """
    import qcfractal.interface as qcfi

    geom_str = f"{charges[1][0]} {charges[1][1]}\n{M1}--\n"
    geom_str += f"{charges[2][0]} {charges[2][1]}\n{M2}"
    # geom = f"{charges[1][0]} {charges[1][1]}\n{M1}{M2}"
    geom = qcfi.Molecule.from_data(geom_str)
    out = []
    psi4.core.be_quiet()
    for l in level_theory:
        m, bs = l.split("/")
        options = {
            # "reference": "rhf",
            "basis": bs,
            "sapt_dft_grac_shift_a": sapt_dft_grac_shift_a,
            "sapt_dft_grac_shift_b": sapt_dft_grac_shift_b,
            "SAPT_DFT_FUNCTIONAL": m,
        }
        # kw = qcfi.models.KeywordSet(
        #     **{
        #         "values": {
        #             # "reference": "rhf", # rks
        #             "basis": bs,
        #             "sapt_dft_grac_shift_a": sapt_dft_grac_shift_a,
        #             "sapt_dft_grac_shift_b": sapt_dft_grac_shift_b,
        #             "SAPT_DFT_FUNCTIONAL": m,
        #         }
        #     }
        # )
        kw = qcfi.models.KeywordSet(**{"values": options})
        kw_id = client.add_keywords([kw])[0]
        if gather_results:
            if print_file:
                create_psi4_input_file_from_args(
                    geom_str, ppm, method="sapt(dft)", options=options
                )
            print(f"{kw = }")
            r1 = client.add_compute(
                program="psi4",
                method="sapt(dft)",
                basis=bs,
                driver="energy",
                molecule=[geom],
                keywords=kw_id,
                # protocols={"stdout": True, "stderr": True},
            )
            print(f"{r1 = }")

            try:
                id1 = int(r1.ids[0])
                ret1 = client.query_results(id=r1.ids)[0]
                pp(ret1)
                result1 = ret1.dict()
                # print(f"{result1 = }")
                pp(result1["extras"]["qcvars"])
                qcvars = ret1.dict()["extras"]["qcvars"]
                # pp(ret1.dict())
                v = np.array(
                    [
                        qcvars["SAPT ELST ENERGY"],
                        qcvars["SAPT EXCH ENERGY"],
                        qcvars["SAPT DISP ENERGY"],
                        qcvars["SAPT IND ENERGY"],
                        qcvars["SAPT TOTAL ENERGY"],
                    ]
                )
                # E_n = ret1.dict()["properties"]["return_energy"]
                # grac = E_c - E_n + HOMO
                # out.append(E_n)
                out.append(v)
            except (AttributeError, TypeError, KeyError):
                print(f"job incomplete")
                v = [
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
                out.append(v)
        else:
            client.add_compute(
                program="psi4",
                method="sapt(dft)",
                basis=bs,
                driver="energy",
                molecule=[geom],
                keywords=kw_id,
                # protocols={"stdout": True, "stderr": True},
                # protocols={"wavefunction": "orbitals_and_eigenvalues"},
            )
            v = [
                None,
                None,
                None,
                None,
                None,
            ]
            out.append(v)
        return out


def run_saptdft_qcng(js: jobspec.saptdft_js) -> np.array:
    """
    run_saptdft computes scaling factor and uses to run SAPT-DFT
    """
    ma, mb = [], []
    for i in js.monAs:
        ma.append(js.geometry[i, :])
    for i in js.monBs:
        mb.append(js.geometry[i, :])
    ma = np_carts_to_string(ma)
    mb = np_carts_to_string(mb)
    print(js.id_label)
    options = {
        "reference": "uhf",
        "level_shift": 0.1,
        "level_shift_cutoff": 1e-5,
        "MAXITER": 500,
        "SCF_INITIAL_ACCELERATOR": "ADIIS",
        "E_CONVERGENCE": 5,
        "D_CONVERGENCE": 5,
    }
    options = {
        "reference": "uhf",
        "level_shift": 0.1,
        "level_shift_cutoff": 1e-5,
        "MAXITER": 200,
        "SCF_INITIAL_ACCELERATOR": "ADIIS",
        "E_CONVERGENCE": 5,
        "D_CONVERGENCE": 4,
    }
    nca = run_dft_neutral_cation_qca_qcng(
        js.client,
        ma,
        charges=js.charges[1],
        ppm=js.mem,
        level_theory=js.level_theory,
        gather_results=True,
        # print_file=True,
        options=options,
    )
    ncb = run_dft_neutral_cation_qca_qcng(
        js.client,
        mb,
        charges=js.charges[2],
        ppm=js.mem,
        level_theory=js.level_theory,
        gather_results=True,
        print_file=True,
        options=options,
    )
    out = []
    # print('\nsapt\n', len(nca), range(len(nca)//4))
    for i in range(len(nca) // 4):
        # print('sapt(dft)')
        if nca[i * 4 + 3] == None or ncb[i * 4 + 3] == None:
            out.extend([None])
            continue
        vs = run_saptdft_with_grads(
            js.client,
            ma,
            mb,
            charges=js.charges,
            ppm=js.mem,
            level_theory=js.level_theory,
            gather_results=False,
            sapt_dft_grac_shift_a=nca[i * 4 + 3],
            sapt_dft_grac_shift_b=ncb[i * 4 + 3],
        )
        out.extend(vs)
    return out
