import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def check_for_empty_df_rows(
    df_p="data/dimers_10k.pkl",
    col_check={
        "vac_multipole_A": 0,
        "vac_multipole_B": 0,
        "environment_multipole_A": 0,
        "environment_multipole_B": 0,
        "vac_widths_A": 0,
        "vac_widths_B": 0,
        "vac_vol_rat_A": 0,
        "vac_vol_rat_B": 0,
    },
) -> None:
    """
    check_for_empty_df_rows
    """
    df = pd.read_pickle(df_p)
    zeros = 0
    flawed_inds = []
    for idx, r in df.iterrows():
        for k in col_check.keys():
            a = df.iloc[idx][k]
            if np.array_equal(a, np.zeros(5)):
                col_check[k] += 1
                if idx not in flawed_inds:
                    flawed_inds.append(idx)
    print(col_check)
    print(flawed_inds)


def get_mon_inds(ra, rb, monA=True):
    la = len(ra)
    lb = len(rb)
    if monA:
        return range(la)
    return range(la, la + lb)


def calc_charge_dif(charge_dimer, charge_mon, ind_mon):
    c_d = charge_dimer[ind_mon, 0]
    c_m = charge_mon[:, 0]
    dif = c_d - c_m
    dif_sum = np.sum(dif)
    return dif_sum


def T_cart(RA, RB):

    dR = RB - RA
    R = np.linalg.norm(dR)

    delta = np.identity(3)

    T0 = R**-1
    T1 = (R**-3) * (-1.0 * dR)
    T2 = (R**-5) * (3 * np.outer(dR, dR) - R * R * delta)

    Rdd = np.multiply.outer(dR, delta)
    T3 = (
        (R**-7)
        * -1.0
        * (
            15 * np.multiply.outer(np.outer(dR, dR), dR)
            - 3 * R * R * (Rdd + Rdd.transpose(1, 0, 2) + Rdd.transpose(2, 0, 1))
        )
    )

    RRdd = np.multiply.outer(np.outer(dR, dR), delta)
    dddd = np.multiply.outer(delta, delta)
    T4 = (R**-9) * (
        105 * np.multiply.outer(np.outer(dR, dR), np.outer(dR, dR))
        - 15
        * R
        * R
        * (
            RRdd
            + RRdd.transpose(0, 2, 1, 3)
            + RRdd.transpose(0, 3, 2, 1)
            + RRdd.transpose(2, 1, 0, 3)
            + RRdd.transpose(3, 1, 2, 0)
            + RRdd.transpose(2, 3, 0, 1)
        )
        + 3
        * (R**4)
        * (dddd + dddd.transpose(0, 2, 1, 3) + dddd.transpose(0, 3, 2, 1))
    )

    return T0, T1, T2, T3, T4


def eval_interaction(RA, qA, muA, thetaA, RB, qB, muB, thetaB):
    T0, T1, T2, T3, T4 = T_cart(RA, RB)
    # Trace is already taken care of if False: traceA = np.trace(thetaA) thetaA[0,0] -= traceA / 3.0 thetaA[1,1] -= traceA / 3.0 thetaA[2,2] -= traceA / 3.0 traceB = np.trace(thetaB) thetaB[0,0] -= traceB / 3.0 thetaB[1,1] -= traceB / 3.0 thetaB[2,2] -= traceB / 3.0

    E_qq = np.sum(T0 * qA * qB)
    E_qu = np.sum(T1 * (qA * muB - qB * muA))
    E_qQ = np.sum(T2 * (qA * thetaB + qB * thetaA)) * (1.0 / 3.0)

    E_uu = np.sum(T2 * np.outer(muA, muB)) * (-1.0)
    E_uQ = np.sum(
        T3 * (np.multiply.outer(muA, thetaB) - np.multiply.outer(muB, thetaA))
    ) * (
        -1.0 / 3.0
    )  # sign ??

    E_QQ = np.sum(T4 * np.multiply.outer(thetaA, thetaB)) * (1.0 / 9.0)

    # partial-charge electrostatic energy
    E_q = E_qq

    # dipole correction
    E_u = E_qu + E_uu

    # quadrupole correction
    E_Q = E_qQ + E_uQ + E_QQ
    return E_q + E_u + E_Q


# def deconstruct_multipoles(multipoles, mol_len):
#     t = np.array([[[0,1,2],[3,4,5],[6,7,8]]])
#     quad = [q[np.triu_indices(3)] for q in t]
#     q, mu, theta = 0, 0, 0
#     q = multipoles[:,0]
#     mu = multipoles[:, 1:4]
#     theta_c = multipoles[:, 4:]
#     theta = np.zeros((mol_len, 3, 3))
#     for i in range(mol_len):
#         v = theta_c[i]
#         size_X = 3
#         X = np.zeros((size_X,size_X))
#         X[np.triu_indices(X.shape[0], k = 0)] = v
#         X = X + X.T - np.diag(np.diag(X))
#         theta[i, :, :] = X
#     return q, mu, theta


def deconstruct_multipoles(multipole):
    q = multipole[0]
    mu = multipole[1:4]
    v = multipole[4:]
    size_X = 3
    X = np.zeros((size_X, size_X))
    X[np.triu_indices(X.shape[0], k=0)] = v
    X = X + X.T - np.diag(np.diag(X))
    theta = X
    return q, mu, theta


def calculate_energy_from_multipoles_static(RA, RB, multipole_A, multipole_B):
    lA = len(RA)
    lB = len(RB)
    tot_energy = 0
    for ia in range(lA):
        for ib in range(lB):
            rA = RA[ia]
            qA, muA, thetaA = deconstruct_multipoles(multipole_A[ia])
            rB = RB[ib]
            qB, muB, thetaB = deconstruct_multipoles(multipole_B[ib])
            pair_energy = eval_interaction(
                rA,
                qA,
                muA,
                thetaA,
                rB,
                qB,
                muB,
                thetaB,
            )
            tot_energy += pair_energy

    Har2Kcalmol = 627.5094737775374055927342256
    E = tot_energy * Har2Kcalmol
    # print("E:", E)
    return E


def calculate_energy_from_multipoles(RA, RB, multipole_AB):
    lA = len(RA)
    lB = len(RB)
    multipole_A = multipole_AB[:lA]
    multipole_B = multipole_AB[lA:]
    tot_energy = 0
    for ia in range(lA):
        for ib in range(lB):
            rA = RA[ia]
            qA, muA, thetaA = deconstruct_multipoles(multipole_A[ia])
            rB = RB[ib]
            qB, muB, thetaB = deconstruct_multipoles(multipole_B[ib])
            pair_energy = eval_interaction(
                rA,
                qA,
                muA,
                thetaA,
                rB,
                qB,
                muB,
                thetaB,
            )
            tot_energy += pair_energy

    Har2Kcalmol = 627.5094737775374055927342256
    E = tot_energy * Har2Kcalmol
    # print("E:", E)
    return E


def charges_exploration(
    df_p="100k_charges_test.pkl",
):
    df = pd.read_pickle(df_p)
    print(df.columns.values)
    # TODO: eval classic induction with eval_interaction
    df["static_q_vac"] = df.apply(
        lambda r: calculate_energy_from_multipoles_static(
            r["RA"], r["RB"], r["vac_multipole_A"], r["vac_multipole_B"]
        ),
        axis=1,
    )
    # df["static_q"] = df.apply(
    #     lambda r: calculate_energy_from_multipoles_static(
    #         r["RA"], r["RB"], r["environment_multipole_A"], r["environment_multipole_B"]
    #     ),
    #     axis=1,
    # )
    # df["changing_q"] = df.apply(
    #     lambda r: calculate_energy_from_multipoles(
    #         r["RA"], r["RB"], r["vac_multipole_AB"],
    #     ),
    #     axis=1,
    # )
    df.to_pickle("100k_charges_test2.pkl")
    # q: charges
    # mu: dipoles
    # theta: quadrapoles
    return


def size_shrink(s1, s2):
    if s1 == 1 or s2 == 1:
        return 0
    return 1


def plot_charge_transfer(
    id_list=[0, 1, 2],
    # df_p="data/dimers_100k.pkl",
    df_p="100k_charges_test2.pkl",
):
    df = pd.read_pickle(df_p)
    print(df.columns.values)
    # df["main_id2"] = df["main_id"]
    # df = df.set_index("main_id")
    # df = df.iloc[id_list]
    # df["charge_mon_A_dif_dimer_mon"] = df.apply(
    #     lambda r: calc_charge_dif(
    #         r["vac_multipole_AB"],
    #         r["vac_multipole_A"],
    #         get_mon_inds(r["RA"], r["RB"]),
    #     ),
    #     axis=1,
    # )
    # df["charge_mon_B_dif_dimer_mon"] = df.apply(
    #     lambda r: calc_charge_dif(
    #         r["vac_multipole_AB"],
    #         r["vac_multipole_B"],
    #         get_mon_inds(r["RA"], r["RB"], monA=False),
    #     ),
    #     axis=1,
    # )
    # df.to_pickle("test_charges.pkl")
    results = {
        "Monomer": ["monA", "monB"],
        "Mean Difference": [
            df["charge_mon_A_dif_dimer_mon"].mean(),
            df["charge_mon_B_dif_dimer_mon"].mean(),
        ],
        "# Points Outside 0.1": [
            (df["charge_mon_A_dif_dimer_mon"] > 0.1).sum(),
            (df["charge_mon_B_dif_dimer_mon"] > 0.1).sum(),
        ],
        "# Points Outside 0.01": [
            (df["charge_mon_A_dif_dimer_mon"] > 0.01).sum(),
            (df["charge_mon_B_dif_dimer_mon"] > 0.01).sum(),
        ],
    }
    # print(pd.DataFrame(results))
    df["diff"] = abs(df["static_q"] - df["changing_q"])
    cnt = (df["diff"] > 5).sum()
    print(f"Greater than 5 kcal/mol difference cnt = {cnt}")
    print(f'MAD: {df["diff"].mad()}')
    print(f'MAD 2nd and 3rd qunatile: {df["diff"].quantile([0.25, 0.75]).mad()}')
    print(f'median: {df["diff"].median()}')
    # df.plot(
    #     x="static_q",
    #     y="changing_q",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="Induction interaction energy (kcal/mol)"
    # )
    # plt.ylabel("changing_q (vac multipoles AB) E (kcal/mol)")
    # plt.xlabel("static_q E (environment multipoles A and B) (kcal/mol)")
    # xs = [i for i in range(int(df['static_q'].min()), int(df['static_q'].max()))]
    # z = np.polyfit(df['static_q'], df['changing_q'], 1)
    # p = np.poly1d(z)
    # plt.plot(xs, p(xs), label=f"trendline: y=%.6fx+%.6f"%(z[0], z[1]))
    # plt.plot(xs, xs, label="line: y = x")
    # plt.legend()
    # plt.savefig("static_vs_changing_q.png")
    # df['diff'] = abs(df['static_q_vac'] - df['changing_q'])
    #
    # cnt = (df['diff'] > 5).sum()
    # print(f"Greater than 5 kcal/mol difference cnt = {cnt}")
    # print(f'MAD: {df["diff"].mad()}')
    # print(f'MAD 2nd and 3rd qunatile: {df["diff"].quantile([0.25, 0.75]).mad()}')
    # print(f'median: {df["diff"].median()}')
    # df.plot(
    #     x="static_q_vac",
    #     y="changing_q",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="Induction interaction energy (kcal/mol)"
    # )
    # plt.ylabel("changing_q (vac multipoles AB) E (kcal/mol)")
    # plt.xlabel("static_q_vac E (vacuum multipoles A and B) (kcal/mol)")
    # xs = [i for i in range(int(df['static_q_vac'].min()), int(df['static_q_vac'].max()))]
    # z = np.polyfit(df['static_q_vac'], df['changing_q'], 1)
    # p = np.poly1d(z)
    # plt.plot(xs, p(xs), label=f"trendline: y=%.6fx+%.6f"%(z[0], z[1]))
    # plt.plot(xs, xs, label="line: y = x")
    # plt.legend()
    # plt.savefig("static_vac_vs_changing_q.png")
    #
    # df['ClassicalInd'] = df['static_q'] - df['changing_q']
    # df.plot(
    #     x="Ind_aug",
    #     y="ClassicalInd",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label=""
    # )
    print(type(df.iloc[0]["TQA"]))
    print(type(df.iloc[0]["TQB"]))
    ind_chargeA = df["TQA"][df["TQA"] != np.float(0)].index.tolist()
    ind_chargeB = df["TQB"][df["TQB"] != np.float(0)].index.tolist()
    ind_charge = list(set(ind_chargeA).intersection(set(ind_chargeB)))

    ind_anionA = df["TQA"][df["TQA"] < np.float(0)].index.tolist()
    ind_anionB = df["TQB"][df["TQB"] < np.float(0)].index.tolist()
    ind_anion = list(set(ind_anionA).union(set(ind_anionB)))

    ind_neutralA = df["TQA"][df["TQA"] == np.float(0)].index.tolist()
    ind_neutralB = df["TQB"][df["TQB"] == np.float(0)].index.tolist()
    ind_neutral = list(set(ind_neutralA).intersection(set(ind_neutralB)))
    # print(ind_neutral)
    # print(len(df))

    ind_chargeA_set = set(ind_chargeA)
    ind_chargeB_set = set(ind_chargeB)
    ind_neutralA_set = set(ind_neutralA)
    ind_neutralB_set = set(ind_neutralB)
    charge_neutral1 = ind_chargeA_set.intersection(ind_neutralB_set)
    charge_neutral2 = ind_chargeB_set.intersection(ind_neutralA_set)
    ind_charge_neutral = list(charge_neutral1.union(charge_neutral2))

    df["charge_diff"] = (df["TQA"] - df["TQB"]).abs()
    df["sizeA"] = df.apply(lambda r: len(r["RA"]), axis=1)
    df["sizeB"] = df.apply(lambda r: len(r["RB"]), axis=1)
    df["sizeColor"] = df.apply(lambda r: size_shrink(r["sizeA"], r["sizeB"]), axis=1)

    neutral = df.copy(deep=True).loc[ind_neutral]
    charges_neutral = df.copy(deep=True).loc[ind_charge_neutral]
    charges = df.copy(deep=True).loc[ind_charge]
    anion = df.copy(deep=True).loc[ind_anion]
    df["ClassicalInd"] = df["static_q"] - df["static_q_vac"]
    # df.plot(x="Ind_aug", y="ClassicalInd", style="o", ms=0.5, use_index=True, label="")
    # charges["ClassicalInd"] = charges["static_q"] - charges["static_q_vac"]
    # charges.plot(
    #     x="Ind_aug",
    #     y="ClassicalInd",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="charges",
    # )
    # charges_neutral["ClassicalInd"] = (
    #     charges_neutral["static_q"] - charges_neutral["static_q_vac"]
    # )
    # charges_neutral.plot(
    #     x="Ind_aug",
    #     y="ClassicalInd",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="charges_neutral",
    # )
    # neutral["ClassicalInd"] = neutral["static_q"] - neutral["static_q_vac"]
    # neutral.plot(
    #     x="Ind_aug",
    #     y="ClassicalInd",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="neutral",
    # )
    # neutral["ClassicalInd"] = neutral["static_q"] - neutral["static_q_vac"]
    # neutral.plot(
    #     x="Ind_aug",
    #     y="ClassicalInd",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label="neutral",
    # )
    # anion["ClassicalInd"] = anion["static_q"] - anion["static_q_vac"]
    # anion.plot(
    #     x="Ind_aug", y="ClassicalInd", style="o", ms=0.5, use_index=True, label="anion"
    # )

    cmap = cm.get_cmap("Spectral")
    x = df["Ind_aug"].tolist()
    y = df["ClassicalInd"].tolist()
    c = df["charge_diff"].tolist()
    c = df["sizeColor"].tolist()
    fig, ax = plt.subplots(1)
    scat = ax.scatter(x, y, c=c, cmap=cmap, s=0.5)
    plt.colorbar(scat)
    plt.ylabel("Ind_aug E (kcal/mol)")
    plt.xlabel("ClassicalInd (kcal/mol)")
    plt.savefig("size.png")
    # df.plot(
    #     x="static_q_vac",
    #     y="static_q",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    #     label=""
    # )
    plt.show()
    # df.plot(
    #     x="main_id2",
    #     y="charge_mon_A_dif_dimer_mon",
    #     style="o",
    #     ms=0.5,
    #     use_index=True,
    # )
    # plt.savefig("charges_monA.png")
    # df.plot(
    #     x="main_id2",
    #     y="charge_mon_B_dif_dimer_mon",
    #     style="o",
    #     c="r",
    #     ms=0.5,
    #     use_index=True,
    # )
    # df.plot(
    #     x="charge_mon_A_dif_dimer_mon",
    #     y="Ind_aug",
    #     style="o",
    #     c="r",
    #     ms=0.5,
    #     use_index=True,
    # )
    # plt.show()
    # plt.savefig("charges_ind.png")
