#!/usr/bin/env python3
"""Builds tables 1-6 and figures from results/ (median of 3 repeats for speed measurements)."""
import csv, glob, json, os, statistics as st, sys

R = sys.argv[1] if len(sys.argv) > 1 else "results"
med = st.median
J = lambda n: json.load(open(os.path.join(R, n + ".json"), encoding="utf-8"))
reps = lambda pat: [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(R, pat)))]
M = lambda runs, k: med(r[k] for r in runs)
out = []
P_ = out.append
row = lambda *c: "| " + " | ".join(str(x) for x in c) + " |"
f = lambda x, d=2: f"{x:,.{d}f}".replace(",", " ")
PMAX, B = 3, 96

env, s1, eq = J("env"), J("single_train"), J("equiv")
ss, ms = reps("single_speed_*.json"), reps("manual_speed_*.json")
strong = {P: reps(f"ddp_speed_P{P}_*.json") for P in (1, 2, 3)}
strong_ns = {P: reps(f"ddp_speed_ns_P{P}_*.json") for P in (1, 2, 3)}
weak = {1: reps("weak_P1_*.json"), 2: reps("weak_P2_*.json"), 3: strong[3]}
tr3, tr1, tr3b = J("train_P3_lr0.02"), J("train_P1_lr0.02"), J("train_P3_lr0.06")
fl = J("fail")

# ---- Table 1
P_("### Таблиця 1. Стенд\n")
P_(row("Процесор", "Фізичних ядер C_phys", "Логічних ядер", "Памʼять, ГБ", "ОС", "Python", "PyTorch", "n_th"))
P_(row(*["---"] * 8))
P_(row(env["cpu"], env["C_phys"], env["logical"], env["ram_gb"], env["os"], env["python"], env["torch"], env["n_th"]))
P_("")
P_(row("Датасет", "K", "C", "c1 / c2 / h", "n_params", "M, МіБ", "N", "B", "P_max"))
P_(row(*["---"] * 9))
P_(row("OrganAMNIST (Д3)", 11, 1, "64 / 128 / 512", f(s1["n_params"], 0), f(s1["M_mib"], 2), 7000, B, PMAX))
P_("")

# ---- Table 2
t1, X1 = M(ss, "t_step_ms"), M(ss, "X")
tm, sm, Xm = M(ms, "t_step_ms"), M(ms, "t_sync_ms"), M(ms, "X")
td, Xd = M(strong[3], "t_step_ms"), M(strong[3], "X")
tn, Xn = M(strong_ns[3], "t_step_ms"), M(strong_ns[3], "X")
comm = td - tn
P_("### Таблиця 2. Синхронізація градієнтів\n")
P_(row("Режим", "t_step, мс", "Час обміну на крок, мс", "Частка кроку в обміні", "X, прикл./с"))
P_(row(*["---"] * 5))
P_(row("1 процес, батч B", f(t1), 0, 0, f(X1, 0)))
P_(row("P_max процесів, ручна синхронізація", f(tm), f(sm), f(sm / tm, 3), f(Xm, 0)))
P_(row("P_max процесів, DDP", f(td), f(comm), f(comm / td, 3), f(Xd, 0)))
P_(row("P_max процесів, DDP, no_sync", f(tn), 0, 0, f(Xn, 0)))
P_("")
P_(row("max abs(g − ḡ) / max abs(g)", "Різниця сум ваг між процесами після 50 кроків", "ACC, 1 процес", "ACC, DDP на P_max",
       "AUC, 1 процес", "AUC, DDP на P_max"))
P_(row(*["---"] * 6))
P_(row(f"{eq['max_ratio']:.2e}", f"{eq['sum_diff']:.2e}", f(s1["acc"], 4), f(tr3["acc"], 4), f(s1["auc"], 4), f(tr3["auc"], 4)))
P_("")

# ---- Table 3
X_1 = M(strong[1], "X")
P_("### Таблиця 3. Strong scaling, B = const (96)\n")
P_(row("P", "b", "t_step, мс", "t_step,ns, мс", "X, прикл./с", "S", "E", "e", "c"))
P_(row(*["---"] * 9))
S = {}
for P in (1, 2, 3):
    t, tns, X = M(strong[P], "t_step_ms"), M(strong_ns[P], "t_step_ms"), M(strong[P], "X")
    s_ = X / X_1
    S[P] = (X, s_)
    e = "−" if P == 1 else f((1 / s_ - 1 / P) / (1 - 1 / P), 3)
    P_(row(P, B // P, f(t), f(tns), f(X, 0), f(s_, 3), f(s_ / P, 3), e, f((t - tns) / t, 3)))
P_("")

# ---- Table 4
Xw1 = M(weak[1], "X")
P_("### Таблиця 4. Weak scaling, b = 32\n")
P_(row("P", "B_P", "t_step, мс", "X, прикл./с", "S_w", "E_w"))
P_(row(*["---"] * 6))
W = {}
for P in (1, 2, 3):
    X = M(weak[P], "X")
    W[P] = X
    P_(row(P, 32 * P, f(M(weak[P], "t_step_ms")), f(X, 0), f(X / Xw1, 3), f(X / Xw1 / P, 3)))
P_("")

# ---- Table 5
P_("### Таблиця 5. Якість за різних глобальних батчів\n")
P_(row("Режим", "B_P", "η", "Кроків оптимізатора за n_ep епох", "ACC", "AUC", "Найбільша втрата за останню епоху"))
P_(row(*["---"] * 7))
for lab, r in (("P = 1", tr1), ("P = P_max, з кроку 2", tr3), ("P = P_max, лінійне правило", tr3b)):
    P_(row(lab, r["B"], r["lr"], r["steps_total"], f(r["acc"], 4), f(r["auc"], 4), f(r["max_loss_last_epoch"], 4)
           + (" (розходження)" if r["diverged"] else "")))
P_("")

# ---- Table 6
ckl = [json.loads(l) for l in open(os.path.join(R, "fail_ckpt", "ckpt_log.jsonl"), encoding="utf-8")]
mark = json.load(open(os.path.join(R, "fail_ckpt", "failed.json"), encoding="utf-8"))
steps = list(csv.reader(open(os.path.join(R, "fail_ckpt", "steps.csv"), encoding="utf-8")))
steps = [(int(a), int(s), float(l), float(t)) for a, s, l, t in steps]
sf, tf = mark["step"], mark["t_f"]
last = [c for c in ckl if c["attempt"] == 0 and c["t_c"] < tf][-1]
done_before = max(s for a, s, l, t in steps if a == 0)
redone = done_before - last["step"]
t1st = min(t for a, s, l, t in steps if a == 1)
t_lost = t1st - last["t_c"]
P_("### Таблиця 6. Відмова і відновлення\n")
P_(row("K_c", "s_f", "Крок останньої контрольної точки", "Кроків виконано повторно", "t_lost, с", "t_f − t_c, с", "t_r, с"))
P_(row(*["---"] * 7))
P_(row(20, sf, last["step"], redone, f(t_lost, 2), f(tf - last["t_c"], 2), f(t1st - tf, 2)))
P_("")
final = {}
for a, s, l, t in steps:
    final[s] = l  # the last attempt overwrites
ref = tr3["losses"]
rng = range(sf - 2, sf + 3)
P_(row("Крок", *[f"s_f{d:+d}" if d else "s_f" for d in range(-2, 3)], "ACC після n_ep епох"))
P_(row(*["---"] * 7))
P_(row("Втрата, прогін з відмовою", *[f(final[s], 6) for s in rng], f(fl["acc"], 4)))
P_(row("Втрата, прогін без відмови", *[f(ref[s - 1], 6) for s in rng], f(tr3["acc"], 4)))
P_("")
md = max(abs(final[s] - ref[s - 1]) for s in final)
P_(f"Найбільша абсолютна різниця втрати між прогонами за всіма кроками: {md:.2e}. "
   f"Кроків за прогін з відмовою: {fl['steps_total']}, без відмови: {tr3['steps_total']}.\n")

open(os.path.join(R, "tables.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))

# ---- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c1, c2, cg = "#2a6fb0", "#d9822b", "#888888"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    Ps = [1, 2, 3]
    ax[0].plot(Ps, [S[P][0] for P in Ps], "o-", color=c1, label="DDP")
    ax[0].plot(Ps, [P * X_1 for P in Ps], "--", color=cg, label="ідеал P·X(1)")
    ax[0].set_title("Strong scaling (B = 96)")
    ax[1].plot(Ps, [W[P] for P in Ps], "o-", color=c1, label="DDP")
    ax[1].plot(Ps, [P * Xw1 for P in Ps], "--", color=cg, label="ідеал P·X(1)")
    ax[1].set_title("Weak scaling (b = 32)")
    for a in ax[:2]:
        a.set_xlabel("P"); a.set_ylabel("X, прикл./с"); a.set_xticks(Ps); a.legend(frameon=False)
    for lab, r, c in (("P=1, b=32, η=0.02", tr1, c1), ("P=3, B=96, η=0.02", tr3, c2), ("P=3, B=96, η=0.06", tr3b, "#3a9d5d")):
        ax[2].plot(range(1, len(r["losses"]) + 1), r["losses"], color=c, lw=1, label=lab)
    ax[2].set_xlabel("крок оптимізатора"); ax[2].set_ylabel("втрата"); ax[2].set_title("Втрата на кроці"); ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "lab6_overview.png"), dpi=150)
except Exception as e:
    print("plot skipped:", e)
