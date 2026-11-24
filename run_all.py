#!/usr/bin/env python3
"""Lab 6 (variant 3) measurement series. Usage:
    python run_all.py [--cphys N] [--force] [--only PREFIX]
Resumable: a run whose result file exists in results/ is skipped (use --force to redo).
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data", "organamnist.npz")
URL = "https://zenodo.org/records/10519652/files/organamnist.npz?download=1"
TRAIN = os.path.join(HERE, "train.py")
PMAX, B, KC, SF, REPS = 3, 96, 20, 50, 3

ap = argparse.ArgumentParser()
ap.add_argument("--cphys", type=int, default=None, help="physical cores (if auto-detection fails)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--only", default="")
args = ap.parse_args()
os.environ.setdefault("USE_LIBUV", "0")  # Windows builds of PyTorch have no libuv support in TCPStore
os.makedirs(RES, exist_ok=True)


def p(name):
    return os.path.join(RES, name + ".json")


LAST_ERR = [""]


def run(cmd, name, show=True, log=None, softfail=False):
    if args.only and not name.startswith(args.only):
        return
    if os.path.exists(p(name)) and not args.force:
        print("skip", name, flush=True)
        return
    t0 = time.time()
    print(">>", name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    LAST_ERR[0] = r.stdout + r.stderr
    if r.returncode != 0:
        if softfail:
            return False
        print(r.stdout[-3000:])
        print(r.stderr[-4000:])
        print(f"!! FAILED: {name} (code {r.returncode}). Send this output back.")
        sys.exit(1)
    if show:
        for line in r.stdout.splitlines():
            if line.startswith("[rank0]") or line.startswith("{"):
                print("  ", line[:300])
    print(f"   {time.time() - t0:.0f} s", flush=True)
    return True


# ---- data
if not os.path.exists(DATA):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    print("downloading", URL, flush=True)
    urllib.request.urlretrieve(URL, DATA)
import numpy as np
d = np.load(DATA)
print("data:", {k: d[k].shape for k in d.files}, flush=True)

# ---- environment
py = sys.executable
run([py, TRAIN, "--mode", "info", "--out", p("env_raw")], "env_raw")
env = json.load(open(p("env_raw"), encoding="utf-8"))
cphys = args.cphys or env.get("phys")
if not cphys:
    sys.exit("Could not detect physical cores: run with --cphys N")
nth = max(1, cphys // PMAX)
env["C_phys"], env["n_th"] = cphys, nth
json.dump(env, open(p("env"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"C_phys={cphys}  n_th={nth}", flush=True)
common = ["--data", DATA, "--nth", str(nth)]


def single(mode, name, b=B, extra=()):
    run([py, TRAIN, "--mode", mode, "--b", str(b), "--out", p(name), *common, *extra], name)


USE_LAUNCHER = [False]
LAUNCH = os.path.join(HERE, "launch.py")


def tr(P, mode, name, b, extra=(), restarts=0, static=False, log=None):
    """torchrun if it works; otherwise (Windows PyTorch without libuv) launch.py with the same semantics."""
    targs = [TRAIN, "--mode", mode, "--b", str(b), "--out", p(name), *common, *extra]
    if not USE_LAUNCHER[0]:
        if static:
            rd = ["--nnodes=1", f"--max-restarts={restarts}", "--rdzv-backend=static", "--master-addr=127.0.0.1",
                  "--master-port=29500", "--node-rank=0"]
        else:
            rd = ["--standalone", "--rdzv-conf=use_libuv=0"]
        ok = run([py, "-m", "torch.distributed.run", *rd, f"--nproc_per_node={P}", *targs], name, log=log, softfail=True)
        if ok is not False:
            return
        if "libuv" not in LAST_ERR[0] and "RendezvousConnectionError" not in LAST_ERR[0]:
            print(LAST_ERR[0][-4000:])
            sys.exit(f"!! FAILED: {name}. Send this output back.")
        print("torchrun rendezvous does not work on this PyTorch build -> using launch.py (same semantics)", flush=True)
        USE_LAUNCHER[0] = True
        open(os.path.join(RES, "launcher.txt"), "w", encoding="utf-8").write(
            "torchrun failed with a libuv/rendezvous error; launch.py was used instead.\n")
    run([py, LAUNCH, "--nproc", str(P), "--max-restarts", str(restarts), "--port", "29500" if static else "0", "--", *targs],
        name, log=log)


# ---- step 1: one process
single("single", "single_train")
for i in range(1, REPS + 1):
    single("single_speed", f"single_speed_{i}")

# ---- step 2: manual sync, equivalence
single("refgrad", "refgrad")
tr(PMAX, "equiv", "equiv", B // PMAX)
for i in range(1, REPS + 1):
    tr(PMAX, "manual_speed", f"manual_speed_{i}", B // PMAX)

# ---- step 3: strong scaling (B = 96), sync and no_sync
for P in (1, 2, 3):
    for i in range(1, REPS + 1):
        tr(P, "ddp_speed", f"ddp_speed_P{P}_{i}", B // P)
        tr(P, "ddp_speed", f"ddp_speed_ns_P{P}_{i}", B // P, ["--no_sync"])

# ---- step 4: weak scaling (b = 32); P = 3 is taken from strong scaling
b = B // PMAX
for P in (1, 2):
    for i in range(1, REPS + 1):
        tr(P, "ddp_speed", f"weak_P{P}_{i}", b)
tr(PMAX, "ddp_train", "train_P3_lr0.02", b, ["--lr", "0.02"])
tr(1, "ddp_train", "train_P1_lr0.02", b, ["--lr", "0.02"])
tr(PMAX, "ddp_train", "train_P3_lr0.06", b, ["--lr", "0.06"])

# ---- step 5: failure and recovery
ck = os.path.join(RES, "fail_ckpt")
if not os.path.exists(p("fail")) or args.force:
    if (not args.only) or "fail".startswith(args.only) or args.only.startswith("fail"):
        shutil.rmtree(ck, ignore_errors=True)
        os.makedirs(ck)
        tr(PMAX, "ddp_train", "fail", b,
           ["--lr", "0.02", "--ckpt", str(KC), "--fail_step", str(SF), "--ckpt_dir", ck],
           restarts=1, static=True, log=os.path.join(RES, "fail_stdout.txt"))
        try:
            os.remove(os.path.join(ck, "ckpt.pt"))  # 26 MB, not needed in the report
        except OSError:
            pass
print("DONE ->", RES)
