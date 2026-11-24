#!/usr/bin/env python3
"""Lab 6, variant 3: data-parallel training of a small CNN on OrganAMNIST (CPU, gloo).

Variant 3: dataset D3 = OrganAMNIST (K=11, C=1), width Sh3 = (64,128,512), P_max=3,
B=96, K_c=20, s_f=50. N=7000 first train examples, n_ep=3, SGD momentum 0.9, eta0=0.02.

Modes (all but 'single*' / 'refgrad' are started with torchrun):
  info          write environment info
  single        train in one process WITHOUT torch.distributed (step 1)
  single_speed  speed of one process, no torch.distributed
  refgrad       gradient g of the first batch of B examples, one process
  equiv         manual gradient sync: equivalence check + 50 steps + weight sums
  manual_speed  t_step and t_sync with manual all-reduce
  ddp_speed     t_step with DDP (--no_sync: backward inside model.no_sync())
  ddp_train     train n_ep epochs with DDP (optional checkpoints and failure simulation)
"""
import argparse, json, math, os, platform, statistics, subprocess, sys, time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

DATASETS = {"D1": ("dermamnist", 7, 3), "D2": ("bloodmnist", 8, 3), "D3": ("organamnist", 11, 1),
            "D4": ("organcmnist", 11, 1), "D5": ("pathmnist", 9, 3), "D6": ("octmnist", 4, 1)}
WIDTHS = {"W1": (16, 32, 128), "W2": (32, 64, 256), "W3": (64, 128, 512)}
VARIANT = dict(ds="D3", width="W3", pmax=3, B=96, Kc=20, sf=50)
N_TRAIN, N_EP, LR0 = 7000, 3, 0.02


def load_split(path, split, n=None):
    d = np.load(path)
    x = d[f"{split}_images"][:n]
    y = d[f"{split}_labels"][:n].reshape(-1)
    if x.ndim == 3:
        x = x[:, None, :, :]
    else:
        x = x.transpose(0, 3, 1, 2)
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    x = (x / 255 - 0.5) / 0.5
    return TensorDataset(x, torch.from_numpy(y.astype(np.int64)))


class Net(nn.Module):
    def __init__(self, C, c1, c2, h, K):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(7 * 7 * c2, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, x):
        return self.head(self.features(x))


def n_params_formula(C, c1, c2, h, K):
    return 9 * C * c1 + c1 + 9 * c1 * c2 + c2 + 49 * c2 * h + h + h * K + K


def build_model():
    name, K, C = DATASETS[VARIANT["ds"]]
    c1, c2, h = WIDTHS[VARIANT["width"]]
    torch.manual_seed(0)
    m = Net(C, c1, c2, h, K)
    n = sum(p.numel() for p in m.parameters())
    assert n == n_params_formula(C, c1, c2, h, K)
    return m, n


# ------------------------------------------------------------------ helpers
def evaluate(model, test):
    from sklearn.metrics import roc_auc_score
    model.eval()
    xs, ys = test.tensors
    logits = []
    with torch.no_grad():
        for i in range(0, len(xs), 1024):
            logits.append(model(xs[i:i + 1024]))
    lg = torch.cat(logits)
    acc = (lg.argmax(1) == ys).float().mean().item()
    prob = torch.softmax(lg.double(), 1).numpy()
    try:
        auc = float(roc_auc_score(ys.numpy(), prob, multi_class="ovr", average="macro"))
    except Exception:  # diverged run: NaN probabilities
        auc = float("nan")
    model.train()
    return acc, auc


def batches(loader, sampler, epoch0=0):
    e = epoch0
    while True:
        sampler.set_epoch(e)
        for xb, yb in loader:
            yield xb, yb
        e += 1


def save(args, obj, rank=0):
    if rank == 0 and args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)


def sysinfo():
    info = {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "logical": os.cpu_count(), "cpu": platform.processor(), "phys": None, "ram_gb": None}

    def ps(cmd):
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                              text=True, encoding="utf-8").stdout.strip()
    try:
        import psutil
        info["phys"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            info["cpu"] = ps("(Get-CimInstance Win32_Processor | Select -First 1).Name")
            if not info["phys"]:
                info["phys"] = int(ps("(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"))
            if not info["ram_gb"]:
                info["ram_gb"] = round(int(ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")) / 2**30, 1)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            txt = open("/proc/cpuinfo", encoding="utf-8").read()
            info["cpu"] = [l.split(":")[1].strip() for l in txt.splitlines() if l.startswith("model name")][0]
            if not info["ram_gb"]:
                info["ram_gb"] = round(int(open("/proc/meminfo").readline().split()[1]) / 2**20, 1)
        except Exception:
            pass
    return info


# ------------------------------------------------------------------ speed
def measure(args, model, opt, world, mode):
    """mode: plain | manual | ddp. 10 warm-up + 50 measured steps."""
    train = load_split(args.data, "train", N_TRAIN)
    sampler = DistributedSampler(train, num_replicas=world, rank=dist.get_rank() if world > 1 or dist.is_initialized() else 0,
                                 shuffle=True, seed=0)
    loader = DataLoader(train, batch_size=args.b, sampler=sampler, drop_last=True)
    it = batches(loader, sampler)
    lossf = nn.CrossEntropyLoss()
    params = [p for p in model.parameters()]
    steps, syncs = [], []
    t_start = t_end = 0.0
    for s in range(60):
        xb, yb = next(it)
        if s == 10:
            t_start = time.perf_counter()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        if mode == "ddp" and args.no_sync:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
        ts = 0.0
        if mode == "manual":
            a = time.perf_counter()
            for p in params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
            ts = time.perf_counter() - a
        opt.step()
        t1 = time.perf_counter()
        if s >= 10:
            steps.append(t1 - t0)
            syncs.append(ts)
            t_end = t1
    dur = t_end - t_start
    B = args.b * world
    return {"P": world, "b": args.b, "B": B, "t_step_ms": statistics.median(steps) * 1e3,
            "t_sync_ms": statistics.median(syncs) * 1e3, "X": 50 * B / dur, "duration_s": dur,
            "no_sync": bool(args.no_sync) if mode == "ddp" else False, "nth": args.nth}


# ------------------------------------------------------------------ training
def train_loop(args, model, wrap, opt, world, rank, test):
    train = load_split(args.data, "train", N_TRAIN)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(train, batch_size=args.b, sampler=sampler, drop_last=True)
    lossf = nn.CrossEntropyLoss()
    use_dist = dist.is_initialized()
    step, epoch0, skip = 0, 0, 0
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    ck = os.path.join(args.ckpt_dir, "ckpt.pt") if args.ckpt_dir else None
    log_path = os.path.join(args.ckpt_dir, "steps.csv") if args.ckpt_dir else None
    marker = os.path.join(args.ckpt_dir, "failed.json") if args.ckpt_dir else None
    resumed = False
    if ck and os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        wrap_mod = wrap.module if isinstance(wrap, DDP) else wrap
        wrap_mod.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, epoch0, skip = st["step"], st["epoch"], st["step_in_epoch"]
        resumed = True
        if rank == 0:
            print(f"[rank0] resumed from checkpoint: step={step} epoch={epoch0} step_in_epoch={skip}", flush=True)
    losses, epoch_losses = [], {}
    t_begin = time.time()
    for epoch in range(epoch0, N_EP):
        sampler.set_epoch(epoch)
        for i, (xb, yb) in enumerate(loader):
            if i < skip:
                continue
            step += 1
            if args.fail_step and rank == world - 1 and step == args.fail_step and not os.path.exists(marker):
                with open(marker, "w", encoding="utf-8") as f:
                    json.dump({"t_f": time.time(), "step": step, "rank": rank}, f)
                print(f"[rank{rank}] simulated failure at start of step {step}", flush=True)
                os._exit(1)
            opt.zero_grad(set_to_none=True)
            loss = lossf(wrap(xb), yb)
            loss.backward()
            opt.step()
            gl = loss.detach().clone()
            if use_dist:
                dist.all_reduce(gl)
                gl /= world
            lv = gl.item()
            losses.append(lv)
            epoch_losses.setdefault(epoch, []).append(lv)
            if rank == 0 and log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{attempt},{step},{lv:.10f},{time.time():.6f}\n")
            if args.ckpt and step % args.ckpt == 0:
                if rank == 0:
                    mod = wrap.module if isinstance(wrap, DDP) else wrap
                    state = {"model": mod.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                             "step_in_epoch": i + 1, "step": step}
                    torch.save(state, ck + ".tmp")
                    os.replace(ck + ".tmp", ck)
                    with open(os.path.join(args.ckpt_dir, "ckpt_log.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "t_c": time.time(), "attempt": attempt}) + "\n")
                if use_dist:
                    dist.barrier()
        skip = 0
    res = {"P": world, "b": args.b, "B": args.b * world, "lr": args.lr, "steps_total": step,
           "steps_this_attempt": len(losses), "losses": losses, "resumed": resumed, "attempt": attempt,
           "max_loss_last_epoch": max(epoch_losses[N_EP - 1]) if (N_EP - 1) in epoch_losses else None,
           "diverged": any((not math.isfinite(x)) for x in losses), "train_time_s": time.time() - t_begin,
           "nth": args.nth}
    if rank == 0:
        mod = wrap.module if isinstance(wrap, DDP) else wrap
        res["acc"], res["auc"] = evaluate(mod, test)
        print(f"[rank0] steps={step} ACC={res['acc']:.4f} AUC={res['auc']:.4f} "
              f"max_loss_last_epoch={res['max_loss_last_epoch']}", flush=True)
    if use_dist:
        dist.barrier()
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--data", default="data/organamnist.npz")
    ap.add_argument("--b", type=int, default=VARIANT["B"], help="batch per process")
    ap.add_argument("--lr", type=float, default=LR0)
    ap.add_argument("--nth", type=int, default=1)
    ap.add_argument("--no_sync", action="store_true")
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--fail_step", type=int, default=0)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.nth)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    if args.mode == "info":
        info = sysinfo()
        print(json.dumps(info, ensure_ascii=False, indent=1))
        save(args, info)
        return

    if args.mode in ("single", "single_speed", "refgrad"):
        model, n_params = build_model()
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        if args.mode == "refgrad":
            train = load_split(args.data, "train", N_TRAIN)
            xb, yb = train.tensors[0][:args.b], train.tensors[1][:args.b]
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            torch.save([p.grad.clone() for p in model.parameters()], os.path.join(os.path.dirname(args.out), "refgrad.pt"))
            save(args, {"B": args.b})
            return
        if args.mode == "single_speed":
            r = measure(args, model, opt, 1, "plain")
            r["n_params"] = n_params
            print(r)
            save(args, r)
            return
        test = load_split(args.data, "test")
        r = train_loop(args, model, model, opt, 1, 0, test)
        r["n_params"], r["M_mib"] = n_params, 4 * n_params / 2**20
        save(args, r)
        return

    # ---- distributed modes (torchrun)
    dist.init_process_group(backend="gloo")
    assert dist.get_world_size() == world
    model, n_params = build_model()
    for p in model.parameters():  # weights of rank 0 are sent to all
        dist.broadcast(p.data, src=0)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.mode == "equiv":
        train = load_split(args.data, "train", N_TRAIN)
        B = args.b * world
        xb, yb = train.tensors[0][:B], train.tensors[1][:B]
        ref = torch.load(os.path.join(os.path.dirname(args.out), "refgrad.pt"))
        idx = list(range(rank, B, world))
        lossf = nn.CrossEntropyLoss()
        lossf(model(xb[idx]), yb[idx]).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        ratios = [((g - p.grad).abs().max() / g.abs().max()).item() for g, p in zip(ref, model.parameters())]
        # 50 steps of training with manual synchronisation
        model2, _ = build_model()
        for p in model2.parameters():
            dist.broadcast(p.data, src=0)
        opt2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=0)
        loader = DataLoader(train, batch_size=args.b, sampler=sampler, drop_last=True)
        it = batches(loader, sampler)
        for _ in range(50):
            x, y = next(it)
            opt2.zero_grad(set_to_none=True)
            lossf(model2(x), y).backward()
            for p in model2.parameters():
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
            opt2.step()
        s = torch.tensor([sum(p.detach().double().sum().item() for p in model2.parameters())], dtype=torch.float64)
        gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(world)]
        dist.all_gather(gathered, s)
        sums = [t.item() for t in gathered]
        res = {"P": world, "B": B, "max_ratio": max(ratios), "ratios": ratios, "weight_sums": sums,
               "sum_diff": max(sums) - min(sums), "n_params": n_params}
        print(res)
        save(args, res, rank)
    elif args.mode == "manual_speed":
        r = measure(args, model, opt, world, "manual")
        if rank == 0:
            print(r)
        save(args, r, rank)
    elif args.mode == "ddp_speed":
        wrap = DDP(model)
        opt = torch.optim.SGD(wrap.parameters(), lr=args.lr, momentum=0.9)
        r = measure(args, wrap, opt, world, "ddp")
        if rank == 0:
            print(r)
        save(args, r, rank)
    elif args.mode == "ddp_train":
        test = load_split(args.data, "test")
        wrap = DDP(model)
        opt = torch.optim.SGD(wrap.parameters(), lr=args.lr, momentum=0.9)
        r = train_loop(args, model, wrap, opt, world, rank, test)
        r["n_params"] = n_params
        save(args, r, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
