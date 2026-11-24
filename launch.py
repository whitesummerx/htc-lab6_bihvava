#!/usr/bin/env python3
"""Minimal torchrun replacement (single node): starts P copies of a script with RANK, WORLD_SIZE,
LOCAL_RANK, MASTER_ADDR, MASTER_PORT; if any process exits non-zero, stops the rest and
restarts ALL processes (up to --max-restarts times), like `torchrun --max-restarts=N`.
Used only when torchrun's own rendezvous does not work (Windows builds without libuv).
"""
import argparse, os, socket, subprocess, sys, tempfile, time


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, required=True)
    ap.add_argument("--max-restarts", type=int, default=0)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)  # script and its arguments (after --)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    port = a.port or free_port()
    code = 1
    logs = []
    for attempt in range(a.max_restarts + 1):
        procs, files = [], []
        for r in range(a.nproc):
            env = dict(os.environ, RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(a.nproc),
                       LOCAL_WORLD_SIZE=str(a.nproc), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                       USE_LIBUV="0", TORCHELASTIC_RESTART_COUNT=str(attempt))
            if a.nproc > 1:
                env.setdefault("OMP_NUM_THREADS", "1")
            fh = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
            files.append(fh)
            procs.append(subprocess.Popen([sys.executable, *cmd], env=env, stdout=fh, stderr=subprocess.STDOUT))
        failed = False
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                failed = True
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                for p in procs:
                    p.wait()
                break
            if all(c == 0 for c in codes):
                break
            time.sleep(0.1)
        for r, fh in enumerate(files):
            fh.seek(0)
            logs.append(f"===== attempt {attempt} rank {r} =====\n" + fh.read())
            fh.close()
        if not failed:
            code = 0
            break
        print(f"[launcher] attempt {attempt} failed (exit codes {[p.returncode for p in procs]})", flush=True)
        time.sleep(1.0)
    print("\n".join(logs))
    sys.exit(code)


if __name__ == "__main__":
    main()
