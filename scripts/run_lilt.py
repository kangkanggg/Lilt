#!/usr/bin/env python3
"""Portable per-process Lilt launcher. Does not modify host CUDA settings."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def real_driver(env: dict[str, str]) -> Path:
    configured = env.get("LILT_REAL_LIBCUDA") or env.get("TGS_REAL_LIBCUDA")
    if configured:
        path = Path(configured)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("LILT_REAL_LIBCUDA must name an existing absolute driver path")
        return path.resolve()
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    result = subprocess.run([ldconfig, "-p"], text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        if line.strip().startswith("libcuda.so.1 ") and "=>" in line:
            path = Path(line.split("=>", 1)[1].strip()).resolve()
            if path.is_file():
                return path
    raise ValueError("Real NVIDIA driver not found; set LILT_REAL_LIBCUDA")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def prepare_environment(args, base: dict[str, str]) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.session):
        raise ValueError("session must be 1-48 letters, digits, underscores or hyphens")
    if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", args.gpu):
        raise ValueError("select exactly one GPU index or GPU UUID")
    if not 1 <= args.be_count <= 64:
        raise ValueError("be-count must be in [1, 64]")
    if not args.be_count <= args.window <= 64:
        raise ValueError("window must be in [be-count, 64]")
    if args.time_budget_ns < args.be_count:
        raise ValueError("time budget must allow at least 1 ns per BE process")
    if args.graphlet_nodes > 4096:
        raise ValueError("graphlet-nodes must not exceed 4096")
    if base.get("LD_PRELOAD"):
        raise ValueError("unset LD_PRELOAD before launching to avoid stacking incompatible proxies")
    # Each command gets a clean policy environment, never an inherited ablation.
    driver = real_driver(base)
    env = {k: v for k, v in base.items()
           if not k.startswith(("TGS_", "LILT_"))}
    proxy = ROOT / "hijack" / f"{args.role}-lib" / "libcuda.so.1"
    if not proxy.is_file():
        raise ValueError("proxy missing; run bash scripts/build_lilt.sh first")
    for role in ("hp", "be"):
        if driver == (ROOT / "hijack" / f"{role}-lib/libcuda.so.1").resolve():
            raise ValueError("real driver path points to a Lilt proxy")
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "LILT_REAL_LIBCUDA": str(driver),
        "LD_PRELOAD": str(proxy),
        "LILT_POLICY": "lilt",
        "LILT_EVENT_SHM_NAME": f"/lilt-{os.getuid()}-{args.session}-{args.gpu}",
        "LILT_EVENT_COMPLETION_MODE": "deferred",
        "LILT_EVENT_BLOCKING_QUIET_NS": str(args.quiet_ns),
        "LILT_EVENT_BE_POLICY": "time-sum",
        "LILT_EVENT_BE_COUNT": str(args.be_count),
        "LILT_EVENT_BE_WINDOW": str(args.window),
        "LILT_EVENT_BE_TIME_BUDGET_NS": str(args.time_budget_ns),
        "LILT_EVENT_BE_UNKNOWN_KERNEL_NS": str(args.unknown_kernel_ns),
        "LILT_EVENT_BE_GRAPH_MODE": args.graph_mode,
        "LILT_EVENT_BE_GRAPHLET_MAX_NODES": str(args.graphlet_nodes),
    })
    # Unlike LD_PRELOAD alone, the local path also serves dlopen("libcuda.so.1").
    # Some frameworks (e.g. bundled JAX CUDA) need their original search order.
    if not args.keep_library_path:
        env["LD_LIBRARY_PATH"] = str(proxy.parent) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("hp", "be"))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--session", required=True, help="same unique value for one HP and all its BEs")
    parser.add_argument("--be-count", type=positive, default=1)
    parser.add_argument("--window", type=positive, default=64, help="group-wide maximum in-flight kernels")
    parser.add_argument("--time-budget-ns", type=positive, default=80000, help="group-wide predicted duration budget")
    parser.add_argument("--unknown-kernel-ns", type=positive, default=20000)
    parser.add_argument("--quiet-ns", type=positive, default=20000)
    parser.add_argument("--graph-mode", choices=("whole", "graphlet"), default="whole")
    parser.add_argument("--graphlet-nodes", type=positive, default=8)
    parser.add_argument("--keep-library-path", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and print configuration; do not run")
    argv = sys.argv[1:]
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    command = argv[separator + 1:]
    if not command or command[0].startswith("-"):
        parser.error("provide the application command after --")
    try:
        env = prepare_environment(args, os.environ.copy())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    if args.dry_run:
        for key, value in sorted(env.items()):
            if key.startswith("LILT_") or key in ("LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
                print(f"{key}={value}")
        print(f"command={command!r}")
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
