#!/usr/bin/env python3
"""
Profiling / introspection helpers for the performance kernel.

Run as:  python3 profile.py

Prints:
  - Total simulated cycles
  - Per-cycle resource utilization (VALU/ALU/load/store/flow slots used)
  - Scratch space usage
  - Top bottleneck engine
"""
import os
import sys
import random
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
sys.path.insert(0, "tests")

from copy import copy
from problem import (
    Machine, Tree, Input, build_mem_image, reference_kernel2,
    N_CORES, VLEN, SCRATCH_SIZE, SLOT_LIMITS,
)
from perf_takehome import KernelBuilder

ENGINE_ORDER = ["alu", "valu", "load", "store", "flow", "debug"]


def analyze_kernel(forest_height, rounds, batch_size, seed=123, verbose=False):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    instrs = kb.instrs
    dbinfo = kb.debug_info()

    # Run the machine under submission-equivalent settings (no pause, no debug)
    # The Machine does `copy(mem)` internally, so `mem` stays pristine.
    machine = Machine(list(mem), instrs, dbinfo, n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    # Reference on a fresh copy
    ref_mem_iter = reference_kernel2(copy(mem))
    for ref_mem in ref_mem_iter:
        pass

    cycles = machine.cycle

    # Correctness: only values are checked by submission
    inp_values_p = ref_mem[6]
    got = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
    want = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    correct = (got == want)

    # Per-cycle slot utilization analysis
    n_instrs = len(instrs)
    cycle_id = 0
    per_cycle_usage = []  # per non-debug-step: counts of ops per engine
    for instr in instrs:
        if not any(name != "debug" for name in instr.keys()):
            continue
        counts = {e: 0 for e in SLOT_LIMITS}
        for name, slots in instr.items():
            if name in counts:
                counts[name] += len(slots)
        per_cycle_usage.append(counts)
        cycle_id += 1
        if cycle_id >= cycles:
            # cycles are counted by has_non_debug; per instr may have fewer
            pass

    # Aggregate stats
    total_ops_per_engine = {e: 0 for e in SLOT_LIMITS if e != "debug"}
    cycles_with_any = {e: 0 for e in SLOT_LIMITS if e != "debug"}
    dim_per_engine = {e: 0 for e in SLOT_LIMITS if e != "debug"}
    for cu in per_cycle_usage:
        for e in SLOT_LIMITS:
            if e == "debug":
                continue
            total_ops_per_engine[e] += cu[e]
            if cu[e] > 0:
                cycles_with_any[e] += 1
            dim_per_engine[e] += cu[e] if cu[e] else 0
            # unused slots for that engine this cycle
    unused_per_cycle = {e: 0 for e in SLOT_LIMITS if e != "debug"}
    for cu in per_cycle_usage:
        for e in SLOT_LIMITS:
            if e == "debug":
                continue
            unused_per_cycle[e] += SLOT_LIMITS[e] - cu[e] if cu[e] else SLOT_LIMITS[e]

    print(f"\n=== KERNEL ANALYSIS ===")
    print(f"forest_height={forest_height} rounds={rounds} batch_size={batch_size}")
    print(f"Correct: {correct}")
    print(f"Total cycles:        {cycles}")
    print(f"Speedup vs baseline: {147734 / cycles:.2f}x")
    print(f"Instructions:        {len(instrs)}")
    print()
    print(f"{'Engine':<6} {'slots/cy':>8} {'total ops':>10} {'cycles_with_op':>15} {'min_cycles_required':>20}")
    for e in SLOT_LIMITS:
        if e == "debug":
            continue
        limit = SLOT_LIMITS[e]
        total = total_ops_per_engine[e]
        if limit == 0:
            min_required = 0
        else:
            # minimum cycles this engine alone would require
            min_required = (total + limit - 1) // limit if limit else 0
        print(f"{e:<6} {limit:>8} {total:>10} {cycles_with_any[e]:>15} {min_required:>20}")

    print()
    # Bottleneck
    bot = None; val = 0
    for e in SLOT_LIMITS:
        if e == "debug": continue
        limit = SLOT_LIMITS[e]
        total = total_ops_per_engine[e]
        min_required = (total + limit - 1) // limit if limit else 0
        if min_required > val:
            bot, val = e, min_required
    print(f"Bottleneck engine (lower bound): {bot}  -> {val} cycles minimum")
    print(f"Scratch used:       {kb.scratch_ptr} / {SCRATCH_SIZE}")
    return cycles, correct


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forest-height", type=int, default=10)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    analyze_kernel(args.forest_height, args.rounds, args.batch_size, args.seed, args.verbose)