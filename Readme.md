# Anthropic's Original Performance Take-Home

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

The original take-home was a 4-hour one that starts close to the contents of this repo, after Claude Opus 4 beat most humans at that, it was updated to a 2-hour one which started with code which achieved 18532 cycles (7.97x faster than this repo starts you). This repo is based on the newer take-home which has a few more instructions and comes with better debugging tools, but has the starter code reverted to the slowest baseline. After Claude Opus 4.5 we started using a different base for our time-limited take-homes.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

Measured in clock cycles from the simulated machine. All of these numbers are for models doing the 2 hour version which started at 18532 cycles:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **??? cycles**: Best human performance ever is substantially better than the above, but we won't say how much.

While it's no longer a good time-limited test, you can still use this test to get us excited about hiring you! If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed, especially if you get near the best solution we've seen. New model releases may change what threshold impresses us though, and no guarantees that we keep this readme updated with the latest on that.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Warning: LLMs can cheat

None of the solutions we received on the first day post-release below 1300 cycles were valid solutions. In each case, a language model modified the tests to make the problem easier.

If you use an AI agent, we recommend instructing it not to change the `tests/` folder and to use `tests/submission_tests.py` for verification.

Please run the following commands to validate your submission, and mention that you did so when submitting:
```
# This should be empty, the tests folder must be unchanged
git diff origin/main tests/
# You should pass some of these tests and use the cycle count this prints
python tests/submission_tests.py
```

An example of this kind of hack is a model noticing that `problem.py` has multicore support, implementing multicore as an optimization, noticing there's no speedup and "debugging" that `N_CORES = 1` and "fixing" the core count so they get a speedup. Multicore is disabled intentionally in this version.

---

# Solution

Final result: **1126 cycles** (131.2x faster than the 147734-cycle baseline). Only
`perf_takehome.py` was modified; `tests/` and `problem.py` are unchanged. Verify with
`python3 tests/submission_tests.py` (prints `OK` and `CYCLES: 1126`) and inspect the
per-engine breakdown with `python3 profile.py`.

## The workload

The kernel runs `batch_size=256` independent lanes through `rounds=16` of a
tree-traversal + hashing chain on a `forest_height=10` tree (2047 nodes). Each round,
for every lane:

1. **Lookup**: read the tree node at the lane's current `idx` and XOR it into the lane's `val`.
2. **Hash**: run `val` through the 6 fixed `HASH_STAGES` (a bit-mixing function).
3. **Index update**: descend the tree, `idx = 2*idx + (val&1) + 1`; at level 10 the index always wraps back to 0.

Round `r` operates on tree level `r % 11`, so levels cycle `0,1,2,...,10,0,1,...`.

## The target machine

A single-core VLIW simulator that issues, **per cycle**: 12 scalar ALU ops, 6 vector
ALU ops (`valu`, SIMD width `VLEN=8`), 2 loads, 2 stores, and 1 "flow" op
(`select`/`vselect`/`add_imm`). Scratch memory is capped at **1536 words**. There is no
vector gather — random-address reads must be issued as individual scalar `load`s.

## Optimization process (chronological)

Starting from the naive baseline (147734 cycles), the following changes were applied and
measured one at a time, each verified against `reference_kernel2` for correctness:

1. **Vectorize the batch.** Process the 256 lanes as 32 blocks of `VLEN=8`, keeping each
   block's `idx`/`val` resident in scratch for the entire kernel so state never round-trips
   to memory (only a final `vstore` of results).
2. **Global static scheduler.** Instead of emitting VLIW bundles by hand, emit a flat,
   ordered list of single-engine ops annotated with their scratch reads/writes, then run a
   dependency-aware list scheduler (`_schedule_slots`) that packs them into bundles.
   It builds a RAW/WAW/WAR dependency DAG, prioritizes ops by critical-path height, and
   fills the scarcest engines first. This interleaves independent work across blocks and
   rounds automatically.
3. **Preload shallow tree levels (0-3) into registers.** Levels 0-3 have only 15 nodes
   total. Broadcast them into scratch vectors once and select the right node with
   `vselect` (flow engine) instead of a gather. This removes 8 loads/lane for those
   rounds. Levels 4+ still use an 8-lane scalar gather.
4. **Algebraic hash fusion.** The `HASH_STAGES` linear stages `(a + c) + (a << k)`
   collapse to a single `multiply_add(a, 1+2^k, c)`. Stages 2 and 3 combine algebraically
   from 6 VALU ops down to 3. This is the single biggest VALU-op reduction.
5. **Deferred constant XOR.** Stage 5's `^ c6` is folded into the *next* round's
   preloaded `node ^ c6` vectors whenever the next level is 0-3, making it free on those
   rounds.
6. **Offload scalar-friendly work to the ALU.** The lane-wise `val ^= node` and the
   index-update parity bits are computed on the (under-utilized) scalar ALU, freeing VALU
   slots for the hash — because VALU is the bottleneck engine.
7. **Skip the level-10 wrap.** At level 10 every child index exceeds `n_nodes`, so `idx`
   always wraps to 0; replace the compare with a single `idx ^= idx`.
8. **Tuning sweeps.** `group_size` (blocks scheduled together), `round_tile` (round
   chunking), `alpha`/`sched_bias` (scheduler priority weights) were swept.
   `group_size=22, round_tile=14` won. This drove the result to 1131 → 1127 → **1126**.
9. **`blk_large` scheduler nudge.** A priority bias that finishes later-emitted blocks
   first, staggering block completion so the load-heavy final-round gathers de-bunch out of
   the drain (1127 → 1126).
10. **WAR-hazard correctness fix.** The dependency model only linked the *last* reader of a
    scratch address to a subsequent writer, leaving earlier readers unprotected — a latent
    bug that could let the scheduler hoist a writer ahead of an earlier reader whenever the
    address layout shifted. Fixed to track *all* readers since the last write. Keeps 1126
    and makes the scheduler robust to future layout changes.

## Why 1126 is the practical floor

`profile.py` reports the per-engine lower bounds (total ops ÷ slots-per-cycle):

| engine | slots/cy | total ops | min cycles |
|--------|---------:|----------:|-----------:|
| valu   | 6        | 6447      | **1075**   |
| load   | 2        | 2126      | 1063       |
| alu    | 12       | 12398     | 1034       |
| flow   | 1        | 704       | 704        |

**VALU is the bottleneck at 1075 cycles**, and all four engines sit within ~4% of each
other, so there is no slack to trade between them. At 1126 the VALU engine is busy 1124 of
1126 cycles but packs only ~5.73 of 6 slots on average; the ~50-cycle gap to the floor is
pure packing loss at the ramp/drain boundaries of the two block-groups.

Several further ideas were investigated and rejected with measurements:

- **Larger `group_size`** (more in-flight blocks → better packing) is impossible: scratch
  is at 1529/1536 and each additional block-context costs 32 words.
- **VALU→ALU offload** beyond what's done regresses — the ALU is already ~11.6/12.
- **Reducing VALU ops further** (e.g. the one genuine redundancy, a duplicated `idx-7` in
  the level-3 lookup) requires an extra live register per context, which does not fit in
  scratch.
- **Reclaiming the 512 words of persistent `idx`/`val` scratch** would require collapsing
  round-tiling to a single tile (`round_tile>=16`), which costs **+72 cycles** (1198) —
  far more than the reclaim could ever pay back — because the two-tile split isolates the
  load-heavy drain rounds.

Reaching the best-known 1076 (essentially the VALU floor) would require a fundamentally
different algorithm mapping or a non-greedy/optimal scheduler, not incremental tuning.

## The final algorithm, in brief

For each of 2 round-tiles, for each of 2 block-groups (22 + 10 blocks), the scheduler is
fed a flat op stream where every block runs, per round: **lookup → hash → index update**.

- **Lookup** — levels 0-3: `vselect` from preloaded `node ^ c6` vectors (no loads);
  levels 4+: 8 scalar gather loads, XORed into `val` on the ALU.
- **Hash** — 6 stages compressed to ~10 VALU ops via `multiply_add` fusion, the
  stages-2&3 merge, and the deferred `c6` XOR.
- **Index update** — parity bits computed on the ALU, then `idx = 2*idx + offset` as one
  VALU `multiply_add`; level 10 zeroes `idx` directly.

All state stays in scratch for the whole run; only the final `val`s are stored back. The
global list scheduler then packs this stream into 1126 VLIW bundles, keeping the VALU
engine — the binding constraint — busy nearly every cycle.
