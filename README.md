# dds_gpu

An exact double-dummy solver for bridge whose search runs on the GPU (CUDA). The same
header-only core also builds for the CPU, which is how it is tested.

## What it is

- **One solve per GPU thread.** A job is one (deal, strain). The thread solves the four
  leads of that strain one after another, sharing one transposition table (TT), and writes
  the four trick counts. Threads take the next job from a global atomic counter.
- **Null-window boolean search.** The search answers "can NS take >= N tricks?", not a
  minimax over trick counts. The trick count comes from a binary search over N (starting
  from the previous lead's answer), all searches reusing the same TT.
- **Partition-search TT** (the DDS idea). Every search returns the set of cards whose rank
  decided the result. A TT entry stores the owner pattern of the relevant top cards of each
  suit (relative ranks, not absolute) plus the exact suit lengths of every hand, so one entry
  covers many positions. Small cards below the lowest relevant card of a suit are treated
  as equal, so only one of them is tried. Buckets are 16-way with 32-bit tags stored first,
  so a probe normally touches one 64-byte line.
- **DDS bounds and move ordering.** QuickTricks, QuickTricksSecondHand, LaterTricks and the
  DDS move-ordering weights are ported from DDS (`src/dd_bounds.h`), kept close to the
  original so the two can be compared.
- **Explicit-stack search.** The depth-first search is a loop over an array of small frames
  instead of recursion, so a GPU thread needs only a few KB of stack.
- **Wavefront kernel.** The search is also written as a resumable stage machine
  (`src/dd_wave.h`). In the wavefront kernel each warp runs one stage at a time for all its
  threads: the cheap stages first (RET: a child returned; NEXT: play the next move; ROOT
  and JOB: next binary-search step or next job), then the expensive ENTER stage (bounds,
  TT probe, move generation). Threads therefore gather at ENTER and run it together
  instead of diverging through different code paths. This is the fast kernel.

## Layout

```
src/dd.h          core search: null-window search, partition-search TT, iterative search
src/dd_bounds.h   DDS-derived QuickTricks / QuickTricksSecondHand / LaterTricks, move weights
src/dd_wave.h     resumable stage machine used by the wavefront kernel
src/npy.h         loader for Pgx .npy DDS tables
gpu/gpu_main.cu   CUDA driver (plain kernel and wavefront kernel), checks its answers
gpu/gpu_run.sh    upload, build and run on a remote CUDA box over ssh
cpu/cpu_main.cpp  multi-threaded CPU solver, checked against the Pgx tables
tests/            correctness tests (check.sh runs them) and debug tools
bench/            node/time benchmarks, comparison against DDS via endplay
data/deals_20k.npy  20,000 deals with DDS tables (Pgx dataset, see below)
```

Cards are bit `suit * 16 + rank` of a `uint64` (suit 0..3 = S,H,D,C; rank 0..12 = 2..A).
Seats 0..3 = N,E,S,W. Strains 0..4 = C,D,H,S,NT (Pgx order).

## Build

CPU (any C++17 compiler; Apple clang works):

```
make                  # build/dd_cpu plus test and bench tools
# or by hand:
c++ -O3 -std=c++17 -march=native -pthread -Isrc -o dd_cpu cpu/cpu_main.cpp
```

GPU (`sm_XX` = your GPU's compute capability, e.g. 89 for Ada, 120 for Blackwell consumer):

```
make gpu SM=120
# or by hand:
nvcc -O3 -std=c++17 -arch=sm_120 -Isrc -o dd_gpu gpu/gpu_main.cu
```

## Run

GPU:

```
./dd_gpu K DEALS [TT_LOG2] [CHECK]         # random K-card endings; first CHECK deals re-solved on the CPU
./dd_gpu FILE.npy START DEALS [TT_LOG2]    # full deals from a Pgx file, checked against its tables
```

Environment:

- `WAVE=1` selects the wavefront kernel (the fast one). Without it the plain kernel runs.
- `THREADS` number of GPU threads (default 16384). Each thread has its own TT of
  `2^TT_LOG2` buckets (576 bytes each, so TT_LOG2=10 is 576 KB per thread).
- `STACK_KB` per-thread stack limit (default 8). CUDA reserves this for every resident
  thread, so keep it small; the iterative search needs about 4 KB.
- `TIME_LIMIT` seconds (wavefront kernel only). After the limit, threads stop cleanly and
  only finished answers are checked.

Example: `WAVE=1 THREADS=16384 ./dd_gpu data/deals_20k.npy 0 6553 10`

`gpu/gpu_run.sh "ssh -p PORT root@HOST"` copies the sources and data to a remote CUDA box,
builds with the box's compute capability, and runs the ending and full-deal checks.

CPU:

```
./build/dd_cpu FILE.npy START COUNT THREADS TT_LOG2
./build/dd_cpu data/deals_20k.npy 0 100 8 12
```

`GUESS=exact|off1` starts each lead's search from the stored answer (or one off), an
experiment on how much a good first guess saves.

## Tests

```
tests/check.sh              # or: make check
DEALS=10 tests/check.sh     # fewer full deals, faster
```

`check.sh` builds and runs:

1. `test_small.cpp`: plain minimax (no pruning) vs the solver on random 1..4-card endings,
   all strains and leads.
2. `cmp_tt.cpp`: full solver vs plain search (no TT, no bounds, no small-card skip, built
   with `-DDD_NO_TT -DDD_NO_SMALL -DDD_NO_QT -DDD_NO_LT -DDD_NO_QT2`) on 5..7-card endings.
3. `wave_test.cpp`: the stage machine must give the same answers and the same node counts
   as the loop search.
4. `cpu_main.cpp` on DEALS full deals from `data/deals_20k.npy` vs the stored DDS tables.

It exits non-zero on any mismatch and prints `0 wrong of N` lines when everything agrees.

Debug tools in `tests/`: `brute_one.cpp` (minimax one position given as four hex masks),
`verify_one.cpp` (build with `-DDD_VERIFY`: re-checks every TT hit with a TT-free search
and prints bad hits), `one_deal.sh` (solve single deals from the data file with feature
switches, e.g. `tests/one_deal.sh "-DDD_NO_QT" 5013`), `pbn.cpp` (print a deal as PBN
with its stored table).

Bench: `bench/bench_k.cpp` (nodes and time per strain solve on random K-card endings;
`PBN=1` prints the deals instead), `bench/dds_nodes.py` and `bench/dds_nodes_pbn.py`
(DDS nodes and time for the same deals through the `endplay` Python package).

## Correctness

- Brute-force minimax on small endings and full vs plain search agree (see Tests).
- Full deals vs the Pgx DDS tables, CPU: 0 wrong of 10,000 answers on 500 deals.
- GPU wavefront kernel on full deals vs the Pgx tables: 0 wrong of 244,259 answers.
- GPU on 8-card endings vs the CPU build of the same code: 0 wrong on 6,000 checks.

One bug found this way and fixed: move generation tries only the top card of a run of
touching cards of the player to move. When the lowest relevant card of a suit is part of
such a run, the whole run must stay relevant. Otherwise the TT merges positions where a
card of another hand sits inside the run and breaks it (`dd_finish` in `src/dd.h`).

## Results

RTX 5070 Ti 16 GB vs a 32-core CPU on the same vast.ai box, same code. The CUDA driver
compiled and ran there; these are the numbers from those runs.

| Workload | GPU wavefront | GPU plain kernel | CPU 32 threads |
|---|---|---|---|
| 8-card endings | 11,075 deals/s | 6,027 deals/s | ~4,000-4,500 deals/s |
| Full 13-card deals (all 20 results) | ~30-37 deals/s (steady ~28-30, 36.7 avg over 310 s) | not measured cleanly | 8.7 deals/s |

Full-deal GPU run: 16,384 threads, 576 KB TT per thread (TT_LOG2=10), 56,835 strain-jobs
in 310 s.

Single-thread CPU vs DDS (endplay's libdds, SolveBoard per leader): this solver needs
about 2-2.5x more cycles than DDS on full deals.

## Known limits and next steps

- **Long tail.** A single hard job can run 10+ minutes on one GPU thread. A fixed batch
  ends with a tail where few threads are busy; feeding jobs as a stream avoids that.
- **TT sharing.** Solving each lead as its own job (no TT shared between the four leads)
  costs 1.8x.
- Ideas: a TT shared by the 4 leads of a deal (across threads) with lockless checksummed
  entries; splitting ENTER into sub-stages to cut divergence further; hardest-first
  scheduling; a stream mode.
- CUDA reserves `cudaLimitStackSize` for all resident threads, so a large `STACK_KB`
  wastes memory. The iterative search needs about 4 KB.

## Data and credits

`data/deals_20k.npy` holds 20,000 deals with their DDS result tables, taken from the Pgx
DDS dataset (https://huggingface.co/datasets/sotetsuk/dds_dataset, Apache-2.0, by the Pgx
authors). Format: int32 array of shape (2, n, 4); keys hold 2-bit card owners per suit,
values hold 4-bit trick counts per declarer and strain (see `src/npy.h`).

The bounds and move-ordering code in `src/dd_bounds.h` and the partition-search
("win ranks") idea are from DDS by Bo Haglund and Soren Hein
(https://github.com/dds-bridge/dds, Apache-2.0). See `NOTICE`.

## License

Apache License 2.0, see `LICENSE` and `NOTICE`.
