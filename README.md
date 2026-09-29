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
  so a probe normally touches one 64-byte line. Tags are kept newest first and each tag
  names the way of its entry, so a store shifts only the tag line, not the whole bucket
  (+9% full deals/s, +22% on 8-card endings on an RTX 4070 Ti SUPER; same node counts).
- **16-byte entries.** An entry keeps 35 bits of the suit-length hash instead of the
  lengths (the hash is a bijection, so bucket + tag + those bits identify the lengths
  exactly), no generation or leader (the tag has them), and the owner patterns as counts plus
  concatenated owner bits (up to 24 relevant cards; a store with more is skipped). A bucket
  is 320 bytes. The shared-TT build keeps the 32-byte entry (576-byte buckets).
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

## Two modes

- **Batch** (`gpu/gpu_main.cu`, `dd_gpu`): solves a fixed set of deals, for offline data
  generation. About 30-37 full deals/s over a fixed batch (28-30 steady), but the batch ends
  with a tail where a few hard jobs keep a few threads busy.
- **Streaming** (`gpu/stream_main.cu`, `dd_stream`): for continuous use (e.g. a bridge
  platform). One persistent wavefront kernel keeps running; deals flow through a ring of
  slots in pinned, mapped host memory. A deal is reported as soon as all its jobs are done,
  through a finished-slot list, so results come out of input order and one slow deal does
  not hold up the others. No batch tail.

## Layout

```
src/dd.h          core search: null-window search, partition-search TT, iterative search
src/dd_bounds.h   DDS-derived QuickTricks / QuickTricksSecondHand / LaterTricks, move weights
src/dd_wave.h     resumable stage machine used by the wavefront kernel
src/npy.h         loader for Pgx .npy DDS tables
gpu/gpu_main.cu   CUDA batch driver (plain kernel and wavefront kernel), checks its answers
gpu/stream_main.cu  CUDA streaming driver (persistent wavefront kernel, ring buffer)
gpu/gpu_run.sh    upload, build and run on a remote CUDA box over ssh
cpu/cpu_main.cpp  multi-threaded CPU solver, checked against the Pgx tables
tests/            correctness tests (check.sh runs them), live DDS comparison, debug tools
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
make gpu SM=120       # build/dd_gpu and build/dd_stream
# or by hand:
nvcc -O3 -std=c++17 -arch=sm_120 -Isrc -o dd_gpu gpu/gpu_main.cu
nvcc -O3 -std=c++17 -arch=sm_120 -Isrc -o dd_stream gpu/stream_main.cu
```

Add `-DDD_SHARED_TT` for the shared-TT experiment (see below). The default build uses one
TT per thread.

## Run

GPU batch:

```
./dd_gpu K DEALS [TT_LOG2] [CHECK]         # random K-card endings; first CHECK deals re-solved on the CPU
./dd_gpu FILE.npy START DEALS [TT_LOG2]    # full deals from a Pgx file, checked against its tables
```

Environment:

- `WAVE=1` selects the wavefront kernel (the fast one). Without it the plain kernel runs.
- `THREADS` number of GPU threads (default 32768). Each thread has its own TT of
  `2^TT_LOG2` buckets (320 bytes each, so the default TT_LOG2=10 is 320 KB per thread; TT_LOG2 >= 9).
- `STACK_KB` per-thread stack limit (default 8). CUDA reserves this for every resident
  thread, so keep it small; the iterative search needs about 4 KB.
- `TIME_LIMIT` seconds (wavefront kernel only). After the limit, threads stop cleanly and
  only finished answers are checked. The last line reports finished jobs / elapsed time.
- `CHEAP_LOOP=1` (wavefront only) runs the cheap stages back to back until each thread
  reaches ENTER, instead of one stage per round. Measured: no difference.
- Build with `-DDD_PROFILE` to print, per stage, the share of warp cycles, warp steps and
  active lanes out of 32, and where ENTER returned (TT hit, quick tricks, full move list...).

Example: `WAVE=1 TIME_LIMIT=180 ./dd_gpu data/deals_20k.npy 0 20000`

`gpu/gpu_run.sh "ssh -p PORT root@HOST"` copies the sources and data to a remote CUDA box,
builds with the box's compute capability, and runs the ending and full-deal checks.

GPU streaming:

```
./dd_stream < deals.bin > results.bin
./dd_stream bench SECONDS [TT_LOG2] [CHECK_EVERY]
```

- Input: 32 bytes per deal, 4 x `uint64` hand masks (N,E,S,W), card encoding as above.
- Output: 28 bytes per deal, written when the deal is solved (not in input order): `uint64`
  input index (0-based) + 20 bytes `tricks[strain C,D,H,S,NT][declarer N,E,S,W]`.
- `bench` streams random full deals for SECONDS, prints deals/s every 10 s, and re-solves
  every CHECK_EVERY-th deal (default 50) on CPU threads to check it.
- `TT_LOG2` default 10 (minimum 9). Env: `THREADS` (default 32768), `STACK_KB` (default 8), `PER_LEAD=1` (20 jobs per deal,
  one per lead, instead of 5 per-strain jobs where the 4 leads share the TT).
- With `-DDD_SHARED_TT`, TT_LOG2 is the total bucket count of the one shared TT
  (e.g. 24 = 9.7 GB).

CPU:

```
./build/dd_cpu FILE.npy START COUNT THREADS TT_LOG2
./build/dd_cpu data/deals_20k.npy 0 100 8 12
```

`GUESS=exact|off1` starts each lead's search from the stored answer (or one off), an
experiment on how much a good first guess saves.

**Shared TT (`-DDD_SHARED_TT`, experiment).** The TT key is the trump suit instead of a
per-solve generation, so an entry describes a position class valid for any deal with that
trump suit and one TT can serve every deal and thread. Entries are lock-free: a checksum in
the top 32 bits of `meta` turns a half-written entry into a miss, and new entries replace
one way in place. Built with it, `dd_cpu` makes all threads share one TT.

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
   as the loop search. `JPD=20 build/wave_test 8 200` tests one job per lead (answers must
   match; node counts differ because the leads no longer share a TT).
4. `cpu_main.cpp` on DEALS full deals from `data/deals_20k.npy` vs the stored DDS tables.

It exits non-zero on any mismatch and prints `0 wrong of N` lines when everything agrees.

Debug tools in `tests/`: `brute_one.cpp` (minimax one position given as four hex masks),
`verify_one.cpp` (build with `-DDD_VERIFY`: re-checks every TT hit with a TT-free search
and prints bad hits), `one_deal.sh` (solve single deals from the data file with feature
switches, e.g. `tests/one_deal.sh "-DDD_NO_QT" 5013`), `pbn.cpp` (print a deal as PBN
with its stored table).

Live comparison against DDS: `python tests/vs_dds.py K DEALS [SEED]` deals fresh random
K-card endings, solves them with `build/dd_pbn` (reads PBN lines, prints tricks for the side
on lead per strain and leader; override the path with `DD_PBN=...`) and with DDS through the
`endplay` package's libdds, all strains and leads, and prints `0 wrong of N` on agreement.

Bench: `bench/bench_k.cpp` (nodes and time per strain solve on random K-card endings;
`PBN=1` prints the deals instead), `bench/dds_nodes.py` and `bench/dds_nodes_pbn.py`
(DDS nodes and time for the same deals through the `endplay` Python package).

## Correctness

- Brute-force minimax on small endings and full vs plain search agree (see Tests).
- Full deals vs the Pgx DDS tables, CPU: 0 wrong of 10,000 answers on 500 deals.
- GPU wavefront kernel on full deals vs the Pgx tables: 0 wrong of 244,259 answers.
- GPU on 8-card endings vs the CPU build of the same code: 0 wrong on 6,000 checks.
- Live vs DDS (`tests/vs_dds.py`, fresh random deals): 0 wrong of 12,000 answers on 5-, 8-
  and 10-card endings (200 deals each, all strains and leads).
- Streaming mode, live CPU re-solve of a sample of the streamed deals: 0 wrong in all runs
  (396, 345 and 276 full deals checked, see Results).
- Shared TT (`-DDD_SHARED_TT`): 0 wrong in all tests; 12 CPU threads sharing one TT: 0
  wrong of 1,920 answers on full deals.

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

Full-deal GPU run (batch mode): 16,384 threads, 576 KB TT per thread (TT_LOG2=10), 56,835
strain-jobs in 310 s.

Streaming mode, same box: RTX 5070 Ti, 16,384 threads, 5 minutes of random full deals
(`dd_stream bench`), a sample of the deals re-solved live on the CPU:

| Setup | Full deals/s | Live CPU check |
|---|---|---|
| Per-thread TT 576 KB, per-strain jobs (default) | 25.6 (about 21-22 in the last minutes) | 0 wrong of 396 |
| Shared TT 9.7 GB, per-strain jobs | 22.3 | 0 wrong of 345 |
| Shared TT 9.7 GB, per-lead jobs (`PER_LEAD=1`) | 19.1 | 0 wrong of 276 |

The streaming rate drifts down over the first minutes because hard deals gradually occupy
more threads; ~21-25 deals/s is the long-run figure, about 2.5-3x the 32-core CPU (8.7
deals/s).

Tuning, batch mode, full deals, 3 minutes each (`TIME_LIMIT=180`, finished jobs / time;
a 3-minute window still contains the easy start, so these are higher than long-run rates):

| Threads | TT per thread | Full deals/s | Check vs Pgx tables |
|---|---|---|---|
| 4,096 | 2.3 MB (TT_LOG2=12) | 18.5 | 0 wrong of 69,859 |
| 8,192 | 1.2 MB (11) | 30.9 | 0 wrong of 118,684 |
| 16,384 | 576 KB (10) | 44.2 | 0 wrong of 174,696 |
| **32,768 (new default)** | 288 KB (9) | **46.8** | 0 wrong of 195,767 |

More threads in flight beat a bigger TT per thread: the kernel waits on memory, and more
warps hide more of that latency. With the new defaults, 8-card endings run at 12,454 deals/s
(0 wrong of 6,000 checked against the CPU).

RTX 4070 Ti SUPER, defaults, 120 s of full deals:

| Version | Full deals/s | 8-card endings/s | Check |
|---|---|---|---|
| Before | 47.2 | 9,069 | 0 wrong of 136,248 |
| TT store shifts only the tag line | 51.6 | 10,502 | 0 wrong of 147,694 |
| + owners, suit codes, TT probe, move skips with bit operations (no loops over cards) | **59.5** | **13,299** | 0 wrong of 168,668 |

Node counts are the same in all three; only the work per node changed.

Profile of the wavefront kernel on full deals (`-DDD_PROFILE`; Nsight Compute could not be
used because the vast.ai container blocks GPU performance counters, `ERR_NVGPUCTRPERM`):

| Stage | Share of warp cycles | Active lanes of 32 |
|---|---|---|
| ENTER (bounds, TT probe, move list) | 65% | 32.0 |
| NEXT (play next move) | 15% | 10.0 |
| RET (undo, cut test, TT store) | 20% | 3.8 |

ENTER exits: full move list 73%, TT hit 22%, quick tricks 1.3%, trivial 1.7%, others < 1.5%.
The wavefront works (ENTER always runs with all 32 lanes). The cheap stages run with few
lanes, but running them back to back (`CHEAP_LOOP=1`) gave the same speed (44.2 vs 43.7
deals/s), so their cost is memory latency (frames in local memory, TT stores), not
scheduling. Per-thread search state is ~3.4 KB, ~110 MB at 32,768 threads, more than L2.

Single-thread CPU vs DDS (endplay's libdds, SolveBoard per leader): this solver needs
about 2-2.5x more cycles than DDS on full deals.

Other findings:

- Sharing the TT across deals on one CPU thread gives almost no gain (same node count, -7%
  cycles): positions rarely repeat across deals.
- Starting from a perfect guess of the trick count (as a neural net might give) saves only
  ~16% cycles (off-by-one guess: ~7%), because the TT makes the extra null-window searches
  cheap.
- One job per lead without TT sharing costs 1.8x total work on the CPU.
- The shared GPU TT is slower, likely because of random access into one 9.7 GB table,
  in-place replacement (+20% nodes on CPU) and checksum overhead.

## Known limits and next steps

- **Long tail.** A single hard job can run 10+ minutes on one GPU thread. A fixed batch
  ends with a tail where few threads are busy; the streaming mode avoids that.
- **TT sharing.** Solving each lead as its own job (no TT shared between the four leads)
  costs 1.8x. A shared lock-free TT across threads works (`-DDD_SHARED_TT`) but is slower
  on the GPU (see Results).
- **Memory-bound.** The kernel mostly waits on memory (see the profile). The next real gain
  would come from less memory traffic per node: a smaller per-thread search state (fewer
  and smaller frames, keeping the hot parts in registers) or a cheaper TT layout.
- Ideas: splitting ENTER into sub-stages (only ~27% of ENTER calls leave early, so the gain
  is limited); hardest-first scheduling.
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
