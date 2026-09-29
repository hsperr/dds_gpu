# dds_gpu

An exact double-dummy solver for bridge whose search runs on the GPU (CUDA). The same
header-only core also builds for the CPU, which is how it is tested.

## How it works

- **One solve per GPU thread.** A job is one (deal, strain). The thread solves the four
  leads of that strain one after another, sharing one transposition table (TT), and writes
  the four trick counts. Threads take the next job from a global atomic counter.
- **Null-window boolean search.** The search answers "can NS take >= N tricks?". The trick
  count comes from a binary search over N (starting from the previous lead's answer), all
  searches reusing the same TT.
- **Partition-search TT** (the DDS idea). Every search returns the set of cards whose rank
  decided the result. A TT entry stores the owners of the relevant top cards of each suit
  (relative ranks) plus the suit lengths of every hand, so one entry covers many positions.
  Small cards below the lowest relevant card of a suit are treated as equal.
- **Compact TT.** Buckets are 16-way with a 64-byte line of 32-bit tags first, so a probe
  normally reads one line. Each tag holds hash bits, the way of its entry, the tricks left,
  the solve generation and the leader.
  - Default: 16-byte exact entries. The suit lengths are not stored: the hash is a
    bijection, so bucket + tag + 39 stored hash bits identify them. Up to 29 relevant cards
    per entry; larger results are not stored.
  - `-DDD_TT_SIG`: 12-byte entries holding a 64-bit hash of the key instead of the key. No
    card limit, fewer nodes and faster, but not strictly exact: a wrong hit needs two keys
    with the same tag and the same 64-bit hash.
- **Eviction by depth.** A new entry in a full bucket replaces the tag with the least
  2 x tricks left + age, so entries that saved deep searches stay longer. The choice reads
  only the tag line; a store rewrites only the tag line and one entry.
- **DDS bounds and move ordering.** QuickTricks, QuickTricksSecondHand, LaterTricks and the
  DDS move-ordering weights are ported from DDS (`src/dd_bounds.h`).
- **Bit operations.** Cards are bits of a `uint64`; card owners, suit patterns, suit
  lengths, TT matching and equal-card skips are computed with bit operations instead of
  loops over cards.
- **Explicit-stack search.** The depth-first search is a loop over an array of small frames
  instead of recursion. On the GPU each thread's state (~4 KB) lives in one contiguous block
  of global memory.
- **Wavefront kernel.** The search is also written as a resumable stage machine
  (`src/dd_wave.h`). Each warp runs one stage at a time for all its threads: the cheap
  stages first (RET, NEXT, ROOT, JOB), then the expensive ENTER stage (bounds, TT probe,
  move generation), so threads gather at ENTER and run it together.

Two drivers:

- **Batch** (`gpu/gpu_main.cu`, `dd_gpu`): solves a fixed set of deals, for offline data
  generation. The batch ends with a tail where a few hard jobs keep a few threads busy.
- **Streaming** (`gpu/stream_main.cu`, `dd_stream`): one persistent kernel keeps running
  while deals flow through a ring of slots in pinned host memory. A deal is reported as soon
  as it is solved (out of input order), so one slow deal does not hold up the others.

## Layout

```
src/dd.h            core search: null-window search, partition-search TT, iterative search
src/dd_bounds.h     DDS-derived QuickTricks / QuickTricksSecondHand / LaterTricks, move weights
src/dd_wave.h       resumable stage machine used by the wavefront kernel
src/npy.h           loader for Pgx .npy DDS tables
gpu/gpu_main.cu     CUDA batch driver (plain and wavefront kernel), checks its answers
gpu/stream_main.cu  CUDA streaming driver (persistent wavefront kernel, ring buffer)
gpu/gpu_run.sh      upload, build and run on a remote CUDA box over ssh
cpu/cpu_main.cpp    multi-threaded CPU solver, checked against the Pgx tables
tests/              correctness tests (check.sh runs them), live DDS comparison, debug tools
bench/              node/time benchmarks, comparison against DDS via endplay
data/deals_20k.npy  20,000 deals with DDS tables (Pgx dataset, see below)
```

Cards are bit `suit * 16 + rank` of a `uint64` (suit 0..3 = S,H,D,C; rank 0..12 = 2..A).
Seats 0..3 = N,E,S,W. Strains 0..4 = C,D,H,S,NT (Pgx order).

## Build

CPU (any C++17 compiler; Apple clang works):

```
make                  # build/dd_cpu plus test and bench tools
```

GPU (`SM` = compute capability, e.g. 89 for Ada, 120 for Blackwell consumer):

```
make gpu SM=89                          # build/dd_gpu and build/dd_stream, exact TT
make gpu SM=89 NVFLAGS=-DDD_TT_SIG      # 12-byte hash TT entries
# or by hand:
nvcc -O3 -std=c++17 -arch=sm_89 -Isrc -o dd_gpu gpu/gpu_main.cu
```

Other build flags: `-DDD_SHARED_TT` (one lock-free TT shared by all threads, keyed by trump
suit; slower on the GPU), `-DDD_PROFILE` (per-stage cycle and active-lane counters in the
wavefront kernel).

## Run

GPU batch:

```
./dd_gpu K DEALS [TT_LOG2] [CHECK]         # random K-card endings; first CHECK deals re-solved on the CPU
./dd_gpu FILE.npy START DEALS [TT_LOG2]    # full deals from a Pgx file, checked against its tables
WAVE=1 TIME_LIMIT=120 ./dd_gpu data/deals_20k.npy 0 20000
```

- `WAVE=1` selects the wavefront kernel (the fast one).
- `THREADS` GPU threads (default 32768). Each has its own TT of `2^TT_LOG2` buckets
  (default 10, minimum 9; 320 KB per thread with exact entries, 256 KB with `-DDD_TT_SIG`).
- `TIME_LIMIT` seconds (wavefront only): stop cleanly and check only finished answers.
- `STACK_KB` per-thread stack limit (default 8).

GPU streaming:

```
./dd_stream < deals.bin > results.bin
./dd_stream bench SECONDS [TT_LOG2] [CHECK_EVERY]
```

- Input: 32 bytes per deal, 4 x `uint64` hand masks (N,E,S,W), card encoding as above.
- Output: 28 bytes per deal, written when the deal is solved: `uint64` input index
  (0-based) + 20 bytes `tricks[strain C,D,H,S,NT][declarer N,E,S,W]`.
- `bench` streams random full deals, prints deals/s every 10 s and re-solves every
  CHECK_EVERY-th deal (default 50) on the CPU to check it.
- Env: `THREADS`, `STACK_KB`, `PER_LEAD=1` (one job per lead instead of per strain).

Remote box: `gpu/gpu_run.sh "ssh -p PORT root@HOST"` uploads the sources and data, builds
for the box's GPU and runs the ending and full-deal checks.

CPU:

```
./build/dd_cpu FILE.npy START COUNT THREADS TT_LOG2
./build/dd_cpu data/deals_20k.npy 0 100 8 12
```

## Tests

```
tests/check.sh              # or: make check
DEALS=10 tests/check.sh     # fewer full deals, faster
CXX="c++ -DDD_TT_SIG" tests/check.sh
```

`check.sh` runs:

1. `test_small.cpp`: plain minimax vs the solver on random 1..4-card endings.
2. `cmp_tt.cpp`: full solver vs plain search (no TT, no bounds, no small-card skip) on
   5..7-card endings.
3. `wave_test.cpp`: the stage machine must give the same answers and node counts as the
   loop search.
4. `cpu_main.cpp` on DEALS full deals from `data/deals_20k.npy` vs the stored DDS tables.

It exits non-zero on any mismatch.

Other tools in `tests/`: `brute_one.cpp` (minimax one position), `verify_one.cpp` (build
with `-DDD_VERIFY`: re-checks every TT hit with a TT-free search), `one_deal.sh` (single
deals with feature switches), `pbn.cpp` (print a deal as PBN), `vs_dds.py` (live comparison
against DDS through the `endplay` package). `bench/` measures nodes and time per solve, also
for DDS.

## Correctness

- Full deals vs the Pgx DDS tables: 0 wrong on every GPU run (latest: 176,672 answers exact
  TT, 182,060 with `-DDD_TT_SIG`) and on the CPU.
- Random endings, GPU vs the CPU build: 0 wrong. Live vs DDS on fresh 5-, 8- and 10-card
  endings: 0 wrong of 12,000 answers.
- Streaming mode, live CPU re-solve of a sample of the streamed deals: 0 wrong.

## Performance

Full 13-card deals, all 20 results per deal, wavefront kernel, 32,768 threads, 120 s from
`data/deals_20k.npy` (`TIME_LIMIT=120`), RTX 4070 Ti SUPER 16 GB:

| Version | Full deals/s |
|---|---|
| First wavefront version (787cc62) | 37.9* |
| TT store shifts only the tag line | 41.5* |
| Bit operations instead of loops over cards | 47.8 |
| 16-byte TT entries | 52.8 |
| + eviction by depth | 59.9 |
| + thread state in contiguous global memory (**default**) | **~65** |
| Same with `-DDD_TT_SIG` | **67.7** |

\* Measured on a second RTX 4070 Ti SUPER box and scaled by 0.80 (the same code ran 59.5
there and 47.8 on the box used for the other rows).

Full deals/s over 120 s count finished (deal, strain) jobs / 5 and favour the easier deals;
over long streaming runs the rate is lower. For reference, the CPU build of an earlier
version ran 8.7 full deals/s on 32 cores, and on one CPU core this solver needs about
2-2.5x the cycles of DDS.

## Known limits

- **Long tail.** A single hard job can run for minutes on one GPU thread. A fixed batch ends
  with few threads busy; the streaming mode avoids that.
- **Memory-bound.** The kernel mostly waits on memory (TT probes and per-thread state). The
  TT per thread is small (~256-320 KB), which costs nodes compared to a CPU with a large
  cached TT.
- **Exact entries** skip results with more than 29 relevant cards; `-DDD_TT_SIG` has no such
  limit.
- CUDA reserves `cudaLimitStackSize` for all resident threads, so keep `STACK_KB` small.

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
