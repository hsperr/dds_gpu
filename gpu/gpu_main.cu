// GPU run: one thread per (deal, strain) solve, each with its own TT slice.
//   ./dd_gpu K DEALS [TT_LOG2] [CHECK]     random K-card endings, CHECK solves re-done on CPU
//   ./dd_gpu FILE.npy START DEALS [TT_LOG2] full Pgx deals, checked against the file
// Env: WAVE=1 wavefront kernel, THREADS (default 32768), STACK_KB (default 8), TIME_LIMIT
// (seconds, WAVE only), CHEAP_LOOP=1 (run cheap stages back to back; no gain measured).
// Build with -DDD_PROFILE for per-stage cycle / active-lane counters (WAVE only).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "dd_wave.h"
#include "npy.h"

#define CK(x)                                                                   \
  do {                                                                          \
    cudaError_t e = (x);                                                        \
    if (e != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Each thread takes the next unsolved job from a shared counter, so threads that drew
// short solves keep working instead of idling behind the long ones.
__global__ void solve_kernel(const uint64_t* hands, long jobs, DDBucket* tt, int tt_log2,
                             int* out, unsigned long long* nodes, unsigned long long* next) {
  long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  if (tid >= stride || tid >= jobs) return;
  DDCtx c;
  memset(&c, 0, sizeof(c));
  c.tt = tt + ((size_t)tid << tt_log2);
  c.tt_mask = (1u << tt_log2) - 1;
  for (size_t i = 0; i < ((size_t)1 << tt_log2); i++)
    for (int w = 0; w < DD_TT_WAYS; w++) c.tt[i].tag[w] = 0;
  for (long job = tid; job < jobs; job = (long)atomicAdd(next, 1ull)) {
    long deal = job / 5;
    int strain = job % 5;
    dd_solve_strain(c, hands + deal * 4, strain, out + deal * 20 + strain * 4);
  }
  atomicAdd(nodes, (unsigned long long)c.nodes);
}

#ifdef DD_PROFILE
// Per stage: [0..5] warp cycles, [8..13] warp iterations, [16..21] active lanes summed;
// [24..30] how often dd_enter left at each exit point (see enter_exit in dd.h).
__device__ unsigned long long g_prof[32];
#endif

// Wavefront kernel: every thread runs its own solve as a stage machine, but the warp runs
// one stage at a time for all threads in it. Cheap stages go first, so threads gather at
// the expensive ENTER stage and run it together instead of diverging.
__global__ void wave_kernel(const uint64_t* hands, long jobs, DDBucket* tt, int tt_log2,
                            int* out, unsigned long long* nodes, unsigned long long* next,
                            volatile unsigned long long* done, volatile int* stop,
                            int cheap_loop) {
  long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  DDWave w;
  memset(&w.c, 0, sizeof(w.c));
  w.c.tt = tt + ((size_t)tid << tt_log2);
  w.c.tt_mask = (1u << tt_log2) - 1;
  for (size_t i = 0; i < ((size_t)1 << tt_log2); i++)
    for (int k = 0; k < DD_TT_WAYS; k++) w.c.tt[i].tag[k] = 0;
  w.hands = hands;
  w.out = out;
  w.jobs = jobs;
  w.ring = jobs / 5;
  w.work = 0;
  w.jpd = 5;
  w.stage = DD_S_JOB;
  auto take = [&] { return (long)atomicAdd(next, 1ull); };
  const unsigned all = 0xffffffffu;
  unsigned steps = 0;
#ifdef DD_PROFILE
  unsigned long long prof_cyc[DD_S_COUNT] = {0}, prof_it[DD_S_COUNT] = {0},
                     prof_lanes[DD_S_COUNT] = {0}, prof_exit[7] = {0};
#endif
  while (__ballot_sync(all, w.stage == DD_S_EXIT) != all) {
    if ((++steps & 4095) == 0 && *stop) w.stage = DD_S_EXIT;  // host time limit
    if (cheap_loop) {
      // Cheap stages run back to back until the thread reaches ENTER (or runs out of
      // jobs); then the warp runs ENTER for everyone at once.
      bool cheap = w.stage == DD_S_RET || w.stage == DD_S_NEXT || w.stage == DD_S_ROOT ||
                   w.stage == DD_S_JOB;
      if (__ballot_sync(all, cheap)) {
#ifdef DD_PROFILE
        long long t_start = clock64();
#endif
        while (w.stage == DD_S_RET || w.stage == DD_S_NEXT || w.stage == DD_S_ROOT ||
               w.stage == DD_S_JOB) {
          int before = w.stage;
          dd_wave_step(w, take);
          if (before != DD_S_JOB && w.stage == DD_S_JOB)
            atomicAdd((unsigned long long*)done, 1ull);  // a job just finished
        }
#ifdef DD_PROFILE
        unsigned act = __ballot_sync(all, cheap);
        if ((threadIdx.x & 31) == 0) {
          prof_cyc[DD_S_NEXT] += clock64() - t_start;
          prof_it[DD_S_NEXT]++;
          prof_lanes[DD_S_NEXT] += __popc(act);
        }
#endif
        continue;
      }
    }
    int pick;
    if (__ballot_sync(all, w.stage == DD_S_RET)) pick = DD_S_RET;
    else if (__ballot_sync(all, w.stage == DD_S_NEXT)) pick = DD_S_NEXT;
    else if (__ballot_sync(all, w.stage == DD_S_ROOT)) pick = DD_S_ROOT;
    else if (__ballot_sync(all, w.stage == DD_S_JOB)) pick = DD_S_JOB;
    else pick = DD_S_ENTER;
#ifdef DD_PROFILE
    long long t_start = clock64();
    bool ran = w.stage == pick;
#endif
    if (w.stage == pick) {
      dd_wave_step(w, take);
      if (pick != DD_S_JOB && w.stage == DD_S_JOB)
        atomicAdd((unsigned long long*)done, 1ull);  // a job just finished
    }
#ifdef DD_PROFILE
    unsigned act = __ballot_sync(all, ran);
    if ((threadIdx.x & 31) == 0) {
      prof_cyc[pick] += clock64() - t_start;
      prof_it[pick]++;
      prof_lanes[pick] += __popc(act);
    }
    if (ran && pick == DD_S_ENTER) prof_exit[w.c.enter_exit]++;
#endif
  }
#ifdef DD_PROFILE
  for (int i = 0; i < DD_S_COUNT; i++) {
    atomicAdd(&g_prof[i], prof_cyc[i]);
    atomicAdd(&g_prof[8 + i], prof_it[i]);
    atomicAdd(&g_prof[16 + i], prof_lanes[i]);
  }
  for (int i = 0; i < 7; i++) atomicAdd(&g_prof[24 + i], prof_exit[i]);
#endif
  atomicAdd(nodes, (unsigned long long)w.c.nodes);
}

static void random_deals(int k, long n, std::vector<uint64_t>& hands) {
  std::mt19937_64 rng(12345);
  hands.assign(n * 4, 0);
  for (long d = 0; d < n; d++) {
    int cards[52];
    for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
    std::shuffle(cards, cards + 52, rng);
    for (int i = 0; i < 4 * k; i++) hands[d * 4 + i % 4] |= 1ull << cards[i];
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s K DEALS [TT_LOG2] [CHECK] | %s FILE START DEALS [TT_LOG2]\n",
            argv[0], argv[0]);
    return 2;
  }
  bool file_mode = strstr(argv[1], ".npy") != nullptr;
  std::vector<uint64_t> hands;
  std::vector<int> want;
  long n;
  int tt_log2, check = 0;
  if (file_mode) {
    long start = atol(argv[2]);
    n = atol(argv[3]);
    tt_log2 = argc > 4 ? atoi(argv[4]) : 9;
    PgxDeals d;
    if (!pgx_load(argv[1], start, n, &d)) { fprintf(stderr, "cannot load\n"); return 1; }
    hands.resize(n * 4);
    want.resize(n * 20);
    for (long i = 0; i < n; i++) {
      pgx_hands(d, i, &hands[i * 4]);
      for (int st = 0; st < 5; st++)
        for (int seat = 0; seat < 4; seat++) want[i * 20 + st * 4 + seat] = pgx_tricks(d, i, seat, st);
    }
  } else {
    int k = atoi(argv[1]);
    n = atol(argv[2]);
    tt_log2 = argc > 3 ? atoi(argv[3]) : 10;
    check = argc > 4 ? atoi(argv[4]) : 200;
    random_deals(k, n, hands);
  }

  long jobs = n * 5;
  long threads = getenv("THREADS") ? atol(getenv("THREADS")) : 32768;
  if (threads > jobs) threads = jobs;
  threads = (threads + 63) / 64 * 64;  // whole blocks; spare threads find no job
  size_t tt_bytes = (size_t)threads * sizeof(DDBucket) * ((size_t)1 << tt_log2);
  printf("%ld threads, ", threads);
  printf("%ld deals, %ld solves, TT %.0f KB/solve, %.2f GB total\n", n, jobs,
         sizeof(DDBucket) * ((size_t)1 << tt_log2) / 1024.0, tt_bytes / 1e9);

  size_t stack_kb = getenv("STACK_KB") ? atol(getenv("STACK_KB")) : 8;
  CK(cudaDeviceSetLimit(cudaLimitStackSize, stack_kb * 1024));
  uint64_t* d_hands;
  DDBucket* d_tt;
  int* d_out;
  unsigned long long* d_nodes;
  unsigned long long* d_next;
  CK(cudaMalloc(&d_hands, hands.size() * 8));
  CK(cudaMalloc(&d_tt, tt_bytes));
  CK(cudaMalloc(&d_out, n * 20 * sizeof(int)));
  CK(cudaMalloc(&d_nodes, 8));
  CK(cudaMemcpy(d_hands, hands.data(), hands.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_nodes, 0, 8));
  CK(cudaMalloc(&d_next, 8));
  unsigned long long first_free = (unsigned long long)threads;  // jobs 0..threads-1 start
  CK(cudaMemcpy(d_next, &first_free, 8, cudaMemcpyHostToDevice));

  int block = 64;
  long grid = (threads + block - 1) / block;
  unsigned long long* h_done_ptr = nullptr;
  auto t0 = std::chrono::steady_clock::now();
  bool wave = getenv("WAVE") && atoi(getenv("WAVE"));
  if (wave) {
    CK(cudaMemset(d_next, 0, 8));
    printf("wavefront kernel\n");
    // Jobs taken so far (host-visible), printed every 10 s while the kernel runs.
    unsigned long long* h_done;
    CK(cudaHostAlloc(&h_done, 8, cudaHostAllocMapped));
    *h_done = 0;
    h_done_ptr = h_done;
    unsigned long long* d_done;
    CK(cudaHostGetDevicePointer((void**)&d_done, h_done, 0));
    int* h_stop;
    CK(cudaHostAlloc(&h_stop, sizeof(int), cudaHostAllocMapped));
    *h_stop = 0;
    int* d_stop;
    CK(cudaHostGetDevicePointer((void**)&d_stop, h_stop, 0));
    double limit = getenv("TIME_LIMIT") ? atof(getenv("TIME_LIMIT")) : 1e9;
    CK(cudaMemset(d_out, 0xff, n * 20 * sizeof(int)));  // -1 = not solved (time limit)
    int cheap_loop = getenv("CHEAP_LOOP") && atoi(getenv("CHEAP_LOOP"));
    printf("cheap stages %s\n", cheap_loop ? "run back to back" : "one step per round");
    wave_kernel<<<grid, block>>>(d_hands, jobs, d_tt, tt_log2, d_out, d_nodes, d_next, d_done,
                                 d_stop, cheap_loop);
    auto p0 = std::chrono::steady_clock::now();
    double next_print = 10;
    while (cudaStreamQuery(0) == cudaErrorNotReady) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
      if (el < next_print) continue;
      next_print += 10;
      unsigned long long fin = *(volatile unsigned long long*)h_done;
      printf("  %.0fs: %llu of %ld jobs done (%.1f deals/s)\n", el, fin, jobs, fin / 5.0 / el);
      if (el > limit && !*h_stop) {
        *(volatile int*)h_stop = 1;
        printf("  time limit: stopping, unfinished jobs are skipped in the check\n");
      }
      fflush(stdout);
    }
  } else {
    solve_kernel<<<grid, block>>>(d_hands, jobs, d_tt, tt_log2, d_out, d_nodes, d_next);
  }
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  if (wave) {
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("finished jobs: %llu in %.1f s = %.1f full-job deals/s\n", *h_done_ptr, el,
           *h_done_ptr / 5.0 / el);
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

#ifdef DD_PROFILE
  if (wave) {
    unsigned long long pr[32];
    CK(cudaMemcpyFromSymbol(pr, g_prof, sizeof(pr)));
    const char* names[DD_S_COUNT] = {"JOB", "ROOT", "ENTER", "NEXT", "RET", "EXIT"};
    double tot = 0;
    for (int i = 0; i < DD_S_COUNT; i++) tot += pr[i];
    printf("stage   cycles%%  warp-steps  active lanes/32\n");
    for (int i = 0; i < DD_S_COUNT; i++)
      if (pr[8 + i])
        printf("%-6s  %6.1f  %10llu  %5.1f\n", names[i], 100.0 * pr[i] / tot, pr[8 + i],
               (double)pr[16 + i] / pr[8 + i]);
    const char* ex[7] = {"trivial", "last trick", "TT hit", "quick tricks", "later tricks",
                         "2nd-hand QT", "full (moves)"};
    double et = 0;
    for (int i = 0; i < 7; i++) et += pr[24 + i];
    printf("ENTER exits:");
    for (int i = 0; i < 7; i++) printf("  %s %.1f%%", ex[i], 100.0 * pr[24 + i] / et);
    printf("\n");
  }
#endif
  std::vector<int> got(n * 20);
  unsigned long long nodes;
  CK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&nodes, d_nodes, 8, cudaMemcpyDeviceToHost));
  printf("GPU: %.3fs, %.0f deals/s (%.0f strain-solves/s), %.0f nodes/solve\n", sec, n / sec,
         jobs / sec, (double)nodes / jobs);

  long bad = 0, checked = 0;
  if (file_mode) {
    for (long i = 0; i < n * 20; i++) {
      if (got[i] < 0) continue;
      bad += got[i] != want[i];
      checked++;
    }
  } else {
    // Same code on the CPU for the first `check` deals, all cores.
    long m = check < n ? check : n;
    std::vector<int> cpu(m * 20);
    std::atomic<long> next{0};
    auto c0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    int nt = std::thread::hardware_concurrency();
    for (int t = 0; t < nt; t++)
      pool.emplace_back([&] {
        std::vector<DDBucket> tt((size_t)1 << 12);
        DDCtx c{};
        c.tt = tt.data();
        c.tt_mask = (1u << 12) - 1;
        for (long job; (job = next++) < m * 5;)
          dd_solve_strain(c, &hands[(job / 5) * 4], job % 5, &cpu[(job / 5) * 20 + (job % 5) * 4]);
      });
    for (auto& th : pool) th.join();
    double cs = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    printf("CPU (%d threads, same code): %.0f deals/s\n", nt, m / cs);
    for (long i = 0; i < m * 20; i++) bad += got[i] != cpu[i];
    checked = m * 20;
  }
  printf("%ld wrong of %ld checked\n", bad, checked);
  return bad != 0;
}
