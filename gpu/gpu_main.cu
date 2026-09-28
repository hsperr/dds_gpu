// GPU run: one thread per (deal, strain) solve, each with its own TT slice.
//   ./dd_gpu K DEALS [TT_LOG2] [CHECK]     random K-card endings, CHECK solves re-done on CPU
//   ./dd_gpu FILE.npy START DEALS [TT_LOG2] full Pgx deals, checked against the file
// Env: WAVE=1 wavefront kernel, THREADS, STACK_KB (default 8), TIME_LIMIT (seconds, WAVE only).
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

// Wavefront kernel: every thread runs its own solve as a stage machine, but the warp runs
// one stage at a time for all threads in it. Cheap stages go first, so threads gather at
// the expensive ENTER stage and run it together instead of diverging.
__global__ void wave_kernel(const uint64_t* hands, long jobs, DDBucket* tt, int tt_log2,
                            int* out, unsigned long long* nodes, unsigned long long* next,
                            volatile unsigned long long* done, volatile int* stop) {
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
  w.stage = DD_S_JOB;
  auto take = [&] { return (long)atomicAdd(next, 1ull); };
  const unsigned all = 0xffffffffu;
  unsigned steps = 0;
  while (__ballot_sync(all, w.stage == DD_S_EXIT) != all) {
    if ((++steps & 4095) == 0 && *stop) w.stage = DD_S_EXIT;  // host time limit
    int pick;
    if (__ballot_sync(all, w.stage == DD_S_RET)) pick = DD_S_RET;
    else if (__ballot_sync(all, w.stage == DD_S_NEXT)) pick = DD_S_NEXT;
    else if (__ballot_sync(all, w.stage == DD_S_ROOT)) pick = DD_S_ROOT;
    else if (__ballot_sync(all, w.stage == DD_S_JOB)) pick = DD_S_JOB;
    else pick = DD_S_ENTER;
    if (w.stage == pick) {
      dd_wave_step(w, take);
      if (pick != DD_S_JOB && w.stage == DD_S_JOB)
        atomicAdd((unsigned long long*)done, 1ull);  // a job just finished
    }
  }
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
    tt_log2 = argc > 4 ? atoi(argv[4]) : 10;
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
  long threads = getenv("THREADS") ? atol(getenv("THREADS")) : 16384;
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
  auto t0 = std::chrono::steady_clock::now();
  bool wave = getenv("WAVE") && atoi(getenv("WAVE"));
  if (wave) {
    CK(cudaMemset(d_next, 0, 8));
    printf("wavefront kernel\n");
    // Jobs taken so far (host-visible), printed every 10 s while the kernel runs.
    unsigned long long* h_done;
    CK(cudaHostAlloc(&h_done, 8, cudaHostAllocMapped));
    *h_done = 0;
    unsigned long long* d_done;
    CK(cudaHostGetDevicePointer((void**)&d_done, h_done, 0));
    int* h_stop;
    CK(cudaHostAlloc(&h_stop, sizeof(int), cudaHostAllocMapped));
    *h_stop = 0;
    int* d_stop;
    CK(cudaHostGetDevicePointer((void**)&d_stop, h_stop, 0));
    double limit = getenv("TIME_LIMIT") ? atof(getenv("TIME_LIMIT")) : 1e9;
    CK(cudaMemset(d_out, 0xff, n * 20 * sizeof(int)));  // -1 = not solved (time limit)
    wave_kernel<<<grid, block>>>(d_hands, jobs, d_tt, tt_log2, d_out, d_nodes, d_next, d_done,
                                 d_stop);
    auto p0 = std::chrono::steady_clock::now();
    while (cudaStreamQuery(0) == cudaErrorNotReady) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
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
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

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
