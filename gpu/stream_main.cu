// Streaming GPU solver: deals flow through a ring buffer in pinned host memory while one
// persistent wavefront kernel keeps solving, so the GPU never runs dry (no batch tail).
//
//   ./dd_stream                     read deals from stdin (4 x uint64 hand masks each,
//                                   bit suit*16+rank, suits S,H,D,C, ranks 2..A = 0..12),
//                                   write one record per deal as soon as it is solved (not
//                                   in input order): uint64 deal index (0-based input
//                                   position) + 20 bytes tricks[strain C,D,H,S,NT][decl N,E,S,W]
//   ./dd_stream bench SECONDS [TT_LOG2] [CHECK_EVERY]
//                                   random deals; prints deals/s every 10 s and re-solves
//                                   every CHECK_EVERY-th deal on the CPU to check it
// Env: THREADS (default 32768), STACK_KB (default 8), PER_LEAD=1 (one job per lead).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "dd_wave.h"

#define CK(x)                                                                   \
  do {                                                                          \
    cudaError_t e = (x);                                                        \
    if (e != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Shared with the host through mapped pinned memory.
struct StreamCtl {
  long long produced;  // deals queued so far
  int eof;             // no more deals will be produced
  int stop;            // quit now (bench end), unfinished deals are dropped
  unsigned long long deals_done;
  unsigned long long comp_head;  // entries written to the finished-slot list
};

// Slots hold deals; work[i % work_ring] is the slot of the i-th queued deal. A finished
// deal's slot + 1 goes to comp[] so the host can take it at once, in any order.
__global__ void stream_kernel(const uint64_t* hands, int* out, unsigned* deal_done,
                              const unsigned* work, long work_ring, unsigned* comp,
                              long comp_ring, volatile StreamCtl* ctl, DDBucket* tt, int tt_log2,
                              unsigned long long* next, unsigned long long* nodes, int jpd,
                              DDWave* waves) {
  long tid = blockIdx.x * (long)blockDim.x + threadIdx.x;
  // Search state in global memory, one contiguous block per thread. In local memory CUDA
  // interleaves it across threads in 4-byte words, so a frame read by threads at different
  // depths touches ~6x more sectors (+13% full deals/s measured).
  DDWave& w = waves[tid];
  memset(&w.c, 0, sizeof(w.c));
#ifdef DD_SHARED_TT
  w.c.tt = tt;  // one TT for all threads (cleared by the host), 2^tt_log2 buckets
  w.c.tt_mask = (1u << tt_log2) - 1;
#else
  w.c.tt = tt + ((size_t)tid << tt_log2);
  w.c.tt_mask = (1u << tt_log2) - 1;
  for (size_t i = 0; i < ((size_t)1 << tt_log2); i++)
    for (int k = 0; k < DD_TT_WAYS; k++) w.c.tt[i].tag[k] = 0;
#endif
  w.jpd = jpd;
  w.hands = hands;
  w.out = out;
  w.jobs = 1ll << 62;  // unbounded; the stream ends through ctl
  w.work = work;
  w.work_ring = work_ring;
  w.stage = DD_S_JOB;
  long long pending = -1;  // job reserved but maybe not produced yet
  auto take = [&] { return (long)pending; };
  const unsigned all = 0xffffffffu;
  unsigned steps = 0;
  while (__ballot_sync(all, w.stage == DD_S_EXIT) != all) {
    if ((++steps & 4095) == 0 && ctl->stop) w.stage = DD_S_EXIT;
    bool ready = false;
    if (w.stage == DD_S_JOB) {
      if (pending < 0) pending = (long long)atomicAdd(next, 1ull);
      int eof = ctl->eof;
      long long avail = ctl->produced;
      if (pending / jpd < avail) ready = true;
      else if (eof || ctl->stop) w.stage = DD_S_EXIT;
    }
    int pick;
    if (__ballot_sync(all, w.stage == DD_S_RET)) pick = DD_S_RET;
    else if (__ballot_sync(all, w.stage == DD_S_NEXT)) pick = DD_S_NEXT;
    else if (__ballot_sync(all, w.stage == DD_S_ROOT)) pick = DD_S_ROOT;
    else if (__ballot_sync(all, ready)) pick = DD_S_JOB;
    else if (__ballot_sync(all, w.stage == DD_S_ENTER)) pick = DD_S_ENTER;
    else {
      __nanosleep(2000);  // everyone left is waiting for input
      continue;
    }
    if (w.stage != pick || (pick == DD_S_JOB && !ready)) continue;
    long slot = w.slot;
    dd_wave_step(w, take);
    if (pick == DD_S_JOB) pending = -1;
    if (pick != DD_S_JOB && w.stage == DD_S_JOB) {  // a job finished
      __threadfence_system();
      if (atomicAdd_system(&deal_done[slot], 1u) == (unsigned)jpd - 1) {  // deal done
        __threadfence_system();
        unsigned long long pos = atomicAdd_system((unsigned long long*)&ctl->comp_head, 1ull);
        atomicExch_system(&comp[pos % comp_ring], (unsigned)slot + 1);
        atomicAdd_system((unsigned long long*)&ctl->deals_done, 1ull);
      }
    }
  }
  atomicAdd(nodes, (unsigned long long)w.c.nodes);
}

static void random_deal(std::mt19937_64& rng, uint64_t* h) {
  int cards[52];
  for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
  std::shuffle(cards, cards + 52, rng);
  h[0] = h[1] = h[2] = h[3] = 0;
  for (int i = 0; i < 52; i++) h[i % 4] |= 1ull << cards[i];
}

int main(int argc, char** argv) {
  bool bench = argc > 1 && !strcmp(argv[1], "bench");
  double seconds = bench && argc > 2 ? atof(argv[2]) : 0;
  int tt_log2 = bench && argc > 3 ? atoi(argv[3]) : 10;
  if (tt_log2 < DD_TT_MIN_LOG2) {
    fprintf(stderr, "TT_LOG2 must be >= %d\n", DD_TT_MIN_LOG2);
    return 2;
  }
  long check_every = bench && argc > 4 ? atol(argv[4]) : 50;
  long threads = getenv("THREADS") ? atol(getenv("THREADS")) : 32768;
  threads = (threads + 63) / 64 * 64;
  size_t stack_kb = getenv("STACK_KB") ? atol(getenv("STACK_KB")) : 8;
  const long ring = 1 << 16;          // deal slots
  const long work_ring = 1 << 20;     // queued-deal order; > ring, so never overtakes
  const long comp_ring = ring;        // at most `ring` deals are in flight

  CK(cudaDeviceSetLimit(cudaLimitStackSize, stack_kb * 1024));
  uint64_t* hands;
  int* out;
  unsigned* deal_done;
  unsigned *work, *comp;
  StreamCtl* ctl;
  CK(cudaHostAlloc(&hands, ring * 4 * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostAlloc(&out, ring * 20 * sizeof(int), cudaHostAllocMapped));
  CK(cudaHostAlloc(&deal_done, ring * sizeof(unsigned), cudaHostAllocMapped));
  CK(cudaHostAlloc(&work, work_ring * sizeof(unsigned), cudaHostAllocMapped));
  CK(cudaHostAlloc(&comp, comp_ring * sizeof(unsigned), cudaHostAllocMapped));
  CK(cudaHostAlloc(&ctl, sizeof(StreamCtl), cudaHostAllocMapped));
  memset(comp, 0, comp_ring * sizeof(unsigned));
  memset(deal_done, 0, ring * sizeof(unsigned));
  memset(ctl, 0, sizeof(StreamCtl));
  uint64_t* d_hands;
  int* d_out;
  unsigned *d_deal_done, *d_work, *d_comp;
  StreamCtl* d_ctl;
  CK(cudaHostGetDevicePointer((void**)&d_hands, hands, 0));
  CK(cudaHostGetDevicePointer((void**)&d_out, out, 0));
  CK(cudaHostGetDevicePointer((void**)&d_deal_done, deal_done, 0));
  CK(cudaHostGetDevicePointer((void**)&d_work, work, 0));
  CK(cudaHostGetDevicePointer((void**)&d_comp, comp, 0));
  CK(cudaHostGetDevicePointer((void**)&d_ctl, ctl, 0));
  DDBucket* d_tt;
  unsigned long long *d_next, *d_nodes;
  int jpd = getenv("PER_LEAD") && atoi(getenv("PER_LEAD")) ? 20 : 5;
#ifdef DD_SHARED_TT
  // One TT for everyone: tt_log2 here is the total bucket count (e.g. 24 = 9.7 GB).
  size_t tt_bytes = sizeof(DDBucket) * ((size_t)1 << tt_log2);
  CK(cudaMalloc(&d_tt, tt_bytes));
  CK(cudaMemset(d_tt, 0, tt_bytes));
#else
  size_t tt_bytes = (size_t)threads * sizeof(DDBucket) * ((size_t)1 << tt_log2);
  CK(cudaMalloc(&d_tt, tt_bytes));
#endif
  CK(cudaMalloc(&d_next, 8));
  CK(cudaMalloc(&d_nodes, 8));
  CK(cudaMemset(d_next, 0, 8));
  CK(cudaMemset(d_nodes, 0, 8));
  fprintf(stderr, "%ld threads, %s TT %.2f GB, %d jobs per deal, ring %ld deals\n", threads,
#ifdef DD_SHARED_TT
          "shared",
#else
          "per-thread",
#endif
          tt_bytes / 1e9, jpd, ring);

  std::atomic<long long> consumed{0};
  volatile StreamCtl* vctl = ctl;
  std::vector<long long> deal_id(ring);  // input position of the deal in each slot
  std::mutex fm;
  std::vector<unsigned> free_slots;
  for (long i = ring - 1; i >= 0; i--) free_slots.push_back((unsigned)i);

  // Producer: puts deals into free slots and queues them.
  std::atomic<bool> producing{true};
  std::thread producer([&] {
    std::mt19937_64 rng(2026);
    long long p = 0;
    while (producing) {
      unsigned slot;
      {
        std::lock_guard<std::mutex> g(fm);
        if (free_slots.empty()) slot = ~0u;
        else {
          slot = free_slots.back();
          free_slots.pop_back();
        }
      }
      if (slot == ~0u) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      uint64_t* h = hands + (size_t)slot * 4;
      if (bench) {
        random_deal(rng, h);
      } else if (fread(h, sizeof(uint64_t), 4, stdin) != 4) {
        break;
      }
      deal_id[slot] = p;
      deal_done[slot] = 0;
      work[p % work_ring] = slot;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      p++;
      vctl->produced = p;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    vctl->eof = 1;
  });

  // CPU checker for bench mode.
  std::mutex qm;
  std::vector<std::vector<uint64_t>> check_q;
  std::vector<std::vector<int>> check_want;
  std::atomic<long> checked{0}, wrong{0};
  std::atomic<bool> checking{true};
  std::vector<std::thread> checkers;
  if (bench) {
    int nc = std::max(1u, std::thread::hardware_concurrency() - 2);
    for (int t = 0; t < nc; t++)
      checkers.emplace_back([&] {
        std::vector<DDBucket> tt((size_t)1 << 12);
        DDCtx c{};
        c.tt = tt.data();
        c.tt_mask = (1u << 12) - 1;
        for (;;) {
          std::vector<uint64_t> h;
          std::vector<int> got;
          {
            std::lock_guard<std::mutex> g(qm);
            if (!check_q.empty()) {
              h = check_q.back();
              got = check_want.back();
              check_q.pop_back();
              check_want.pop_back();
            }
          }
          if (h.empty()) {
            if (!checking) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
          }
          int want[20];
          for (int st = 0; st < 5; st++) dd_solve_strain(c, h.data(), st, want + st * 4);
          checked++;
          for (int i = 0; i < 20; i++)
            if (want[i] != got[i]) {
              wrong++;
              break;
            }
        }
      });
  }

  DDWave* d_waves;
  CK(cudaMalloc(&d_waves, (size_t)threads * sizeof(DDWave)));
  int block = 64;
  stream_kernel<<<threads / block, block>>>(d_hands, d_out, d_deal_done, d_work, work_ring,
                                            d_comp, comp_ring, d_ctl, d_tt, tt_log2, d_next,
                                            d_nodes, jpd, d_waves);
  CK(cudaGetLastError());

  // Consumer: takes finished deals as they come and frees their slots.
  auto t0 = std::chrono::steady_clock::now();
  auto last = t0;
  unsigned long long last_done = 0;
  unsigned long long tail = 0;
  for (;;) {
    volatile unsigned* e = &comp[tail % comp_ring];
    if (*e) {
      unsigned slot = *e - 1;
      *e = 0;
      tail++;
      std::atomic_thread_fence(std::memory_order_seq_cst);
      int* o = out + (size_t)slot * 20;
      long long id = deal_id[slot];
      if (!bench) {
        unsigned char b[28];
        memcpy(b, &id, 8);
        for (int i = 0; i < 20; i++) b[8 + i] = (unsigned char)o[i];
        fwrite(b, 1, 28, stdout);
      } else if (id % check_every == 0) {
        std::lock_guard<std::mutex> g(qm);
        check_q.emplace_back(hands + (size_t)slot * 4, hands + (size_t)slot * 4 + 4);
        check_want.emplace_back(o, o + 20);
      }
      {
        std::lock_guard<std::mutex> g(fm);
        free_slots.push_back(slot);
      }
      consumed++;
      continue;
    }
    long long c = consumed.load();
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - t0).count();
    if (std::chrono::duration<double>(now - last).count() >= 10) {
      unsigned long long d = vctl->deals_done;
      fprintf(stderr, "  %.0fs: %llu deals solved, last 10 s: %.1f deals/s, checked %ld wrong %ld\n",
              el, d, (d - last_done) / std::chrono::duration<double>(now - last).count(),
              checked.load(), wrong.load());
      last = now;
      last_done = d;
    }
    if (bench && el > seconds) {
      vctl->stop = 1;
      break;
    }
    if (!bench && vctl->eof && c >= vctl->produced) break;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  producing = false;
  CK(cudaDeviceSynchronize());
  producer.join();
  checking = false;
  for (auto& t : checkers) t.join();
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  unsigned long long nodes;
  CK(cudaMemcpy(&nodes, d_nodes, 8, cudaMemcpyDeviceToHost));
  fprintf(stderr, "%llu deals solved in %.1f s (%.1f deals/s), CPU check: %ld wrong of %ld\n",
          (unsigned long long)vctl->deals_done, el, vctl->deals_done / el, wrong.load(),
          checked.load());
  return wrong.load() != 0;
}
