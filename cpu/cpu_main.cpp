// CPU check: solve Pgx deals with dd.h and compare with the stored DDS tables.
//   ./dd_cpu data/deals_20k.npy START COUNT [THREADS] [TT_LOG2]
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "dd.h"
#include "npy.h"
#include <string.h>

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s FILE START COUNT [THREADS] [TT_LOG2]\n", argv[0]);
    return 2;
  }
  long start = atol(argv[2]), count = atol(argv[3]);
  int threads = argc > 4 ? atoi(argv[4]) : (int)std::thread::hardware_concurrency();
  int tt_log2 = argc > 5 ? atoi(argv[5]) : 18;
  if (tt_log2 < DD_TT_MIN_LOG2) {
    fprintf(stderr, "TT_LOG2 must be >= %d\n", DD_TT_MIN_LOG2);
    return 2;
  }
  // GUESS=exact: start from the true answer; GUESS=off1: true answer +-1 (alternating).
  const char* guess_mode = getenv("GUESS") ? getenv("GUESS") : "";
  PgxDeals d;
  if (!pgx_load(argv[1], start, count, &d)) {
    fprintf(stderr, "cannot load %s\n", argv[1]);
    return 1;
  }

  std::vector<int> got(count * 20);
  std::atomic<long> next{0};
  std::atomic<uint64_t> nodes{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
#ifdef DD_SHARED_TT
  // One TT for all threads (lock-free entries), sized like all per-thread TTs together.
  std::vector<DDBucket> shared_tt((size_t)threads << tt_log2);
#endif
  for (int t = 0; t < threads; t++) {
    pool.emplace_back([&] {
      DDCtx c{};
#ifdef DD_SHARED_TT
      c.tt = shared_tt.data();
      c.tt_mask = (uint32_t)(shared_tt.size() - 1);
#else
      std::vector<DDBucket> tt((size_t)1 << tt_log2);
      c.tt = tt.data();
      c.tt_mask = (1u << tt_log2) - 1;
#endif
      uint64_t hands[4];
      for (long job; (job = next++) < count * 5;) {
        long i = job / 5;
        int strain = job % 5;
        pgx_hands(d, i, hands);
        if (*guess_mode) {
          int ns4[4];
          for (int decl = 0; decl < 4; decl++) {
            int t = pgx_tricks(d, i, decl, strain);
            int ns = (decl & 1) == 0 ? t : 13 - t;
            if (!strcmp(guess_mode, "off1")) ns += ((i + decl + strain) & 1) ? 1 : -1;
            ns4[decl] = ns;
          }
          dd_solve_strain_guess(c, hands, strain, ns4, &got[i * 20 + strain * 4]);
        } else {
          dd_solve_strain(c, hands, strain, &got[i * 20 + strain * 4]);
        }
      }
      nodes += c.nodes;
    });
  }
  for (auto& th : pool) th.join();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  long bad = 0;
  for (long i = 0; i < count; i++)
    for (int strain = 0; strain < 5; strain++)
      for (int seat = 0; seat < 4; seat++) {
        int want = pgx_tricks(d, i, seat, strain), have = got[i * 20 + strain * 4 + seat];
        if (want != have && bad++ < 10)
          printf("deal %ld strain %d seat %d: want %d got %d\n", start + i, strain, seat, want,
                 have);
      }
  printf("%ld deals, %d threads: %.2fs, %.1f deals/s, %.0f nodes/deal, %ld wrong of %ld\n",
         count, threads, sec, count / sec, (double)nodes / count, bad, count * 20);
  return bad != 0;
}
