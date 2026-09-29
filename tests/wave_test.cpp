// Host check: the stage machine (dd_wave.h) must give the same results and node counts as
// dd_solve_strain.   ./wave_test K DEALS
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <vector>
#include "dd_wave.h"
int main(int argc, char** argv) {
  int k = atoi(argv[1]);
  long n = atol(argv[2]);
  std::mt19937_64 rng(5);
  std::vector<uint64_t> hands(n * 4, 0);
  for (long d = 0; d < n; d++) {
    int cards[52];
    for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
    std::shuffle(cards, cards + 52, rng);
    for (int i = 0; i < 4 * k; i++) hands[d * 4 + i % 4] |= 1ull << cards[i];
  }
  std::vector<DDBucket> tt1((size_t)1 << 12), tt2((size_t)1 << 12);
  // Reference.
  std::vector<int> want(n * 20);
  DDCtx c{};
  c.tt = tt1.data();
  c.tt_mask = (1u << 12) - 1;
  for (long j = 0; j < n * 5; j++) dd_solve_strain(c, &hands[j / 5 * 4], j % 5, &want[j / 5 * 20 + j % 5 * 4]);
  // Stage machine.
  std::vector<int> got(n * 20);
  static DDWave w;
  w = DDWave{};
  w.c.tt = tt2.data();
  w.c.tt_mask = (1u << 12) - 1;
  w.hands = hands.data();
  w.out = got.data();
  w.jobs = n * 5;
  w.ring = n;
  w.work = 0;
  w.jpd = getenv("JPD") ? atoi(getenv("JPD")) : 5;
  w.jobs = n * w.jpd;
  w.stage = DD_S_JOB;
  long next = 0;
  long steps[DD_S_COUNT] = {0};
  while (w.stage != DD_S_EXIT) {
    steps[w.stage]++;
    dd_wave_step(w, [&] { return next++; });
  }
  long bad = 0;
  for (long i = 0; i < n * 20; i++) bad += got[i] != want[i];
  printf("k=%d: %ld wrong of %ld, nodes %llu vs %llu\n", k, bad, n * 20,
         (unsigned long long)w.c.nodes, (unsigned long long)c.nodes);
  printf("steps: job %ld root %ld enter %ld next %ld ret %ld\n", steps[0], steps[1], steps[2],
         steps[3], steps[4]);
  return bad != 0;
}
