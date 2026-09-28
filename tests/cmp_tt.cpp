// Print solve results for random k-card deals, one line per (deal, strain), to diff builds
// with and without a feature (e.g. -DDD_NO_TT):  ./cmp_tt K N
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <random>
#include <vector>

#include "dd.h"

int main(int argc, char** argv) {
  int k = atoi(argv[1]), n = atoi(argv[2]);
  std::mt19937_64 rng(99);
  std::vector<DDBucket> tt((size_t)1 << 14);
  DDCtx c{};
  c.tt = tt.data();
  c.tt_mask = (1u << 14) - 1;
  for (int d = 0; d < n; d++) {
    int cards[52];
    for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
    std::shuffle(cards, cards + 52, rng);
    uint64_t h[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4 * k; i++) h[i % 4] |= 1ull << cards[i];
    for (int st = 0; st < 4; st++) {
      int out[4];
      dd_solve_strain(c, h, st, out);
      printf("%d %d %d %d %d %d  %llx %llx %llx %llx\n", d, st, out[0], out[1], out[2], out[3],
             (unsigned long long)h[0], (unsigned long long)h[1], (unsigned long long)h[2],
             (unsigned long long)h[3]);
    }
  }
}
