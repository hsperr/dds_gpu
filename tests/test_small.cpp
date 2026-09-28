// Compare dd.h with plain minimax on random small deals (k cards per hand).
//   ./test_small [K_MAX] [DEALS_PER_K]
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <vector>

#include "dd.h"

// Max NS tricks from here, no pruning. Trick state as in DDCtx.
static int brute(DDCtx& c) {
  if (c.nplayed == 0 && c.left == 0) return 0;
  int seat = (c.leader + c.nplayed) & 3;
  uint64_t legal = c.hand[seat];
  if (c.nplayed) {
    uint64_t f = legal & DD_SUIT(c.trick[0] >> 4);
    if (f) legal = f;
  }
  bool ns = (seat & 1) == 0;
  int best = ns ? -1 : 99;
  for (uint64_t rest = legal; rest; rest &= rest - 1) {
    int card = dd_lsb(rest);
    c.hand[seat] ^= 1ull << card;
    c.trick[c.nplayed++] = card;
    int v;
    if (c.nplayed == 4) {
      int w = (c.leader + dd_winning(c, 4)) & 3, old = c.leader;
      c.leader = w;
      c.nplayed = 0;
      c.left--;
      int saved[4] = {c.trick[0], c.trick[1], c.trick[2], c.trick[3]};
      v = ((w & 1) == 0) + brute(c);
      for (int k = 0; k < 4; k++) c.trick[k] = saved[k];
      c.left++;
      c.nplayed = 4;
      c.leader = old;
    } else {
      v = brute(c);
    }
    c.nplayed--;
    c.hand[seat] ^= 1ull << card;
    if (ns ? v > best : v < best) best = v;
  }
  return best;
}

int main(int argc, char** argv) {
  int kmax = argc > 1 ? atoi(argv[1]) : 5;
  int per = argc > 2 ? atoi(argv[2]) : 300;
  std::mt19937_64 rng(1);
  std::vector<DDBucket> tt((size_t)1 << 12);
  DDCtx c{};
  c.tt = tt.data();
  c.tt_mask = (1u << 12) - 1;
  int bad = 0, total = 0;
  for (int k = 1; k <= kmax; k++) {
    for (int t = 0; t < per; t++) {
      int cards[52];
      for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
      std::shuffle(cards, cards + 52, rng);
      uint64_t hands[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4 * k; i++) hands[i % 4] |= 1ull << cards[i];
      for (int strain = 0; strain < 5; strain++) {
        for (int i = 0; i < 4; i++) c.hand[i] = hands[i];
        c.trump = dd_strain_suit(strain);
        dd_next_gen(c);
        for (int leader = 0; leader < 4; leader++) {
          int got = dd_ns_tricks(c, leader, 7);
          c.leader = leader;
          c.nplayed = 0;
          c.left = k;
          int want = brute(c);
          total++;
          if (got != want && bad++ < 10)
            printf("k=%d strain=%d leader=%d want %d got %d  hands %llx %llx %llx %llx\n", k,
                   strain, leader, want, got, (unsigned long long)hands[0],
                   (unsigned long long)hands[1], (unsigned long long)hands[2],
                   (unsigned long long)hands[3]);
        }
      }
    }
  }
  printf("%d wrong of %d\n", bad, total);
  return bad != 0;
}
