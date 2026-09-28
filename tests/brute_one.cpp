#include <stdio.h>
#include <stdlib.h>
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
  DDCtx c{};
  for (int i = 0; i < 4; i++) c.hand[i] = strtoull(argv[1 + i], 0, 16);
  c.trump = dd_strain_suit(atoi(argv[5]));
  c.leader = atoi(argv[6]);
  c.left = dd_popc(c.hand[0]);
  printf("brute NS tricks: %d\n", brute(c));
}
