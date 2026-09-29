// Read PBN deals ("N:hand hand hand hand", k cards each) from stdin; print, per deal, the
// tricks for each (strain C,D,H,S,NT) x (leader N,E,S,W) won by the side on lead.
#include <stdio.h>
#include <string.h>
#include <vector>
#include "dd.h"
int main() {
  std::vector<DDBucket> tt((size_t)1 << 14);
  DDCtx c{};
  c.tt = tt.data();
  c.tt_mask = (1u << 14) - 1;
  const char* R = "23456789TJQKA";
  char line[256];
  while (fgets(line, sizeof line, stdin)) {
    if (strncmp(line, "N:", 2)) continue;
    uint64_t h[4] = {0, 0, 0, 0};
    int p = 0, s = 0;
    for (char* q = line + 2; *q && *q != '\n'; q++) {
      if (*q == ' ') { p++; s = 0; continue; }
      if (*q == '.') { s++; continue; }
      h[p] |= 1ull << (s * 16 + (int)(strchr(R, *q) - R));
    }
    for (int strain = 0; strain < 5; strain++) {
      for (int i = 0; i < 4; i++) c.hand[i] = h[i];
      c.trump = dd_strain_suit(strain);
      dd_next_gen(c);
      for (int leader = 0; leader < 4; leader++) {
        int ns = dd_ns_tricks(c, leader, 7);
        int k = dd_popc(h[0]);
        printf("%d ", (leader & 1) == 0 ? ns : k - ns);
      }
    }
    printf("\n");
  }
}
