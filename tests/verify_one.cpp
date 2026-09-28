#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "dd.h"
int main(int argc, char** argv) {
  uint64_t h[4];
  for (int i = 0; i < 4; i++) h[i] = strtoull(argv[1 + i], 0, 16);
  std::vector<DDBucket> tt((size_t)1 << 14);
  DDCtx c{};
  c.tt = tt.data();
  c.tt_mask = (1u << 14) - 1;
  int out[4];
  dd_solve_strain(c, h, atoi(argv[5]), out);
  printf("tricks N E S W: %d %d %d %d\n", out[0], out[1], out[2], out[3]);
}
