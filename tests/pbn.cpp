// Print deal i of a Pgx file as PBN (N first) and its stored table.
#include "npy.h"
int main(int argc, char** argv) {
  PgxDeals d;
  long i = atol(argv[2]);
  if (!pgx_load(argv[1], i, 1, &d)) return 1;
  uint64_t h[4];
  pgx_hands(d, 0, h);
  const char* R = "23456789TJQKA";
  printf("N:");
  for (int p = 0; p < 4; p++) {
    for (int s = 0; s < 4; s++) {
      for (int r = 12; r >= 0; r--)
        if (h[p] >> (s * 16 + r) & 1) putchar(R[r]);
      if (s < 3) putchar('.');
    }
    putchar(p < 3 ? ' ' : '\n');
  }
  for (int seat = 0; seat < 4; seat++)
    for (int st = 0; st < 5; st++) printf("%d%c", pgx_tricks(d, 0, seat, st), seat == 3 && st == 4 ? '\n' : ',');
}
