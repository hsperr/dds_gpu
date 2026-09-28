// Nodes and time per strain solve (4 leads) for random k-card deals.
#include <chrono>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "dd.h"
int main(int argc, char** argv) {
  int k = atoi(argv[1]), n = argc > 2 ? atoi(argv[2]) : 20;
  std::mt19937_64 rng(7);
  std::vector<DDBucket> tt((size_t)1 << 16);
  DDCtx c{};
  c.tt = tt.data();
  c.tt_mask = (1u << 16) - 1;
  auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < n; t++) {
    int cards[52];
    for (int i = 0; i < 52; i++) cards[i] = (i / 13) * 16 + i % 13;
    std::shuffle(cards, cards + 52, rng);
    uint64_t h[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4 * k; i++) h[i % 4] |= 1ull << cards[i];
    if (getenv("PBN")) {
      const char* R = "23456789TJQKA";
      printf("N:");
      for (int p = 0; p < 4; p++) {
        for (int s = 0; s < 4; s++) {
          for (int r = 12; r >= 0; r--) if (h[p] >> (s * 16 + r) & 1) putchar(R[r]);
          if (s < 3) putchar('.');
        }
        putchar(p < 3 ? ' ' : '\n');
      }
      continue;
    }
    int out[4];
    dd_solve_strain(c, h, 4, out);
  }
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("probe %llu hit %.1f%%  qt cut %llu  nocut %llu  cut@i:", (unsigned long long)c.st_probe, 100.0*c.st_hit/c.st_probe, (unsigned long long)c.st_qt, (unsigned long long)c.st_nocut);
  for (int i = 0; i < 6; i++) printf(" %llu", (unsigned long long)c.st_cut[i]);
  printf("\n");
  printf("trick nodes per leader-solve %.0f\n", c.st_tricks / (4.0 * n));
  printf("lead nodes: all %llu  cut@i:", (unsigned long long)c.st_lead_all);
  for (int i = 0; i < 8; i++) printf(" %llu", (unsigned long long)c.st_lead_cut[i]);
  printf("\n");
  printf("k=%d  %.0f nodes/solve  %.2f ms/solve\n", k, (double)c.nodes / n, 1000 * s / n);
}
