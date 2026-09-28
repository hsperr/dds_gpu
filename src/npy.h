// Load a Pgx DDS results file (.npy, int32, shape (2, n, 4)) and decode it.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

struct PgxDeals {
  std::vector<int32_t> keys, values;  // (n, 4) each
  long n = 0;
};

// Reads deals [start, start + count) of the file.
static bool pgx_load(const char* path, long start, long count, PgxDeals* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[10];
  if (fread(magic, 1, 10, f) != 10 || memcmp(magic, "\x93NUMPY", 6) != 0) return false;
  uint32_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
  long data_off = 10 + hlen;
  if (magic[6] >= 2) {  // version 2+: 4-byte header length
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2) return false;
    hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8) | (b[0] << 16) | (b[1] << 24);
    data_off = 12 + hlen;
  }
  std::vector<char> hdr(hlen + 1, 0);
  fseek(f, data_off - hlen, SEEK_SET);
  if (fread(hdr.data(), 1, hlen, f) != hlen) return false;
  const char* sh = strstr(hdr.data(), "'shape': (2, ");
  if (!sh || !strstr(hdr.data(), "<i4")) return false;
  long total = atol(sh + 13);
  if (start < 0) start += total;
  if (start < 0 || start + count > total) return false;
  out->n = count;
  out->keys.resize(count * 4);
  out->values.resize(count * 4);
  fseek(f, data_off + start * 16, SEEK_SET);
  if (fread(out->keys.data(), 16, count, f) != (size_t)count) return false;
  fseek(f, data_off + (total + start) * 16, SEEK_SET);
  if (fread(out->values.data(), 16, count, f) != (size_t)count) return false;
  fclose(f);
  return true;
}

// Four hand masks for deal i. Pgx: key[suit] holds 2-bit owners, card A,2..K at shift 2*(12-i).
static void pgx_hands(const PgxDeals& d, long i, uint64_t* hands) {
  hands[0] = hands[1] = hands[2] = hands[3] = 0;
  for (int s = 0; s < 4; s++) {
    uint32_t key = (uint32_t)d.keys[i * 4 + s];
    for (int k = 0; k < 13; k++) {
      int owner = (key >> (2 * (12 - k))) & 3;
      int rank = k == 0 ? 12 : k - 1;
      hands[owner] |= 1ull << (s * 16 + rank);
    }
  }
}

// Tricks for declarer seat in strain (C,D,H,S,NT).
static int pgx_tricks(const PgxDeals& d, long i, int seat, int strain) {
  return ((uint32_t)d.values[i * 4 + seat] >> (4 * (4 - strain))) & 15;
}
