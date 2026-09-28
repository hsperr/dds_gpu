#!/usr/bin/env bash
# Correctness checks for the solver. Exit code 0 = all passed.
#   1. brute-force minimax on random 1..4-card endings, all strains and leads
#   2. full solver vs plain search (no TT, no bounds, no small-card skip), 5..7 cards
#   3. stage machine (dd_wave.h, used by the GPU wavefront kernel) vs the plain loop search
#   4. full deals vs the Pgx DDS tables (DEALS deals from data/deals_20k.npy)
set -euo pipefail
cd "$(dirname "$0")/.."
DEALS=${DEALS:-100}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
CXX="${CXX:-c++} -O3 -std=c++17 -march=native -pthread -Isrc"

$CXX -o "$OUT/test_small" tests/test_small.cpp
"$OUT/test_small" 4 300

$CXX -o "$OUT/full" tests/cmp_tt.cpp
$CXX -DDD_NO_TT -DDD_NO_SMALL -DDD_NO_QT -DDD_NO_LT -DDD_NO_QT2 -o "$OUT/plain" tests/cmp_tt.cpp
for k in 5 6 7; do
  "$OUT/full" $k 300 > "$OUT/full_$k.txt"
  "$OUT/plain" $k 300 > "$OUT/plain_$k.txt"
  diff -q "$OUT/full_$k.txt" "$OUT/plain_$k.txt" > /dev/null || { echo "k=$k: full != plain"; exit 1; }
  echo "k=$k: full == plain (1200 solves)"
done

$CXX -o "$OUT/wave_test" tests/wave_test.cpp
for k in 5 8; do "$OUT/wave_test" $k 200; done

$CXX -o "$OUT/dd_cpu" cpu/cpu_main.cpp
"$OUT/dd_cpu" data/deals_20k.npy 0 "$DEALS" 8 12
