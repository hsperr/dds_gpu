#!/usr/bin/env bash
# Solve single deals with optional feature switches: ./tests/one_deal.sh "-DDD_NO_QT" 5013 5078 ...
# Deal numbers index data/deals_20k.npy (override with FILE=...).
cd "$(dirname "$0")/.."
FILE=${FILE:-data/deals_20k.npy}
FLAGS=$1; shift
TMP=$(mktemp -d)
c++ -O3 -std=c++17 -march=native -pthread -Isrc $FLAGS -o "$TMP/dd_one" cpu/cpu_main.cpp || exit 1
for d in "$@"; do "$TMP/dd_one" "$FILE" $d 1 1 14 | tail -1 | sed "s/^/deal $d: /"; done
rm -rf "$TMP"
