"""Nodes and time dds needs per (strain, leader) solve, via endplay's libdds.

  python bench/dds_nodes.py data/deals_20k.npy START COUNT   (needs build/pbn: make pbn)
"""
import os, subprocess, sys, time
import endplay._dds as d

from endplay.dds import calc_dd_table
from endplay.types import Deal
calc_dd_table(Deal.from_pbn("N:963.KQT.K865.J92 KT.J742.743.AQ84 A7542.A6.AJ.K765 QJ8.9853.QT92.T3"))
PBN = os.environ.get("PBN_BIN", os.path.join(os.path.dirname(__file__), "..", "build", "pbn"))
path, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
nodes, secs, n = 0, 0.0, 0
for i in range(start, start + count):
    lines = subprocess.run([PBN, path, str(i)], capture_output=True, text=True).stdout.split("\n")
    pbn = lines[0]
    for strain in range(5):              # dds: S,H,D,C,NT = 0..4
        for first in range(4):
            deal = d.dealPBN()
            deal.trump, deal.first = strain, first
            deal.remainCards = pbn.encode()
            fut = d.futureTricks()
            t = time.perf_counter()
            d.SolveBoardPBN(deal, -1, 1, 1, fut, 0)
            secs += time.perf_counter() - t
            nodes += fut.nodes
            n += 1
print(f"dds: {nodes / n:.0f} nodes/solve, {1000 * secs / n:.2f} ms/solve, {count / secs:.1f} deals/s (1 thread)")
