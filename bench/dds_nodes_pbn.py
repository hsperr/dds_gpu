"""dds nodes per (strain, leader) solve for PBN deals on stdin (NT only by default)."""
import sys, time
import endplay._dds as d
from endplay.dds import calc_dd_table
from endplay.types import Deal
calc_dd_table(Deal.from_pbn("N:963.KQT.K865.J92 KT.J742.743.AQ84 A7542.A6.AJ.K765 QJ8.9853.QT92.T3"))
strains = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4").split(",")]
nodes = n = 0
secs = 0.0
for pbn in sys.stdin.read().split("\n"):
    if not pbn.strip(): continue
    for strain in strains:
        for first in range(4):
            deal = d.dealPBN(); deal.trump, deal.first = strain, first
            deal.remainCards = pbn.encode()
            fut = d.futureTricks(); t = time.perf_counter()
            d.SolveBoardPBN(deal, -1, 1, 1, fut, 0)
            secs += time.perf_counter() - t; nodes += fut.nodes; n += 1
print(f"dds: {nodes / n:.0f} nodes per leader-solve, {1000 * secs / n:.2f} ms")
