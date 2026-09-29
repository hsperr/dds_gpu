"""Compare dd.h with DDS (endplay's libdds) on fresh random deals, all strains and leads.
   python tests/vs_dds.py K DEALS [SEED]   (K cards per hand, 13 = full deals)
   Needs `make` (build/dd_pbn; override with DD_PBN=path) and `pip install endplay`."""
import os, random, subprocess, sys
import endplay._dds as d
from endplay.dds import calc_dd_table
from endplay.types import Deal

k, n = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
calc_dd_table(Deal.from_pbn("N:963.KQT.K865.J92 KT.J742.743.AQ84 A7542.A6.AJ.K765 QJ8.9853.QT92.T3"))
R = "23456789TJQKA"
deals = []
for _ in range(n):
    cards = [(s, r) for s in range(4) for r in range(13)]
    rng.shuffle(cards)
    hands = []
    for p in range(4):
        mine = cards[p * k:(p + 1) * k]
        hands.append(".".join("".join(R[r] for r in sorted((r for s2, r in mine if s2 == s), reverse=True)) for s in range(4)))
    deals.append("N:" + " ".join(hands))

dd_pbn = os.environ.get("DD_PBN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "dd_pbn"))
ours = subprocess.run([dd_pbn], input="\n".join(deals) + "\n", capture_output=True, text=True).stdout.split("\n")
bad = total = 0
for pbn, line in zip(deals, ours):
    got = list(map(int, line.split()))
    for strain in range(5):              # ours: C,D,H,S,NT ; dds: S,H,D,C,NT
        dstrain = 4 if strain == 4 else 3 - strain
        for leader in range(4):
            deal = d.dealPBN(); deal.trump, deal.first = dstrain, leader
            deal.remainCards = pbn.encode()
            fut = d.futureTricks()
            d.SolveBoardPBN(deal, -1, 1, 1, fut, 0)
            want = fut.score[0]           # tricks for the side on lead
            total += 1
            if got[strain * 4 + leader] != want:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH {pbn} strain {strain} leader {leader}: dds {want} ours {got[strain * 4 + leader]}")
print(f"k={k}: {bad} wrong of {total} (fresh random deals vs DDS)")
