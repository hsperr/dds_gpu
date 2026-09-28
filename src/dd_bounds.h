// Cheap trick bounds, ported from DDS (quick_tricks.cpp, later_tricks.cpp; Apache 2.0,
// (C) Bo Haglund & Soren Hein). Kept close to the original so the two can be compared.
// DDS "win_ranks" is our `wr` card mask; MAX = NS (even seats), MIN = EW.
#pragma once

#define DD_P(h) (((h) + 2) & 3)
#define DD_L(h) (((h) + 1) & 3)
#define DD_R(h) (((h) + 3) & 3)
#define DD_MAX(h) (((h) & 1) == 0)

// Top three cards of every suit and all suit lengths.
struct DDTop {
  int8_t len[4][4];  // [hand][suit]
  int8_t win_h[4], win_c[4], sec_h[4], sec_c[4], thr_h[4], thr_c[4];  // hand -1 = no card
};

DD_FN void dd_top_init(DDTop& t, const uint64_t* hand) {
  for (int h = 0; h < 4; h++)
    for (int s = 0; s < 4; s++) t.len[h][s] = dd_popc(hand[h] & DD_SUIT(s));
  for (int s = 0; s < 4; s++) {
    uint64_t a = (hand[0] | hand[1] | hand[2] | hand[3]) & DD_SUIT(s);
    int8_t* hs[3] = {&t.win_h[s], &t.sec_h[s], &t.thr_h[s]};
    int8_t* cs[3] = {&t.win_c[s], &t.sec_c[s], &t.thr_c[s]};
    for (int i = 0; i < 3; i++) {
      if (!a) { *hs[i] = -1; *cs[i] = 0; continue; }
      int card = dd_msb(a);
      a ^= 1ull << card;
      int h = 0;
      while (!(hand[h] & (1ull << card))) h++;
      *hs[i] = (int8_t)h;
      *cs[i] = (int8_t)card;
    }
  }
}

#define DD_BIT(card) (1ull << (card))

DD_FN int dd_next_qt_suit(int suit, int trump) {
  if (trump >= 0 && suit == trump) return trump == 0 ? 1 : 0;
  suit++;
  if (trump >= 0 && suit == trump) suit++;
  return suit;
}

// res: 0 continue with same suit, 1 cutoff, 2 next suit.
DD_FN int dd_qt_lead_trump(const DDTop& t, int hand, int cutoff, int countLho, int countRho,
                           int lhoT, int rhoT, int countOwn, int countPart, int suit, int qt,
                           uint64_t& wr, int& res) {
  res = 1;
  if ((countLho != 0 || lhoT == 0) && (countRho != 0 || rhoT == 0)) {
    wr |= DD_BIT(t.win_c[suit]);
    qt++;
    if (qt >= cutoff) return qt;
    if (countLho <= 1 && countRho <= 1 && countPart <= 1 && lhoT == 0 && rhoT == 0) {
      qt += countOwn - 1;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  }
  if (t.sec_h[suit] == hand) {
    if (lhoT == 0 && rhoT == 0) {
      wr |= DD_BIT(t.sec_c[suit]);
      qt++;
      if (qt >= cutoff) return qt;
      if (countLho <= 2 && countRho <= 2 && countPart <= 2) {
        qt += countOwn - 2;
        if (qt >= cutoff) return qt;
        res = 2;
        return qt;
      }
    }
  } else if (t.sec_h[suit] == DD_P(hand) && countOwn > 1 && countPart > 1) {
    if (lhoT == 0 && rhoT == 0) {
      wr |= DD_BIT(t.sec_c[suit]);
      qt++;
      if (qt >= cutoff) return qt;
      if (countLho <= 2 && countRho <= 2 && (countPart <= 2 || countOwn <= 2)) {
        qt += (countOwn > countPart ? countOwn : countPart) - 2;
        if (qt >= cutoff) return qt;
        res = 2;
        return qt;
      }
    }
  }
  res = 0;
  return qt;
}

DD_FN int dd_qt_lead_nt(const DDTop& t, int hand, int cutoff, int countLho, int countRho,
                        int& lhoT, int& rhoT, bool comm, int commSuit, int countOwn,
                        int countPart, int suit, int qt, int trump, uint64_t& wr, int& res) {
  res = 1;
  wr |= DD_BIT(t.win_c[suit]);
  qt++;
  if (qt >= cutoff) return qt;
  if (trump == suit && (!comm || suit != commSuit)) {
    lhoT = lhoT > 0 ? lhoT - 1 : 0;
    rhoT = rhoT > 0 ? rhoT - 1 : 0;
  }
  if (countLho <= 1 && countRho <= 1 && countPart <= 1) {
    qt += countOwn - 1;
    if (qt >= cutoff) return qt;
    res = 2;
    return qt;
  }
  if (t.sec_h[suit] == hand) {
    wr |= DD_BIT(t.sec_c[suit]);
    qt++;
    if (qt >= cutoff) return qt;
    if (trump == suit && (!comm || suit != commSuit)) {
      lhoT = lhoT > 0 ? lhoT - 1 : 0;
      rhoT = rhoT > 0 ? rhoT - 1 : 0;
    }
    if (countLho <= 2 && countRho <= 2 && countPart <= 2) {
      qt += countOwn - 2;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  } else if (t.sec_h[suit] == DD_P(hand) && countOwn > 1 && countPart > 1) {
    wr |= DD_BIT(t.sec_c[suit]);
    qt++;
    if (qt >= cutoff) return qt;
    if (trump == suit && (!comm || suit != commSuit)) {
      lhoT = lhoT > 0 ? lhoT - 1 : 0;
      rhoT = rhoT > 0 ? rhoT - 1 : 0;
    }
    if (countLho <= 2 && countRho <= 2 && (countPart <= 2 || countOwn <= 2)) {
      qt += (countOwn > countPart ? countOwn : countPart) - 2;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  }
  res = 0;
  return qt;
}

DD_FN int dd_qt_partner_trump(const DDTop& t, int hand, int cutoff, int countLho, int countRho,
                              int lhoT, int rhoT, int countOwn, int countPart, int suit, int qt,
                              int commSuit, int commCard, uint64_t& wr, int& res) {
  res = 1;
  if ((countLho != 0 || lhoT == 0) && (countRho != 0 || rhoT == 0)) {
    wr |= DD_BIT(t.win_c[suit]) | DD_BIT(commCard);
    qt++;
    if (qt >= cutoff) return qt;
    if (countLho <= 1 && countRho <= 1 && countOwn <= 1 && lhoT == 0 && rhoT == 0) {
      qt += countPart - 1;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  }
  if (t.sec_h[suit] == DD_P(hand)) {
    if (lhoT == 0 && rhoT == 0) {
      wr |= DD_BIT(t.sec_c[suit]) | DD_BIT(commCard);
      qt++;
      if (qt >= cutoff) return qt;
      if (countLho <= 2 && countRho <= 2 && countOwn <= 2) {
        qt += countPart - 2;
        if (qt >= cutoff) return qt;
        res = 2;
        return qt;
      }
    }
  } else if (t.sec_h[suit] == hand && countPart > 1 && countOwn > 1) {
    if (lhoT == 0 && rhoT == 0) {
      wr |= DD_BIT(t.sec_c[suit]) | DD_BIT(commCard);
      qt++;
      if (qt >= cutoff) return qt;
      if (countLho <= 2 && countRho <= 2 && (countOwn <= 2 || countPart <= 2)) {
        qt += (countPart > countOwn ? countPart : countOwn) - 2;
        if (qt >= cutoff) return qt;
        res = 2;
        return qt;
      }
    }
  } else if (suit == commSuit && t.sec_h[suit] == DD_L(hand) && (countLho >= 2 || lhoT == 0) &&
             (countRho >= 2 || rhoT == 0)) {
    if (t.thr_h[suit] == DD_P(hand)) {
      wr |= DD_BIT(t.thr_c[suit]) | DD_BIT(commCard);
      qt++;
      if (qt >= cutoff) return qt;
      if (countOwn <= 2 && countLho <= 2 && countRho <= 2 && lhoT == 0 && rhoT == 0) {
        qt += countPart - 2;
        if (qt >= cutoff) return qt;
      }
    }
  }
  res = 0;
  return qt;
}

DD_FN int dd_qt_partner_nt(const DDTop& t, int hand, int cutoff, int countLho, int countRho,
                           int countOwn, int countPart, int suit, int qt, int commSuit,
                           int commCard, uint64_t& wr, int& res) {
  res = 1;
  wr |= DD_BIT(t.win_c[suit]) | DD_BIT(commCard);
  qt++;
  if (qt >= cutoff) return qt;
  if (countLho <= 1 && countRho <= 1 && countOwn <= 1) {
    qt += countPart - 1;
    if (qt >= cutoff) return qt;
    res = 2;
    return qt;
  }
  if (t.sec_h[suit] == DD_P(hand)) {
    wr |= DD_BIT(t.sec_c[suit]);
    qt++;
    if (qt >= cutoff) return qt;
    if (countLho <= 2 && countRho <= 2 && countOwn <= 2) {
      qt += countPart - 2;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  } else if (t.sec_h[suit] == hand && countPart > 1 && countOwn > 1) {
    wr |= DD_BIT(t.sec_c[suit]);
    qt++;
    if (qt >= cutoff) return qt;
    if (countLho <= 2 && countRho <= 2 && (countOwn <= 2 || countPart <= 2)) {
      qt += (countPart > countOwn ? countPart : countOwn) - 2;
      if (qt >= cutoff) return qt;
      res = 2;
      return qt;
    }
  } else if (suit == commSuit && t.sec_h[suit] == DD_L(hand)) {
    if (t.thr_h[suit] == DD_P(hand)) {
      wr |= DD_BIT(t.thr_c[suit]);
      qt++;
      if (qt >= cutoff) return qt;
      if (countOwn <= 2 && countLho <= 2 && countRho <= 2) {
        qt += countPart - 2;
        if (qt >= cutoff) return qt;
      }
    }
  }
  res = 0;
  return qt;
}

// DDS QuickTricks for the hand on lead. `cutoff` = tricks the leading side needs.
// result = true: decided. Then qtricks > 0 means the leading side makes it, 0 means it fails
// (in NT with no quick trick the opponents win the first trick; `cutoff_other` is what
// they need).
DD_FN int dd_quick_tricks(const uint64_t* hands, const DDTop& t, int trump, int hand, int cutoff,
                          int cutoff_other, uint64_t& wr, bool& result) {
  const int8_t(*len)[4] = t.len;
  int P = DD_P(hand), L = DD_L(hand), R = DD_R(hand);
  int commCard = 0, commSuit = -1, lowestQ = 0, qtricks = 0;
  bool comm = false;
  result = true;

  for (int s = 0; s < 4; s++) {
    if (trump >= 0 && trump != s) {
      if (t.win_h[s] == P) {
        if (len[hand][s] != 0 && (len[L][s] != 0 || len[L][trump] == 0) &&
            (len[R][s] != 0 || len[R][trump] == 0)) {
          comm = true; commSuit = s; commCard = t.win_c[s];
          break;
        }
      } else if (t.sec_h[s] == P && t.win_h[s] == hand && len[hand][s] >= 2 && len[P][s] >= 2) {
        if ((len[L][s] != 0 || len[L][trump] == 0) && (len[R][s] != 0 || len[R][trump] == 0)) {
          comm = true; commSuit = s; commCard = t.sec_c[s];
          break;
        }
      }
    } else if (trump < 0) {
      if (t.win_h[s] == P) {
        if (len[hand][s] != 0) {
          comm = true; commSuit = s; commCard = t.win_c[s];
          break;
        }
      } else if (t.sec_h[s] == P && t.win_h[s] == hand && len[hand][s] >= 2 && len[P][s] >= 2) {
        comm = true; commSuit = s; commCard = t.sec_c[s];
        break;
      }
    }
  }
  if (trump >= 0 && !comm && len[hand][trump] != 0 && t.win_h[trump] == P) {
    comm = true; commSuit = trump; commCard = t.win_c[trump];
  }

  int lhoT = 0, rhoT = 0, suit = 0;
  if (trump >= 0) {
    suit = trump;
    lhoT = len[L][trump];
    rhoT = len[R][trump];
  }
  do {
    int countOwn = len[hand][suit], countLho = len[L][suit], countRho = len[R][suit];
    int countPart = len[P][suit];
    int opps = countLho | countRho;
    if (!opps && countPart == 0) {
      if (countOwn == 0) { suit = dd_next_qt_suit(suit, trump); continue; }
      // Long tricks when only the leading hand has cards in the suit.
      if (trump >= 0 && trump != suit) {
        if (lhoT == 0 && rhoT == 0) {
          qtricks += countOwn;
          if (qtricks >= cutoff) return qtricks;
        }
        suit = dd_next_qt_suit(suit, trump);
        continue;
      }
      qtricks += countOwn;
      if (qtricks >= cutoff) return qtricks;
      suit = dd_next_qt_suit(suit, trump);
      continue;
    } else {
      if (!opps && trump >= 0 && suit == trump) {
        int sum = countOwn > countPart ? countOwn : countPart;
        for (int s = 0; s < 4; s++)
          if (sum > 0 && s != trump && countOwn >= countPart && len[hand][s] > 0 && len[P][s] == 0) {
            sum++;
            break;
          }
        if (sum >= cutoff) return sum;
      } else if (!opps) {
        int sum = countOwn < countPart ? countOwn : countPart;
        if (trump < 0) {
          if (sum >= cutoff) return sum;
        } else if (suit != trump && lhoT == 0 && rhoT == 0) {
          if (sum >= cutoff) return sum;
        }
      }
      if (comm) {
        if (!opps && countOwn == 0) {
          if (trump >= 0 && trump != suit) {
            if (lhoT == 0 && rhoT == 0) {
              qtricks += countPart;
              wr |= DD_BIT(commCard);
              if (qtricks >= cutoff) return qtricks;
            }
            suit = dd_next_qt_suit(suit, trump);
            continue;
          }
          qtricks += countPart;
          wr |= DD_BIT(commCard);
          if (qtricks >= cutoff) return qtricks;
          suit = dd_next_qt_suit(suit, trump);
          continue;
        } else {
          if (!opps && trump >= 0 && suit == trump) {
            int sum = countOwn > countPart ? countOwn : countPart;
            for (int s = 0; s < 4; s++)
              if (sum > 0 && s != trump && countOwn <= countPart && len[P][s] > 0 &&
                  len[hand][s] == 0) {
                sum++;
                break;
              }
            if (sum >= cutoff) {
              wr |= DD_BIT(commCard);
              return sum;
            }
          } else if (!opps) {
            int sum = countOwn < countPart ? countOwn : countPart;
            if (trump < 0) {
              if (sum >= cutoff) return sum;
            } else if (suit != trump && lhoT == 0 && rhoT == 0) {
              if (sum >= cutoff) return sum;
            }
          }
        }
      }
    }
    if (t.win_h[suit] < 0) { suit = dd_next_qt_suit(suit, trump); continue; }
    int res;
    if (t.win_h[suit] == hand) {
      if (trump >= 0 && trump != suit)
        qtricks = dd_qt_lead_trump(t, hand, cutoff, countLho, countRho, lhoT, rhoT, countOwn,
                                   countPart, suit, qtricks, wr, res);
      else
        qtricks = dd_qt_lead_nt(t, hand, cutoff, countLho, countRho, lhoT, rhoT, comm, commSuit,
                                countOwn, countPart, suit, qtricks, trump, wr, res);
      if (res == 1) return qtricks;
      if (res == 2) { suit = dd_next_qt_suit(suit, trump); continue; }
    } else if (t.win_h[suit] == P && comm) {
      if (trump >= 0 && trump != suit)
        qtricks = dd_qt_partner_trump(t, hand, cutoff, countLho, countRho, lhoT, rhoT, countOwn,
                                      countPart, suit, qtricks, commSuit, commCard, wr, res);
      else
        qtricks = dd_qt_partner_nt(t, hand, cutoff, countLho, countRho, countOwn, countPart,
                                   suit, qtricks, commSuit, commCard, wr, res);
      if (res == 1) return qtricks;
      if (res == 2) { suit = dd_next_qt_suit(suit, trump); continue; }
    }

    // A ruff by partner.
    if (trump >= 0 && suit != trump && countOwn > 0 && lowestQ == 0 &&
        (qtricks == 0 || (t.win_h[suit] != hand && t.win_h[suit] != P &&
                          t.win_h[trump] != hand && t.win_h[trump] != P))) {
      uint64_t pt = hands[P] & DD_SUIT(trump);
      if (countPart == 0 && len[P][trump] > 0) {
        if ((countRho > 0 || len[R][trump] == 0) && (countLho > 0 || len[L][trump] == 0)) {
          lowestQ = 1;
          if (1 >= cutoff) return 1;
          suit = dd_next_qt_suit(suit, trump);
          continue;
        } else if (countRho == 0 && countLho == 0) {
          if (((hands[L] | hands[R]) & DD_SUIT(trump)) < pt) {
            lowestQ = 1;
            wr |= DD_BIT(dd_msb(pt));
            if (1 >= cutoff) return 1;
          }
          suit = dd_next_qt_suit(suit, trump);
          continue;
        } else if (countLho == 0) {
          if ((hands[L] & DD_SUIT(trump)) < pt) {
            lowestQ = 1;
            wr |= DD_BIT(dd_msb(pt));
            if (1 >= cutoff) return 1;
          }
          suit = dd_next_qt_suit(suit, trump);
          continue;
        } else if (countRho == 0) {
          if ((hands[R] & DD_SUIT(trump)) < pt) {
            lowestQ = 1;
            wr |= DD_BIT(dd_msb(pt));
            if (1 >= cutoff) return 1;
          }
          suit = dd_next_qt_suit(suit, trump);
          continue;
        }
      }
    }
    if (qtricks >= cutoff) return qtricks;
    suit = dd_next_qt_suit(suit, trump);
  } while (suit <= 3);

  if (qtricks == 0 && (trump < 0 || t.win_h[trump] < 0)) {
    for (int s = 0; s < 4; s++) {
      if (t.win_h[s] < 0) continue;
      if (len[hand][s] > 0) wr = (wr & ~DD_SUIT(s)) | DD_BIT(t.win_c[s]);
    }
    if (1 >= cutoff_other) return 0;
  }
  result = false;
  return qtricks;
}

// DDS QuickTricksSecondHand: can the side of `hand` (2nd to play) win this trick and enough
// more? `cutoff` = tricks that side needs, counting this one. `t` includes the led card.
DD_FN bool dd_quick_tricks_2nd(const uint64_t* hands, const DDTop& t, int trump, int hand,
                               int lead, int cutoff, uint64_t& wr) {
  int ss = lead >> 4, P = DD_P(hand), L = DD_L(hand);
  uint64_t S = DD_SUIT(ss);
  uint64_t ranks = (hands[hand] | hands[P]) & S;
  wr = 0;
  if (trump >= 0 && ss != trump &&
      ((!(hands[hand] & S) && (hands[hand] & DD_SUIT(trump))) ||
       (!(hands[P] & S) && (hands[P] & DD_SUIT(trump))))) {
    if (!(hands[L] & S) && (hands[L] & DD_SUIT(trump))) return false;
  } else if (ranks > (DD_BIT(lead) | (hands[L] & S))) {
    if (trump >= 0 && ss != trump && (hands[L] & DD_SUIT(trump)) && !(hands[L] & S)) return false;
    wr = DD_BIT(dd_msb(ranks));
  } else {
    return false;
  }
  int qtricks = 1;
  if (qtricks >= cutoff) return true;
  if (trump >= 0) return false;
  int hh = (hands[hand] & S) > (hands[P] & S) ? hand : P;
  if (t.win_h[ss] == hh && t.sec_h[ss] == hh) {
    qtricks++;
    wr |= DD_BIT(t.sec_c[ss]);
    if (qtricks >= cutoff) return true;
  }
  for (int s = 0; s < 4; s++) {
    if (s == ss || !(hands[hh] & DD_SUIT(s))) continue;
    if (!((hands[DD_L(hh)] | hands[DD_R(hh)] | hands[DD_P(hh)]) & DD_SUIT(s))) {
      qtricks += dd_popc(hands[hh] & DD_SUIT(s));
      if (qtricks >= cutoff) return true;
    } else if (t.win_h[s] == hh) {
      qtricks++;
      wr |= DD_BIT(t.win_c[s]);
      if (qtricks >= cutoff) return true;
    }
  }
  return false;
}

// DDS LaterTricksMIN, leader on the MAX side. false = MAX cannot reach target.
DD_FN bool dd_later_min(const DDTop& t, int trump, int hand, int ns_won, int left, int target,
                        uint64_t& wr) {
  int dq = left - 1;  // DDS depth >> 2
  if (trump < 0 || t.win_h[trump] < 0) {
    int sum = 0;
    for (int s = 0; s < 4; s++) {
      int hh = t.win_h[s];
      if (hh >= 0 && DD_MAX(hh)) sum += t.len[hh][s] > t.len[DD_P(hh)][s] ? t.len[hh][s] : t.len[DD_P(hh)][s];
    }
    if (ns_won + sum < target && sum > 0) {
      if (ns_won + dq >= target) return true;
      wr = 0;
      for (int s = 0; s < 4; s++) {
        int wh = t.win_h[s];
        if (wh >= 0 && !DD_MAX(wh) &&
            (t.len[DD_P(wh)][s] || t.len[DD_L(wh)][s] || t.len[DD_R(wh)][s]))
          wr |= DD_BIT(t.win_c[s]);
      }
      return false;
    }
  } else if (!DD_MAX(t.win_h[trump])) {
    if (t.len[hand][trump] == 0 && t.len[DD_P(hand)][trump] == 0) {
      int m = t.len[DD_L(hand)][trump] > t.len[DD_R(hand)][trump] ? t.len[DD_L(hand)][trump]
                                                                 : t.len[DD_R(hand)][trump];
      if (ns_won + dq + 1 - m < target) { wr = 0; return false; }
    } else if (ns_won + dq < target) {
      wr = DD_BIT(t.win_c[trump]);
      return false;
    } else if (ns_won + dq == target) {
      int hh = t.sec_h[trump];
      if (hh < 0) return true;
      if (!DD_MAX(hh) && (t.len[hh][trump] > 1 || t.len[DD_P(hh)][trump] > 1)) {
        wr = DD_BIT(t.sec_c[trump]);
        return false;
      }
    }
  } else {
    int hh = t.sec_h[trump];
    if (hh < 0) return true;
    if (DD_MAX(hh) || t.len[hh][trump] <= 1) return true;
    if (t.win_h[trump] == DD_R(hh)) {
      if (ns_won + dq < target) { wr = DD_BIT(t.sec_c[trump]); return false; }
    } else {
      int h = t.thr_h[trump];
      if (h < 0) return true;
      if (!DD_MAX(h) && ns_won + dq < target) { wr = DD_BIT(t.thr_c[trump]); return false; }
    }
  }
  return true;
}

// DDS LaterTricksMAX, leader on the MIN side. true = MAX reaches target.
DD_FN bool dd_later_max(const DDTop& t, int trump, int hand, int ns_won, int left, int target,
                        uint64_t& wr) {
  int dq = left - 1;
  if (trump < 0 || t.win_h[trump] < 0) {
    int sum = 0;
    for (int s = 0; s < 4; s++) {
      int hh = t.win_h[s];
      if (hh >= 0 && !DD_MAX(hh)) sum += t.len[hh][s] > t.len[DD_P(hh)][s] ? t.len[hh][s] : t.len[DD_P(hh)][s];
    }
    if (ns_won + dq + 1 - sum >= target && sum > 0) {
      if (ns_won + 1 < target) return false;
      wr = 0;
      for (int s = 0; s < 4; s++) {
        int wh = t.win_h[s];
        if (wh >= 0 && DD_MAX(wh) &&
            (t.len[DD_P(wh)][s] || t.len[DD_L(wh)][s] || t.len[DD_R(wh)][s]))
          wr |= DD_BIT(t.win_c[s]);
      }
      return true;
    }
  } else if (DD_MAX(t.win_h[trump])) {
    if (t.len[hand][trump] == 0 && t.len[DD_P(hand)][trump] == 0) {
      int m = t.len[DD_L(hand)][trump] > t.len[DD_R(hand)][trump] ? t.len[DD_L(hand)][trump]
                                                                 : t.len[DD_R(hand)][trump];
      if (ns_won + m >= target) { wr = 0; return true; }
    } else if (ns_won + 1 >= target) {
      wr = DD_BIT(t.win_c[trump]);
      return true;
    } else {
      int hh = t.sec_h[trump];
      if (hh < 0) return false;
      if (DD_MAX(hh) && (t.len[hh][trump] > 1 || t.len[DD_P(hh)][trump] > 1) &&
          ns_won + 2 >= target) {
        wr = DD_BIT(t.sec_c[trump]);
        return true;
      }
    }
  } else {
    int hh = t.sec_h[trump];
    if (hh < 0) return false;
    if (!DD_MAX(hh) || t.len[hh][trump] <= 1) return false;
    if (t.win_h[trump] == DD_R(hh)) {
      if (ns_won + 1 >= target) { wr = DD_BIT(t.sec_c[trump]); return true; }
    } else {
      int h = t.thr_h[trump];
      if (h < 0) return false;
      if (DD_MAX(h) && ns_won + 1 >= target) { wr = DD_BIT(t.thr_c[trump]); return true; }
    }
  }
  return false;
}

// DDS lead weights (heuristic_sorting.cpp: weight_alloc_trump0 / weight_alloc_nt0).
// tt = the card (or its equal) is the TT's best move.
DD_FN int dd_lead_weight(const uint64_t* hands, const DDTop& t, int trump, int left, int lead,
                         int card, bool tt) {
  const int8_t(*len)[4] = t.len;
  int s = card >> 4, P = DD_P(lead), L = DD_L(lead), R = DD_R(lead);
  uint64_t S = DD_SUIT(s), all = hands[0] | hands[1] | hands[2] | hands[3];
  int r_rank = dd_popc(all & S & ~((1ull << card) - 1));  // 1 = highest card of the suit
  uint64_t below = all & S & ((1ull << card) - 1);
  bool seq = below && (hands[lead] >> dd_msb(below) & 1);
  uint64_t risP = hands[P] & S, risL = hands[L] & S, risR = hands[R] & S;
  int cntLH = (len[L][s] == 0 ? left : len[L][s]) << 2;
  int cntRH = (len[R][s] == 0 ? left : len[R][s]) << 2;
  int w;
  if (trump >= 0 && (all & DD_SUIT(trump))) {
    uint64_t T = DD_SUIT(trump);
    int bonus = 0;
    bool win = false;
    if (s != trump && ((!risL && (hands[L] & T)) || (!risR && (hands[R] & T)))) bonus = -12;
    if (s != trump && len[P][s] == 0 && len[P][trump] > 0 && len[R][s] > 0) bonus += 17;
    if (t.win_h[s] == R || t.sec_h[s] == R) {
      if (len[R][s] != 1) bonus += -12;
    } else if (t.win_h[s] == L && t.sec_h[s] == P) {
      if (len[P][s] != 1) bonus += 27;
    }
    if (s != trump && len[lead][s] == 1 && len[lead][trump] > 0 && len[P][s] > 1 && t.win_h[s] == P)
      bonus += 19;
    int delta = bonus - (((cntLH + cntRH) << 5) / 13);
    if (t.win_c[s] == card) {
      if (s != trump) {
        if (len[P][s] != 0 || len[P][trump] == 0) {
          if ((len[L][s] != 0 || len[L][trump] == 0) && (len[R][s] != 0 || len[R][trump] == 0))
            win = true;
        } else if ((len[L][s] != 0 || (hands[P] & T) > (hands[L] & T)) &&
                   (len[R][s] != 0 || (hands[P] & T) > (hands[R] & T))) {
          win = true;
        }
      } else {
        win = true;
      }
    } else if (risP > (risL | risR)) {
      if (s != trump) {
        if ((len[L][s] != 0 || len[L][trump] == 0) && (len[R][s] != 0 || len[R][trump] == 0))
          win = true;
      } else {
        win = true;
      }
    } else if (s != trump) {
      if (len[P][s] == 0 && len[P][trump] != 0) {
        if (len[L][s] == 0 && len[L][trump] != 0 && len[R][s] == 0 && len[R][trump] != 0) {
          if ((hands[P] & T) > ((hands[L] | hands[R]) & T)) win = true;
        } else if (len[L][s] == 0 && len[L][trump] != 0) {
          if ((hands[P] & T) > (hands[L] & T)) win = true;
        } else if (len[R][s] == 0 && len[R][trump] != 0) {
          if ((hands[P] & T) > (hands[R] & T)) win = true;
        } else {
          win = true;
        }
      }
    }
    if (win) {
      if ((len[L][s] == 1 && t.win_h[s] == L) || (len[R][s] == 1 && t.win_h[s] == R))
        w = delta + 35 + r_rank;
      else if (t.win_h[s] == lead) {
        if (t.sec_h[s] == P) w = delta + 48 + r_rank;
        else if (t.win_c[s] == card) w = delta + 31;
        else w = delta - 3 + r_rank;
      } else if (t.win_h[s] == P) {
        w = delta + (t.sec_h[s] == lead ? 42 : 28) + r_rank;
      } else if (seq && card == t.sec_c[s]) {
        w = delta + 40;
      } else if (seq) {
        w = delta + 22 + r_rank;
      } else {
        w = delta + 11 + r_rank;
      }
      if (tt) w += 18;
    } else {
      int third = t.thr_h[s];
      if (t.sec_h[s] == P && P == third) delta += 20;
      else if ((t.sec_h[s] == lead && P == third && len[P][s] > 1) ||
               (t.sec_h[s] == P && lead == third && len[P][s] > 1))
        delta += 13;
      if ((len[L][s] == 1 && t.win_h[s] == L) || (len[R][s] == 1 && t.win_h[s] == R))
        w = delta + r_rank + 2;
      else if (t.win_h[s] == lead) {
        if (t.sec_h[s] == P) w = delta + 33 + r_rank;
        else if (t.win_c[s] == card) w = delta + 38;
        else w = delta - 14 + r_rank;
      } else if (t.win_h[s] == P) {
        w = delta + 34 + r_rank;
      } else if (seq && card == t.sec_c[s]) {
        w = delta + 35;
      } else {
        w = delta + 17 - ((card & 15) + 2);
      }
    }
    return w;
  }

  int delta = -(((cntLH + cntRH) << 5) / 19);
  if (len[P][s] == 0) delta += -9;
  if (t.win_c[s] == card || risP > (risL | risR)) {
    if (t.sec_h[s] == R) {
      if (len[R][s] != 1) delta += -1;
    } else if (t.sec_h[s] == L) {
      delta += len[L][s] != 1 ? 22 : 16;
    }
    if ((t.sec_h[s] != L || len[L][s] == 1) && (t.sec_h[s] != R || len[R][s] == 1))
      w = delta + 45 + r_rank;
    else
      w = delta + 18 + r_rank;
    if (tt) w += 32;
  } else {
    if (t.win_h[s] == R || t.sec_h[s] == R) {
      if (len[R][s] != 1) delta += -10;
    } else if (t.win_h[s] == L && t.sec_h[s] == P) {
      if (len[P][s] != 1) delta += 31;
    }
    int third = t.thr_h[s];
    if (t.sec_h[s] == P && P == third) delta += 35;
    else if ((t.sec_h[s] == lead && P == third && len[P][s] > 1) ||
             (t.sec_h[s] == P && lead == third && len[P][s] > 1))
      delta += 25;
    if ((len[L][s] == 1 && t.win_h[s] == L) || (len[R][s] == 1 && t.win_h[s] == R))
      w = delta + 28 + r_rank;
    else if (t.win_h[s] == lead) w = delta - 17 + r_rank;
    else if (!seq) w = delta + 12 + r_rank;
    else if (card == t.sec_c[s]) w = delta + 48;
    else w = delta + 29 - r_rank;
    if (tt) w += 19;
  }
  return w;
}

// DDS rank scale: 2..14, 0 = no card.
DD_FN int dd_R(int card) { return (card & 15) + 2; }
DD_FN int dd_hi(uint64_t m) { return m ? (dd_msb(m) & 15) + 2 : 0; }
DD_FN int dd_lo(uint64_t m) { return m ? (dd_lsb(m) & 15) + 2 : 0; }

// Hand holding the k-th highest card (k = 0, 1) of mask m among hands th, or -1.
DD_FN int dd_kth_hand(const uint64_t* th, uint64_t m, int k) {
  for (int i = 0; i < k && m; i++) m ^= 1ull << dd_msb(m);
  if (!m) return -1;
  uint64_t b = 1ull << dd_msb(m);
  for (int h = 0; h < 4; h++)
    if (th[h] & b) return h;
  return -1;
}

// Per-position data for dd_follow_weight, computed once per node.
struct DDFollow {
  uint64_t th[4];              // hands at trick start (trick cards given back)
  uint64_t allt, all;          // all cards at trick start / now
  int8_t win_h[4], sec_h[4];   // trick-start holders of the top two cards per suit
  int lead, ls, lead0, P, R, high1, high2, move1, move2;
};

DD_FN void dd_follow_init(DDFollow& f, const uint64_t* hands, int trump, int leader,
                          const int* trick, int n) {
  f.lead = leader;
  f.ls = trick[0] >> 4;
  f.P = DD_P(leader);
  f.R = DD_R(leader);
  for (int h = 0; h < 4; h++) f.th[h] = hands[h];
  for (int i = 0; i < n; i++) f.th[(leader + i) & 3] |= 1ull << trick[i];
  f.all = hands[0] | hands[1] | hands[2] | hands[3];
  f.allt = f.th[0] | f.th[1] | f.th[2] | f.th[3];
  for (int s = 0; s < 4; s++) {
    f.win_h[s] = (int8_t)dd_kth_hand(f.th, f.allt & DD_SUIT(s), 0);
    f.sec_h[s] = (int8_t)dd_kth_hand(f.th, f.allt & DD_SUIT(s), 1);
  }
  f.lead0 = dd_R(trick[0]);
  // Winning card after 2 and 3 cards, as DDS tracks it.
  int w1 = n > 1 && dd_beats(trick[1], trick[0], trump) ? 1 : 0;
  int w2 = n > 2 && dd_beats(trick[2], trick[w1], trump) ? 2 : w1;
  f.high1 = w1;
  f.move1 = trick[w1];
  f.high2 = w2;
  f.move2 = trick[w2];
}

// DDS weights for the 2nd, 3rd and 4th hand (heuristic_sorting.cpp, weight_alloc_*1/2/3),
// without the rank_forces_ace / get_top_number bonuses. n = cards played so far.
DD_FN int dd_follow_weight(const DDFollow& f, const uint64_t* hands, int trump, int n, int seat,
                           int card) {
  int lead = f.lead, ls = f.ls;
  uint64_t LS = DD_SUIT(ls);
  int s = card >> 4, rank = dd_R(card);
  int P = f.P, R = f.R;
  int r_rank = dd_popc(f.all & DD_SUIT(s) & ~((1ull << card) - 1));
  uint64_t below = f.allt & DD_SUIT(s) & ((1ull << card) - 1);
  bool seq = below && (hands[seat] >> dd_msb(below) & 1);
  int lead0 = f.lead0;
  int move1 = f.move1, high1 = f.high1, move2 = f.move2, high2 = f.high2;
  bool is_void = s != ls;
  int suitCount = dd_popc(hands[seat] & DD_SUIT(s));
  bool sec_mine = f.sec_h[s] == seat;
  bool win_mine = f.win_h[s] == seat;
  if (n == 1) {
    int max3rd = dd_hi(hands[P] & LS), maxpd = dd_hi(hands[R] & LS);
    int min3rd = dd_lo(hands[P] & LS), minpd = dd_lo(hands[R] & LS);
    if (!is_void) {
      if (trump >= 0) {
        uint64_t T = DD_SUIT(trump);
        bool win = false;
        if (ls == trump) {
          if (maxpd > lead0 && maxpd > max3rd) win = true;
          else if (rank > lead0 && rank > max3rd) win = true;
        } else if (rank > lead0 && rank > max3rd) {
          if (max3rd != 0 || !(hands[P] & T)) win = true;
          else if (maxpd == 0 && (hands[R] & T) && (hands[R] & T) > (hands[P] & T)) win = true;
        } else if (maxpd > lead0 && maxpd > max3rd) {
          if (max3rd != 0 || !(hands[P] & T)) win = true;
        } else if (lead0 > maxpd && lead0 > max3rd && lead0 > rank) {
          if (maxpd == 0 && (hands[R] & T)) {
            if (max3rd != 0 || !(hands[P] & T)) win = true;
            else if ((hands[R] & T) > (hands[P] & T)) win = true;
          }
        } else if (maxpd == 0 && (hands[R] & T)) {
          win = true;
        }
        if (win) {
          if (min3rd > rank) return 40 + r_rank;
          if (maxpd > lead0 && (hands[lead] & LS) > (hands[R] & LS)) return 41 + r_rank;
          if (rank > lead0) {
            if (rank < maxpd) return 78 - rank;
            if (rank > max3rd) return 73 - rank;
            if (seq) return 62 - rank;
            return 49 - rank;
          }
          return maxpd > 0 ? 47 - rank : 40 - rank;
        }
        if (rank < min3rd || rank < minpd) return -9 + r_rank;
        if (rank < lead0) return -16 + r_rank;
        return seq ? 22 - rank : 10 - rank;
      }
      if (maxpd > lead0 && maxpd > max3rd) return -rank;
      if (rank > lead0 && rank > max3rd) return 81 - rank;
      if (min3rd > rank || minpd > rank) return -3 + r_rank;
      if (rank < lead0) return -11 + r_rank;
      return seq ? 10 + r_rank : 13 - rank;
    }
    uint64_t lead_bit = 1ull << ((ls << 4) | (lead0 - 2));
    if (trump >= 0) {
      uint64_t T = DD_SUIT(trump);
      int add;
      if (ls == trump) {
        if ((hands[R] & LS) > ((hands[P] & LS) | lead_bit)) add = (suitCount << 6) / 44;
        else {
          add = (suitCount << 6) / 36;
          if (suitCount == 2 && sec_mine) add += -4;
        }
        return -rank + add;
      }
      if (s != trump) {
        if (hands[P] & LS) {
          if ((hands[R] & LS) > ((hands[P] & LS) | lead_bit)) add = 60 + (suitCount << 6) / 44;
          else if (!(hands[R] & LS) && (hands[R] & T)) add = 60 + (suitCount << 6) / 44;
          else {
            add = -2 + (suitCount << 6) / 36;
            if (suitCount == 2 && sec_mine) add += -4;
          }
        } else if (!(hands[R] & LS) && (hands[R] & T) > (hands[P] & T)) {
          add = 60 + (suitCount << 6) / 44;
        } else if (!(hands[P] & T) && (hands[R] & LS) > lead_bit) {
          add = 60 + (suitCount << 6) / 44;
        } else {
          add = -2 + (suitCount << 6) / 36;
          if (suitCount == 2 && sec_mine) add += -4;
        }
        return -rank + add;
      }
      if ((hands[P] & LS) || (!(hands[R] & LS) && (hands[R] & T) && (hands[R] & T) > (hands[P] & T)))
        return 24 - rank + (suitCount << 6) / 44;
      if ((1ull << card) > (hands[P] & T)) return 24 - rank + (suitCount << 6) / 44;
      add = (suitCount << 6) / 36;
      if (suitCount == 2 && sec_mine) add += -4;
      return 15 - rank + add;
    }
    int add;
    if ((hands[R] & LS) > ((hands[P] & LS) | lead_bit)) {
      add = (suitCount << 6) / 23;
      if (suitCount == 2 && sec_mine) add += -2;
      else if (suitCount == 1 && win_mine) add += -3;
    } else {
      add = (suitCount << 6) / 33;
      if (suitCount == 2 && sec_mine) add += -6;
      else if (suitCount == 1 && win_mine) add += -8;
    }
    return -rank + add;
  }

  if (n == 2) {
    uint64_t cards4th = hands[R] & LS;
    int max4th = dd_hi(cards4th), min4th = dd_lo(cards4th);
    int move1_rank = dd_R(move1), move1_suit = move1 >> 4;
    if (!is_void) {
      int max3rd = dd_hi(hands[seat] & LS);
      if (trump >= 0) {
        if (ls == trump) {
          if (high1 == 0 && lead0 > max4th) return -rank;
          if (max3rd < min4th || max3rd < move1_rank) return -rank;
          if (max3rd > max4th) return rank > max4th && rank > move1_rank ? 58 - rank : -rank;
          return -rank;
        }
        if (move1_suit == trump) return -rank;
        if (high1 == 0) {
          if (max4th == 0 || lead0 > max4th) return -rank;
          if (max3rd < min4th || max3rd < move1_rank) return -rank;
          if (max3rd > max4th) return rank > max4th ? 58 - rank : -rank;
          return rank > move1_rank && rank > max4th ? 60 - rank : -rank;
        }
        if (max4th == 0) return rank > move1_rank ? 20 - rank : -rank;
        if (max3rd < min4th || max3rd < move1_rank) return -rank;
        if (max3rd > max4th) return rank > move1_rank && rank > max4th ? 58 - rank : -rank;
        return rank > move1_rank && rank > max4th ? 60 - rank : -rank;
      }
      if (high1 == 0 && lead0 > max4th) return -rank;
      if (max3rd < min4th || max3rd < move1_rank) return -rank;
      return rank > move1_rank && rank > max4th ? 60 - rank : -rank;
    }
    if (trump >= 0) {
      uint64_t T = DD_SUIT(trump);
      if (ls == trump || s != trump) return -rank + (suitCount << 6) / 40;
      if (high1 == 0 && lead0 > max4th && (max4th != 0 || !(hands[R] & T)))
        return -rank - 50;
      if (move1_suit == trump && rank < move1_rank) return -32 + r_rank + (suitCount << 6) / 40;
      int add = (suitCount << 6) / 50;
      if (high1 == 0) {
        if (max4th != 0) return (f.sec_h[ls] == lead ? 36 : 48) - rank + add;
        if ((1ull << card) > (hands[R] & T)) return 48 - rank + add;
        return -12 - rank + add;
      }
      if (max4th != 0) return 72 - rank + add;
      if ((1ull << card) > (hands[R] & T)) return 48 - rank + add;
      return 36 - rank + add;
    }
    int add = (suitCount << 6) / 24;
    if (suitCount == 2 && sec_mine) add -= 4;
    if (suitCount == 1 && win_mine) add -= 4;
    return -rank + add;
  }

  // 4th hand.
  int move2_rank = dd_R(move2), move2_suit = move2 >> 4;
  if (!is_void) {
    if (high2 == 1 || (ls != trump && move2_suit == trump)) return -rank;
    return rank > move2_rank ? 30 - rank : -rank;
  }
  if (trump >= 0) {
    int val = (suitCount << 6) / 24;
    if (suitCount == 2 && sec_mine) val -= 2;
    if (ls == trump) return -rank + val;
    if (high2 == 1) return s == trump ? 2 - rank + val : 25 - rank + val;
    if (move2_suit == trump) {
      if (s == trump) return rank > move2_rank ? 33 + r_rank : -13 + r_rank;
      return 14 - rank + val;
    }
    if (s == trump) return 33 + r_rank;
    return 14 - rank + val;
  }
  int val = (suitCount << 6) / 27;
  if (suitCount == 2 && sec_mine) val -= 6;
  else if (suitCount == 1 && win_mine) val -= 8;
  return -rank + val;
}
