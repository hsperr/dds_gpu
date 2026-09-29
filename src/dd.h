// Double-dummy solver core. One solve = one (deal, strain); shared by CPU and CUDA builds.
//
// Search is a boolean test "can NS take >= target tricks?" (null window), not minimax
// over trick counts. The trick count comes from a binary search over targets that
// reuses one transposition table (TT).
//
// Partition search (as in DDS): every search also returns `rel`, the cards whose rank
// decided the result. Below the lowest relevant card of a suit, cards are "small" and
// only their count per hand matters. A TT entry stores only the owners of the relevant
// top cards plus every hand's suit lengths, so one entry covers many positions, and
// small cards of one suit are tried only once.
//
// Cards: bit (suit * 16 + rank) of a uint64. suit 0..3 = S,H,D,C. rank 0..12 = 2..A.
// Seats 0..3 = N,E,S,W. Strains 0..4 = C,D,H,S,NT (Pgx order).
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define DD_FN __host__ __device__ __forceinline__
#define DD_REC __host__ __device__
#else
#define DD_FN static inline
#define DD_REC static
#endif

DD_FN int dd_popc(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}
DD_FN int dd_msb(uint64_t x) {  // x != 0
#ifdef __CUDA_ARCH__
  return 63 - __clzll(x);
#else
  return 63 - __builtin_clzll(x);
#endif
}
DD_FN int dd_lsb(uint64_t x) {  // x != 0
#ifdef __CUDA_ARCH__
  return __ffsll((long long)x) - 1;
#else
  return __builtin_ctzll(x);
#endif
}

#define DD_SUIT(s) (0x1FFFull << (16 * (s)))
#define DD_TT_WAYS 16
#define DD_GEN_BITS 6

// TT entry, 32 bytes. A bucket is DD_TT_WAYS entries with the same hash.
//   pat: suits 0,1 | suits 2,3 top-card owner patterns (27 bits each, leading 1 bit)
//   lens: 4 bits per (hand, suit), exact
//   meta: gen (6) | leader (2) | lb (4) | ub (4) | best card + 1 (7)
struct DDEntry {
  uint64_t pat0, pat1, lens, meta;
#ifdef DD_VERIFY
  uint64_t dbg_hand[4], dbg_rel;
  int dbg_leader, dbg_lb, dbg_ub;
#endif
};

// Tags come first so a probe reads one 64-byte line unless a tag matches.
//   tag: hash bits (24) | gen (6) | leader (2); 0 = empty
struct DDBucket {
  uint32_t tag[DD_TT_WAYS];
  DDEntry e[DD_TT_WAYS];
};

struct DDCtx {
  uint64_t hand[4];
  int trump;  // suit 0..3, or -1 for NT
  DDBucket* tt;
  uint32_t tt_mask;  // bucket count - 1
  uint32_t gen;      // 1..63; entries of other generations are empty
  uint64_t nodes;
  uint64_t st_lead_cut[14], st_lead_all, st_tricks, st_probe, st_hit, st_qt, st_cut[14], st_nocut;
  int leader, nplayed, ns_won, left;  // left = tricks not yet finished
  int trick[4];                       // cards of the current trick, in play order
#ifdef DD_VERIFY
  int no_tt;
#endif
};

// One suit's remaining cards, high to low, as 2-bit owners behind a leading 1 bit.
DD_FN uint64_t dd_suit_code(const DDCtx& c, int s) {
  uint64_t code = 1;
  uint64_t h1 = c.hand[1] & DD_SUIT(s), h2 = c.hand[2] & DD_SUIT(s), h3 = c.hand[3] & DD_SUIT(s);
  uint64_t all = (c.hand[0] & DD_SUIT(s)) | h1 | h2 | h3;
  while (all) {
    uint64_t bit = 1ull << dd_msb(all);
    int owner = (h1 & bit) ? 1 : (h2 & bit) ? 2 : (h3 & bit) ? 3 : 0;
    code = (code << 2) | (uint64_t)owner;
    all ^= bit;
  }
  return code;
}

DD_FN uint64_t dd_lengths(const DDCtx& c) {
  uint64_t k = 0;
  for (int h = 0; h < 4; h++)
    for (int s = 0; s < 4; s++)
      k = (k << 4) | (uint64_t)dd_popc(c.hand[h] & DD_SUIT(s));
  return k;
}

DD_FN uint64_t dd_hash(uint64_t lens, int leader) {
  uint64_t h = (lens ^ ((uint64_t)leader << 62)) * 0x9E3779B97F4A7C15ull;
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 32);
}

// Top `n` remaining cards of each suit, as a card mask; n from a stored pattern.
DD_FN uint64_t dd_top_cards(uint64_t all, int s, int n) {
  uint64_t out = 0, a = all & DD_SUIT(s);
  for (int i = 0; i < n; i++) {
    uint64_t bit = 1ull << dd_msb(a);
    out |= bit;
    a ^= bit;
  }
  return out;
}

// Owner pattern of the relevant top cards of suit s (all cards >= lowest rel card).
DD_FN uint64_t dd_pattern(uint64_t code, uint64_t all, uint64_t rel, int s) {
  uint64_t r = rel & DD_SUIT(s);
  if (!r) return 1;
  int low = dd_lsb(r);
  int len = dd_popc(all & DD_SUIT(s));
  int n = dd_popc(all & DD_SUIT(s) & ~((1ull << low) - 1));
  return code >> (2 * (len - n));
}

DD_FN int dd_pattern_len(uint64_t p) { return dd_msb(p) >> 1; }

DD_FN bool dd_pattern_match(uint64_t p, uint64_t code) {
  int n = dd_pattern_len(p), len = dd_pattern_len(code);
  return (code >> (2 * (len - n))) == p;
}

// What separates TT entries of different solves. An entry describes a class of positions
// (owners of the relevant cards, all suit lengths, leader), valid for any deal with the
// same trump suit. DD_SHARED_TT keys on the trump suit only, so one TT can serve every
// deal; otherwise each solve gets its own generation.
#ifdef DD_SHARED_TT
#define DD_KEY(c) ((uint32_t)((c).trump + 2))
#else
#define DD_KEY(c) ((c).gen)
#endif

struct DDNode {  // trick-start data kept for the TT store
  uint64_t code[4], lens, all;
  DDBucket* bucket;
  uint32_t tag;
};

DD_FN void dd_node_init(const DDCtx& c, DDNode& nd) {
  for (int s = 0; s < 4; s++) nd.code[s] = dd_suit_code(c, s);
  nd.lens = dd_lengths(c);
  nd.all = c.hand[0] | c.hand[1] | c.hand[2] | c.hand[3];
  uint64_t h = dd_hash(nd.lens, c.leader);
  nd.bucket = c.tt + (h & c.tt_mask);
  nd.tag = ((uint32_t)(h >> 32) & 0xFFFFFF00u) | (DD_KEY(c) << 2) | (uint32_t)c.leader;
}

#ifdef DD_SHARED_TT
// Shared TT (many threads, no locks). Entries are written word by word, so a reader may
// see a half-written one; the checksum in meta's top 32 bits makes it miss instead. New
// entries overwrite one way in place (no shifting, which would race).
DD_FN uint64_t dd_entry_check(uint64_t pat0, uint64_t pat1, uint64_t lens, uint64_t meta) {
  uint64_t h = (pat0 * 0x9E3779B97F4A7C15ull) ^ (pat1 + 0x632BE59BD9B4E019ull) ^
               (lens * 0xC2B2AE3D27D4EB4Full) ^ ((meta & 0xFFFFFFFFull) * 0x165667B19E3779F9ull);
  h ^= h >> 31;
  h *= 0xBF58476D1CE4E5B9ull;
  return (h >> 32) << 32;
}

DD_FN int dd_tt_probe(const DDCtx& c, const DDNode& nd, int need, uint64_t* rel, int* best) {
  uint64_t tag = ((uint64_t)DD_KEY(c)) | ((uint64_t)c.leader << DD_GEN_BITS);
  for (int i = 0; i < DD_TT_WAYS; i++) {
    if (((volatile uint32_t*)nd.bucket->tag)[i] != nd.tag) continue;
    volatile const DDEntry& v = nd.bucket->e[i];
    uint64_t pat0 = v.pat0, pat1 = v.pat1, lens = v.lens, meta = v.meta;
    if ((meta & 0xFF) != tag || lens != nd.lens) continue;
    if ((meta & ~0xFFFFFFFFull) != dd_entry_check(pat0, pat1, lens, meta)) continue;
    uint64_t p[4] = {pat0 & 0x7FFFFFF, pat0 >> 27, pat1 & 0x7FFFFFF, pat1 >> 27};
    bool match = true;
    for (int s = 0; s < 4 && match; s++) match = dd_pattern_match(p[s], nd.code[s]);
    if (!match) continue;
    int lb = (int)(meta >> 8) & 15, ub = (int)(meta >> 12) & 15;
    if (lb >= need || ub < need) {
      uint64_t r = 0;
      for (int s = 0; s < 4; s++) r |= dd_top_cards(nd.all, s, dd_pattern_len(p[s]));
      *rel = r;
      return lb >= need;
    }
    int b = (int)(meta >> 16) & 127;
    if (b) *best = b - 1;
  }
  return -1;
}

DD_FN void dd_tt_store(DDCtx& c, const DDNode& nd, uint64_t rel, int lb, int ub, int best) {
  uint64_t pat0 = dd_pattern(nd.code[0], nd.all, rel, 0) |
                  (dd_pattern(nd.code[1], nd.all, rel, 1) << 27);
  uint64_t pat1 = dd_pattern(nd.code[2], nd.all, rel, 2) |
                  (dd_pattern(nd.code[3], nd.all, rel, 3) << 27);
  uint64_t tag = ((uint64_t)DD_KEY(c)) | ((uint64_t)c.leader << DD_GEN_BITS);
  DDBucket* b = nd.bucket;
  volatile uint32_t* tags = (volatile uint32_t*)b->tag;
  int victim = -1;
  for (int i = 0; i < DD_TT_WAYS; i++) {
    uint32_t t = tags[i];
    if (t == 0 && victim < 0) victim = i;
    if (t != nd.tag) continue;
    volatile DDEntry& v = b->e[i];
    uint64_t m = v.meta;
    if ((m & 0xFF) == tag && v.lens == nd.lens && v.pat0 == pat0 && v.pat1 == pat1 &&
        (m & ~0xFFFFFFFFull) == dd_entry_check(pat0, pat1, nd.lens, m)) {
      int olb = (int)(m >> 8) & 15, oub = (int)(m >> 12) & 15;
      if (olb > lb) lb = olb;
      if (oub < ub) ub = oub;
      if (best < 0) best = (int)((m >> 16) & 127) - 1;
      uint64_t meta = tag | ((uint64_t)lb << 8) | ((uint64_t)ub << 12) | ((uint64_t)(best + 1) << 16);
      v.meta = meta | dd_entry_check(pat0, pat1, nd.lens, meta);
      return;
    }
  }
  if (victim < 0) victim = (int)((c.nodes ^ (c.nodes >> 7)) & (DD_TT_WAYS - 1));
  uint64_t meta = tag | ((uint64_t)lb << 8) | ((uint64_t)ub << 12) | ((uint64_t)(best + 1) << 16);
  volatile DDEntry& v = b->e[victim];
  tags[victim] = 0;
  v.pat0 = pat0;
  v.pat1 = pat1;
  v.lens = nd.lens;
  v.meta = meta | dd_entry_check(pat0, pat1, nd.lens, meta);
  tags[victim] = nd.tag;
}
#else
// Looks up bounds of NS tricks still to win. Returns 1 true, 0 false, -1 unknown.
DD_FN int dd_tt_probe(const DDCtx& c, const DDNode& nd, int need, uint64_t* rel, int* best) {
  uint64_t tag = ((uint64_t)DD_KEY(c)) | ((uint64_t)c.leader << DD_GEN_BITS);
  for (int i = 0; i < DD_TT_WAYS; i++) {
    if (nd.bucket->tag[i] != nd.tag) continue;
    const DDEntry& e = nd.bucket->e[i];
    if ((e.meta & 0xFF) != tag || e.lens != nd.lens) continue;
    uint64_t p[4] = {e.pat0 & 0x7FFFFFF, e.pat0 >> 27, e.pat1 & 0x7FFFFFF, e.pat1 >> 27};
    bool match = true;
    for (int s = 0; s < 4 && match; s++) match = dd_pattern_match(p[s], nd.code[s]);
    if (!match) continue;
    int lb = (int)(e.meta >> 8) & 15, ub = (int)(e.meta >> 12) & 15;
    if (lb >= need || ub < need) {
      uint64_t r = 0;
      for (int s = 0; s < 4; s++) r |= dd_top_cards(nd.all, s, dd_pattern_len(p[s]));
      *rel = r;
#ifdef DD_VERIFY
      printf("  [probe] entry from hands %llx %llx %llx %llx leader %d lb %d ub %d rel %llx\n",
             (unsigned long long)e.dbg_hand[0], (unsigned long long)e.dbg_hand[1],
             (unsigned long long)e.dbg_hand[2], (unsigned long long)e.dbg_hand[3], e.dbg_leader,
             e.dbg_lb, e.dbg_ub, (unsigned long long)e.dbg_rel);
#endif
      return lb >= need;
    }
    int b = (int)(e.meta >> 16) & 127;
    if (b) *best = b - 1;
  }
  return -1;
}

DD_FN void dd_tt_store(DDCtx& c, const DDNode& nd, uint64_t rel, int lb, int ub, int best) {
#ifdef DD_VERIFY
  if (c.no_tt) return;
#endif
  uint64_t pat0 = dd_pattern(nd.code[0], nd.all, rel, 0) |
                  (dd_pattern(nd.code[1], nd.all, rel, 1) << 27);
  uint64_t pat1 = dd_pattern(nd.code[2], nd.all, rel, 2) |
                  (dd_pattern(nd.code[3], nd.all, rel, 3) << 27);
  uint64_t tag = ((uint64_t)DD_KEY(c)) | ((uint64_t)c.leader << DD_GEN_BITS);
  DDBucket* b = nd.bucket;
  for (int i = 0; i < DD_TT_WAYS; i++) {
    if (b->tag[i] != nd.tag) continue;
    DDEntry& e = b->e[i];
    if ((e.meta & 0xFF) == tag && e.lens == nd.lens && e.pat0 == pat0 && e.pat1 == pat1) {
      int olb = (int)(e.meta >> 8) & 15, oub = (int)(e.meta >> 12) & 15;
      if (olb > lb) lb = olb;
      if (oub < ub) ub = oub;
      if (best < 0) best = (int)((e.meta >> 16) & 127) - 1;
      e.meta = tag | ((uint64_t)lb << 8) | ((uint64_t)ub << 12) | ((uint64_t)(best + 1) << 16);
#ifdef DD_VERIFY
      for (int h = 0; h < 4; h++) e.dbg_hand[h] = c.hand[h];
      e.dbg_rel = rel; e.dbg_leader = c.leader; e.dbg_lb = lb; e.dbg_ub = ub;
#endif
      return;
    }
  }
  // New entries go in front; the last way falls out.
  for (int i = DD_TT_WAYS - 1; i > 0; i--) {
    b->tag[i] = b->tag[i - 1];
    b->e[i] = b->e[i - 1];
  }
  b->tag[0] = nd.tag;
  b->e[0].pat0 = pat0;
  b->e[0].pat1 = pat1;
  b->e[0].lens = nd.lens;
  b->e[0].meta = tag | ((uint64_t)lb << 8) | ((uint64_t)ub << 12) | ((uint64_t)(best + 1) << 16);
#ifdef DD_VERIFY
  for (int h = 0; h < 4; h++) b->e[0].dbg_hand[h] = c.hand[h];
  b->e[0].dbg_rel = rel; b->e[0].dbg_leader = c.leader; b->e[0].dbg_lb = lb; b->e[0].dbg_ub = ub;
#endif
}

#endif

DD_FN bool dd_beats(int card, int best, int trump) {
  int s = card >> 4, bs = best >> 4;
  if (s == bs) return card > best;
  return s == trump;
}

// Index (in play order) of the card winning the current trick so far.
DD_FN int dd_winning(const DDCtx& c, int n) {
  int w = 0;
  for (int i = 1; i < n; i++)
    if (dd_beats(c.trick[i], c.trick[w], c.trump)) w = i;
  return w;
}

#include "dd_bounds.h"

// Same suit, and no card of another hand between them: the two cards are alike.
DD_FN bool dd_equal(uint64_t own, uint64_t all, int a, int b) {
  if ((a >> 4) != (b >> 4)) return false;
  int lo = a < b ? a : b, hi = a < b ? b : a;
  uint64_t between = ((1ull << hi) - 1) & ~((2ull << lo) - 1);
  return !(all & ~own & between);
}

// Candidate cards for the hand to play, one per run of equal cards, best-first.
DD_FN int dd_moves(const DDCtx& c, int seat, int hint, const DDTop* top, int* out) {
  uint64_t own = c.hand[seat];
  uint64_t legal = own;
  if (c.nplayed) {
    uint64_t f = own & DD_SUIT(c.trick[0] >> 4);
    if (f) legal = f;
  }
  uint64_t all = c.hand[0] | c.hand[1] | c.hand[2] | c.hand[3];
  for (int i = 0; i < c.nplayed; i++) all |= 1ull << c.trick[i];

  DDFollow f;
  if (c.nplayed) dd_follow_init(f, c.hand, c.trump, c.leader, c.trick, c.nplayed);

  int n = 0, score[13];
  uint64_t rest = legal;
  while (rest) {
    int card = dd_msb(rest);
    rest ^= 1ull << card;
    int s = card >> 4;
    // Skip cards touching a higher card of the same hand (equivalent).
    uint64_t above = all & DD_SUIT(s) & ~((2ull << card) - 1);
    if (above && (own & (1ull << dd_lsb(above)))) continue;

    int sc;
    if (c.nplayed == 0) {
      sc = dd_lead_weight(c.hand, *top, c.trump, c.left, seat, card,
                          hint >= 0 && dd_equal(own, all, hint, card));
    } else {
      sc = dd_follow_weight(f, c.hand, c.trump, c.nplayed, seat, card);
    }
    if (c.nplayed && hint >= 0 && dd_equal(own, all, hint, card))
      sc = 1000;  // the hinted card, or one equal to it
    int i = n++;
    while (i > 0 && score[i - 1] < sc) {
      score[i] = score[i - 1];
      out[i] = out[i - 1];
      i--;
    }
    score[i] = sc;
    out[i] = card;
  }
  return n;
}

DD_REC bool dd_search(DDCtx& c, int target, uint64_t* rel);

// One level of the search. Kept small: the search runs as a loop over an array of these
// instead of recursing, so GPU threads need no large call stack.
struct DDFrame {
  uint64_t acc, tried;  // relevant cards so far, cards tried
  int8_t moves[13];
  int8_t n, i, best, result, ns_turn, seat, need, lb, ub;
  int8_t saved[4], old_leader, win_card;  // undo data when this ply ended a trick
};

// Enters the node at the current position. Returns 0/1 when it is decided at once (rel
// set), else -1 with `f` holding the ordered moves. `nd` keeps trick-start TT data.
DD_FN int dd_enter(DDCtx& c, int target, DDNode& nd, DDFrame& f, uint64_t* rel) {
  c.nodes++;
  DDTop t;
  int need = 0, hint = -1;
  f.lb = 0;
  f.ub = 0;
  if (c.nplayed == 0) {
    *rel = 0;
    if (c.ns_won >= target) return 1;
    if (c.ns_won + c.left < target) return 0;
    need = target - c.ns_won;
    if (c.left == 1) {
      // Last trick: highest trump, else highest card of the led suit.
      int s = c.trump >= 0 && ((c.hand[0] | c.hand[1] | c.hand[2] | c.hand[3]) &
                               DD_SUIT(c.trump))
                  ? c.trump
                  : dd_msb(c.hand[c.leader]) >> 4;
      uint64_t in = (c.hand[0] | c.hand[1] | c.hand[2] | c.hand[3]) & DD_SUIT(s);
      int top = dd_msb(in);
      if (dd_popc(in) >= 2) *rel = 1ull << top;
      int w = 0;
      while (!(c.hand[w] & (1ull << top))) w++;
      return (w & 1) == 0 ? 1 : 0;
    }
    dd_node_init(c, nd);
#ifdef DD_VERIFY
    int hit = c.no_tt ? -1 : dd_tt_probe(c, nd, need, rel, &hint);
    if (hit >= 0) {
      c.no_tt = 1;
      uint64_t vr;
      bool truth = dd_search(c, target, &vr);
      c.no_tt = 0;
      if (truth != (hit != 0)) {
        printf("BAD TT HIT: leader %d need %d left %d ns_won %d hit %d truth %d rel %llx\n",
               c.leader, need, c.left, c.ns_won, hit, truth, (unsigned long long)*rel);
        for (int h = 0; h < 4; h++) printf("  hand %d %llx\n", h, (unsigned long long)c.hand[h]);
      }
    }
#elif !defined(DD_NO_TT)
    int hit = dd_tt_probe(c, nd, need, rel, &hint);
    c.st_probe++; c.st_hit += hit >= 0;
#else
    int hit = -1;
#endif
    if (hit >= 0) return hit;
    f.ub = (int8_t)c.left;
    dd_top_init(t, c.hand);
    bool ns_lead = (c.leader & 1) == 0;
    int cut_ns = need, cut_ew = c.left - need + 1;
    uint64_t wr = 0;
    bool decided;
    int q = dd_quick_tricks(c.hand, t, c.trump, c.leader, ns_lead ? cut_ns : cut_ew,
                            ns_lead ? cut_ew : cut_ns, wr, decided);
#ifdef DD_NO_QT
    decided = false;
#endif
    if (decided) {
      bool val = ns_lead ? q != 0 : q == 0;
      c.st_qt++;
      dd_tt_store(c, nd, wr, val ? need : 0, val ? c.left : need - 1, -1);
      *rel = wr;
      return val ? 1 : 0;
    }
    wr = 0;
#ifdef DD_NO_LT
    if (false)
#else
    if (ns_lead ? !dd_later_min(t, c.trump, c.leader, c.ns_won, c.left, target, wr)
                : dd_later_max(t, c.trump, c.leader, c.ns_won, c.left, target, wr))
#endif
    {
      bool val = !ns_lead;
      dd_tt_store(c, nd, wr, val ? need : 0, val ? c.left : need - 1, -1);
      *rel = wr;
      return val ? 1 : 0;
    }
  }
#ifndef DD_NO_QT2
  else if (c.nplayed == 1) {
    uint64_t th[4] = {c.hand[0], c.hand[1], c.hand[2], c.hand[3]};
    th[c.leader] |= 1ull << c.trick[0];
    dd_top_init(t, th);
    int second = (c.leader + 1) & 3;
    int need2 = target - c.ns_won;
    int cutoff = (second & 1) == 0 ? need2 : c.left - need2 + 1;
    uint64_t wr;
    if (dd_quick_tricks_2nd(c.hand, t, c.trump, second, c.trick[0], cutoff, wr)) {
      *rel = wr;
      return (second & 1) == 0 ? 1 : 0;
    }
  }
#endif

  int seat = (c.leader + c.nplayed) & 3;
  int moves[13];
  f.n = (int8_t)dd_moves(c, seat, hint, &t, moves);
  for (int k = 0; k < f.n; k++) f.moves[k] = (int8_t)moves[k];
  f.seat = (int8_t)seat;
  f.ns_turn = (seat & 1) == 0;
  f.result = !f.ns_turn;
  f.need = (int8_t)need;
  f.i = 0;
  f.best = -1;
  f.acc = 0;
  f.tried = 0;
  return -1;
}

DD_FN void dd_play(DDCtx& c, DDFrame& f, int card) {
  c.hand[f.seat] ^= 1ull << card;
  c.trick[c.nplayed++] = card;
  if (c.nplayed == 4) {
    c.st_tricks++;
    int wi = dd_winning(c, 4);
    int w = (c.leader + wi) & 3;
    int wc = c.trick[wi], same = 0;
    for (int k = 0; k < 4; k++) same += (c.trick[k] >> 4) == (wc >> 4);
    for (int k = 0; k < 4; k++) f.saved[k] = (int8_t)c.trick[k];
    f.old_leader = (int8_t)c.leader;
    f.win_card = (int8_t)(same >= 2 ? wc : -1);  // won by rank
    c.leader = w;
    c.nplayed = 0;
    c.left--;
    c.ns_won += (w & 1) == 0;
  }
}

DD_FN void dd_unplay(DDCtx& c, DDFrame& f, int card, uint64_t* rel) {
  if (c.nplayed == 0) {  // this ply ended a trick
    for (int k = 0; k < 4; k++) c.trick[k] = f.saved[k];
    c.ns_won -= (c.leader & 1) == 0;
    c.left++;
    c.nplayed = 4;
    c.leader = f.old_leader;
    if (f.win_card >= 0) *rel |= 1ull << f.win_card;
  }
  c.nplayed--;
  c.hand[f.seat] ^= 1ull << card;
}

// All moves of `f` are done: returns its relevant cards and stores trick-start results.
DD_FN uint64_t dd_finish(DDCtx& c, DDFrame& f, const DDNode& nd) {
  uint64_t acc = f.acc;
  // Only the top card of a run of touching cards was tried. If the lowest relevant card
  // of a suit is in such a run, the rest of the run must stay relevant too, else a TT
  // match could pair positions where the run is broken by another hand's card.
  uint64_t own = c.hand[f.seat], all = c.hand[0] | c.hand[1] | c.hand[2] | c.hand[3];
  for (int i = 0; i < c.nplayed; i++) all |= 1ull << c.trick[i];
  for (int s = 0; s < 4; s++) {
    uint64_t r = acc & DD_SUIT(s);
    if (!r) continue;
    int low = dd_lsb(r);
    while (own >> low & 1) {
      uint64_t below = all & DD_SUIT(s) & ((1ull << low) - 1);
      if (!below || !(own >> dd_msb(below) & 1)) break;
      low = dd_msb(below);
      acc |= 1ull << low;
    }
  }
  if (c.nplayed == 0) {
    int lb = f.lb, ub = f.ub;
    if (f.result) lb = f.need;
    else ub = f.need - 1;
    dd_tt_store(c, nd, acc, lb, ub, f.best);
  }
  return acc;
}

// Can NS take >= target tricks in total (counting ns_won)? `rel` gets the cards whose
// rank the answer depends on. Depth-first search as a loop over explicit frames.
DD_REC bool dd_search(DDCtx& c, int target, uint64_t* rel) {
  DDFrame fs[53];
  DDNode nds[14];  // by tricks left; one trick-start node per level is live at a time
  int sp = 0;
  bool v;
  uint64_t r;
  int e = dd_enter(c, target, nds[c.left], fs[0], &r);
  if (e >= 0) {
    *rel = r;
    return e != 0;
  }
next_move : {
  DDFrame& f = fs[sp];
  while (f.i < f.n) {
    int card = f.moves[f.i];
    int s = card >> 4;
    // Small cards (below every relevant card of the suit) are all alike: try one.
    uint64_t rs = f.acc & DD_SUIT(s);
    uint64_t small = rs ? ((1ull << dd_lsb(rs)) - 1) & DD_SUIT(s) : DD_SUIT(s);
#ifndef DD_NO_SMALL
    if ((small & (1ull << card)) && (small & f.tried)) {
      f.i++;
      continue;
    }
#endif
    f.tried |= 1ull << card;
    dd_play(c, f, card);
    int ce = dd_enter(c, target, nds[c.left], fs[sp + 1], &r);
    if (ce < 0) {
      sp++;
      goto next_move;
    }
    v = ce != 0;
    goto child_done;
  }
  v = f.result;
  r = dd_finish(c, f, nds[c.left]);
  goto node_done;
}
child_done : {  // the child of fs[sp] returned (v, r)
  DDFrame& f = fs[sp];
  int card = f.moves[f.i];
  dd_unplay(c, f, card, &r);
  if (v == (f.ns_turn != 0)) {
    f.result = v;
    f.acc = r;
    f.best = (int8_t)card;
    r = dd_finish(c, f, nds[c.left]);
    goto node_done;
  }
  f.acc |= r;
  f.i++;
  goto next_move;
}
node_done:  // fs[sp] is done with (v, r)
  if (sp == 0) {
    *rel = r;
    return v;
  }
  sp--;
  goto child_done;
}

DD_FN void dd_next_gen(DDCtx& c) {
#ifdef DD_SHARED_TT
  return;  // entries stay valid across deals
#endif
  c.gen++;
  if (c.gen >= (1u << DD_GEN_BITS)) {
    for (uint32_t i = 0; i <= c.tt_mask; i++)
      for (int w = 0; w < DD_TT_WAYS; w++) c.tt[i].tag[w] = 0;
    c.gen = 1;
  }
}

// NS tricks with `leader` on lead from the full deal.
DD_FN int dd_ns_tricks(DDCtx& c, int leader, int guess) {
  c.leader = leader;
  c.nplayed = 0;
  c.ns_won = 0;
  c.left = dd_popc(c.hand[0]);
  int lo = 0, hi = 13;
  int t = guess < 1 ? 1 : guess > 13 ? 13 : guess;
  uint64_t rel;
  while (lo < hi) {
    if (dd_search(c, t, &rel)) lo = t;
    else hi = t - 1;
    t = (lo + hi + 1) / 2;
  }
  return lo;
}

DD_FN int dd_strain_suit(int strain) { return strain == 4 ? -1 : 3 - strain; }

// Tricks for each declarer N,E,S,W in one strain. The TT is shared by the four leads.
// NS tricks with `leader` on lead, stepping from a guess: a right guess costs two
// searches ("at least g" yes, "at least g+1" no).
DD_FN int dd_ns_tricks_from(DDCtx& c, int leader, int guess) {
  c.leader = leader;
  c.nplayed = 0;
  c.ns_won = 0;
  c.left = dd_popc(c.hand[0]);
  int t = guess < 0 ? 0 : guess > c.left ? c.left : guess;
  uint64_t rel;
  if (t == 0 || dd_search(c, t, &rel)) {
    while (t < c.left && dd_search(c, t + 1, &rel)) t++;
    return t;
  }
  t--;
  while (t > 0 && !dd_search(c, t, &rel)) t--;
  return t;
}

// Like dd_solve_strain, with a guess of NS tricks per declarer (e.g. from a net).
DD_FN void dd_solve_strain_guess(DDCtx& c, const uint64_t* hands, int strain, const int* ns4,
                                 int* out4) {
  for (int i = 0; i < 4; i++) c.hand[i] = hands[i];
  c.trump = dd_strain_suit(strain);
  dd_next_gen(c);
  for (int decl = 0; decl < 4; decl++) {
    int ns = dd_ns_tricks_from(c, (decl + 1) & 3, ns4[decl]);
    out4[decl] = (decl & 1) == 0 ? ns : 13 - ns;
  }
}

DD_FN void dd_solve_strain(DDCtx& c, const uint64_t* hands, int strain, int* out4) {
  for (int i = 0; i < 4; i++) c.hand[i] = hands[i];
  c.trump = dd_strain_suit(strain);
  dd_next_gen(c);
  int guess = 7;
  for (int decl = 0; decl < 4; decl++) {
#ifdef DD_PER_LEADER_TT
    if (decl) dd_next_gen(c);  // measure: no TT sharing between leads
#endif
    int ns = dd_ns_tricks(c, (decl + 1) & 3, guess);
    guess = ns;
    out4[decl] = (decl & 1) == 0 ? ns : 13 - ns;
  }
}
