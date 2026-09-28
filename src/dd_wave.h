// The whole solve as a resumable state machine, so a GPU warp can run one stage at a time
// for all of its threads (wavefront style) instead of each thread following its own path.
// Same search as dd_search / dd_solve_strain; dd_wave_step does one stage for one thread.
#pragma once
#include "dd.h"

enum {
  DD_S_JOB,    // take the next (deal, strain) job
  DD_S_ROOT,   // start the null-window search for the current lead and target
  DD_S_ENTER,  // enter a position: bounds, TT probe, move list (the expensive stage)
  DD_S_NEXT,   // play the next move of the top frame
  DD_S_RET,    // a child returned: undo its move, cut or continue
  DD_S_EXIT,   // no jobs left
  DD_S_COUNT
};

struct DDWave {
  DDCtx c;
  DDFrame fs[53];
  DDNode nds[14];
  const uint64_t* hands;  // all deals, 4 masks each
  int* out;               // 20 results per deal
  long jobs, job;
  int stage, sp, strain, decl, lo, hi, target;
  bool v;
  uint64_t r;
};

DD_FN void dd_wave_start_search(DDWave& w) {
  DDCtx& c = w.c;
  c.leader = (w.decl + 1) & 3;
  c.nplayed = 0;
  c.ns_won = 0;
  c.left = dd_popc(c.hand[0]);
  w.sp = 0;
  w.stage = DD_S_ENTER;
}

// A search node finished with (v, r): continue in its parent, or finish the root search.
DD_FN void dd_wave_node_done(DDWave& w) {
  if (w.sp > 0) {
    w.sp--;
    w.stage = DD_S_RET;
    return;
  }
  // Root: next target of the binary search, next declarer, or next job.
  if (w.v) w.lo = w.target;
  else w.hi = w.target - 1;
  if (w.lo < w.hi) {
    w.target = (w.lo + w.hi + 1) / 2;
    dd_wave_start_search(w);
    return;
  }
  int ns = w.lo;
  w.out[w.job / 5 * 20 + w.strain * 4 + w.decl] = (w.decl & 1) == 0 ? ns : 13 - ns;
  w.decl++;
  if (w.decl < 4) {
    w.stage = DD_S_ROOT;
    w.target = ns < 1 ? 1 : ns;  // start the next lead's binary search at this answer
    return;
  }
  w.stage = DD_S_JOB;
}

// `next_job` returns the next job index (atomic counter on the GPU).
template <class NextJob>
DD_FN void dd_wave_step(DDWave& w, NextJob next_job) {
  DDCtx& c = w.c;
  switch (w.stage) {
    case DD_S_JOB: {
      w.job = next_job();
      if (w.job >= w.jobs) {
        w.stage = DD_S_EXIT;
        return;
      }
      for (int i = 0; i < 4; i++) c.hand[i] = w.hands[w.job / 5 * 4 + i];
      w.strain = (int)(w.job % 5);
      c.trump = dd_strain_suit(w.strain);
      dd_next_gen(c);
      w.decl = 0;
      w.target = 7;
      w.stage = DD_S_ROOT;
      return;
    }
    case DD_S_ROOT: {
      w.lo = 0;
      w.hi = 13;
      dd_wave_start_search(w);
      return;
    }
    case DD_S_ENTER: {
      int e = dd_enter(c, w.target, w.nds[c.left], w.fs[w.sp], &w.r);
      if (e < 0) {
        w.stage = DD_S_NEXT;
        return;
      }
      w.v = e != 0;
      dd_wave_node_done(w);
      return;
    }
    case DD_S_NEXT: {
      DDFrame& f = w.fs[w.sp];
      while (f.i < f.n) {
        int card = f.moves[f.i];
        int s = card >> 4;
        uint64_t rs = f.acc & DD_SUIT(s);
        uint64_t small = rs ? ((1ull << dd_lsb(rs)) - 1) & DD_SUIT(s) : DD_SUIT(s);
        if ((small & (1ull << card)) && (small & f.tried)) {
          f.i++;
          continue;
        }
        f.tried |= 1ull << card;
        dd_play(c, f, card);
        w.sp++;
        w.stage = DD_S_ENTER;
        return;
      }
      w.v = f.result;
      w.r = dd_finish(c, f, w.nds[c.left]);
      dd_wave_node_done(w);
      return;
    }
    case DD_S_RET: {
      DDFrame& f = w.fs[w.sp];
      int card = f.moves[f.i];
      dd_unplay(c, f, card, &w.r);
      if (w.v == (f.ns_turn != 0)) {
        f.result = w.v;
        f.acc = w.r;
        f.best = (int8_t)card;
        w.r = dd_finish(c, f, w.nds[c.left]);
        dd_wave_node_done(w);
        return;
      }
      f.acc |= w.r;
      f.i++;
      w.stage = DD_S_NEXT;
      return;
    }
  }
}
