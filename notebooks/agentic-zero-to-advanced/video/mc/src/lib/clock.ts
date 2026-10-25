import {waitFor, useThread} from '@motion-canvas/core';

export type Beat = {key: string; text: string; t0: number; dur: number; times: number[]};
export type Timeline = {beats: Beat[]; end: number};

/** Absolute spoken time of word `idx` (0-based, counted over the displayed words) of sentence `key`. */
export function T(tl: Timeline, key: string, idx = 0): number {
  const b = tl.beats.find(x => x.key === key);
  if (!b) throw new Error(`no sentence ${key}`);
  if (idx < 0 || idx >= b.times.length) throw new Error(`${key} has ${b.times.length} words, asked for ${idx}`);
  return b.t0 + b.times[idx];
}

/** Index of a word by text (first match, punctuation/case-insensitive), so scenes say `W(tl,'R3','nbdev-new')` instead of counting. */
export function W(tl: Timeline, key: string, word: string): number {
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
  const b = tl.beats.find(x => x.key === key)!;
  const i = b.text.replace(/\*\*/g, '').split(/\s+/).findIndex(w => norm(w) === norm(word));
  if (i < 0) throw new Error(`word ${word} not in ${key}`);
  return i;
}

/** A clock for one thread: `yield* at(t)` waits until absolute time t (does nothing if it has passed). Reads the thread's own time, so the
 *  one-frame rounding of each waitFor cannot add up over a chapter (an earlier version counted `t - last t` and drifted about 2 s over 50 events). */
export function clock() {
  return function* at(t: number) {
    const d = t - useThread().time();
    if (d > 0) yield* waitFor(d);
  };
}

/** Typing/animation time that fits: min(max, time left in the sentence after word `idx`) so nothing is cut off when the voice is fast. */
export function D(tl: Timeline, key: string, idx: number, max: number): number {
  const b = tl.beats.find(x => x.key === key)!;
  return Math.max(0.25, Math.min(max, b.t0 + b.dur - (b.t0 + b.times[idx]) - 0.3));
}
