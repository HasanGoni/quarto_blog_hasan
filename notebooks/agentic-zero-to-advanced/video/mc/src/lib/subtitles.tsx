import {Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef, Reference} from '@motion-canvas/core';
import {Timeline, clock} from './clock';
import {YELLOW, SANS, SUB_Y, SUB_W} from './theme';

type Word = {w: string; mark: boolean; t: number};

/** displayed words of a sentence; `**marked**` words get the highlighter; punctuation stays attached to its word */
function words(text: string, times: number[], t0: number): Word[] {
  let on = false; const out: Word[] = [];
  for (const tok of text.split(/\s+/).filter(Boolean)) {
    const opens = tok.startsWith('**'); if (opens) on = true;
    out.push({w: tok.replace(/\*\*/g, ''), mark: on, t: t0 + times[out.length]});
    if (/\*\*[^\w*]*$/.test(tok) && (!opens || (tok.match(/\*\*/g) || []).length >= 2)) on = false;
  }
  return out;
}

/** Sentence captions like the Manim chapters: the whole sentence's layout is fixed up front, words appear as they are spoken,
 *  `**marked**` words get the yellow highlighter. Start with `yield addSubtitles(view, tl)` so it runs in parallel with the visuals. */
export function addSubtitles(view: Node, tl: Timeline) {
  const sents = tl.beats.map(b => words(b.text, b.times, b.t0));
  const boxes: Reference<Rect>[] = sents.map(() => createRef<Rect>());
  const wr: Reference<Rect>[][] = sents.map(s => s.map(() => createRef<Rect>()));
  const tx: Reference<Txt>[][] = sents.map(s => s.map(() => createRef<Txt>()));
  sents.forEach((s, i) => view.add(
    <Rect ref={boxes[i]} y={SUB_Y} width={SUB_W} layout wrap="wrap" gap={[6, 16]} padding={[14, 30]} fill="#0d0d0dEE" radius={14} opacity={0}
          justifyContent="center" alignItems="center">
      {s.map((x, j) => (
        <Rect ref={wr[i][j]} padding={[2, 10]} radius={8}>
          <Txt ref={tx[i][j]} text={x.w} fontFamily={SANS} fontSize={50} fill="#fff" opacity={0}/>
        </Rect>))}
    </Rect>));
  const at = clock();
  const run = function* () {
    for (let i = 0; i < sents.length; i++) {
      for (let j = 0; j < sents[i].length; j++) {
        yield* at(sents[i][j].t);
        if (j === 0) { if (i > 0) boxes[i - 1]().opacity(0); boxes[i]().opacity(1); }
        tx[i][j]().opacity(1);
        if (sents[i][j].mark) { wr[i][j]().fill(YELLOW); tx[i][j]().fill('#111'); }
      }
      yield* at(tl.beats[i].t0 + tl.beats[i].dur);
      boxes[i]().opacity(0);
    }
  };
  return run();
}
