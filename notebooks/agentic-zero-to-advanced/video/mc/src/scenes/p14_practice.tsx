import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {createRef, Reference} from '@motion-canvas/core';
import tl from '../p14_practice/timeline.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));

// [practice, where it meets our work, colour]
const ROWS: [string, string, string][] = [
  ['check: does it fit?', 'Part 5: the message list is re-read at every step', GREEN],
  ['cheapest tool first', 'Part 8: code, then a small model, then a language model', GREEN],
  ['paths only if results are big', 'visual search: 20 small records, the plain payload won', GREEN],
  ['narrow brief for sub-agents', 'Part 7: code owns the order, the model owns the judgment', GREEN],
  ['look at individual answers', 'Part 8: the token-probability bug', GREEN],
  ['not tested: skills, injection, stale files', 'later: the Deep Agents and security posts', AMBER],
];
const Y0 = -430, DY = 120;

export default makeScene2D(function* (view) {
  view.fill(BG);
  const chips: Reference<Rect>[] = ROWS.map(() => createRef<Rect>());
  const cards: Reference<Rect>[] = ROWS.map(() => createRef<Rect>());
  ROWS.forEach(([a, b, col], i) => {
    view.add(
      <Rect ref={chips[i]} x={-460} y={Y0 + i * DY} width={800} height={100} radius={20} fill={PANEL} stroke={col} lineWidth={6} opacity={0}
            layout alignItems="center" justifyContent="center">
        <Txt text={a} fontFamily={SANS} fontSize={36} fontWeight={700} fill="#fff"/>
      </Rect>);
    view.add(
      <Rect ref={cards[i]} x={500} y={Y0 + i * DY} width={860} height={100} radius={20} fill={PANEL} stroke={GREY} lineWidth={3} opacity={0}
            layout alignItems="center" justifyContent="center" padding={[0, 24]}>
        <Txt text={b} fontFamily={SANS} fontSize={26} fill={ACCENT} textAlign="center"/>
      </Rect>);
  });
  view.add(<Txt text="connects to our work" x={500} y={Y0 - 70} fontFamily={SANS} fontSize={30} fill={GREY}/>);

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(at('P1', 'parts'));
  yield* go(at('P2', 'check')); yield chips[0]().opacity(1, 0.5);
  yield* go(at('P2', 'pasting')); yield cards[0]().opacity(1, 0.5);
  yield* go(at('P3', 'cheapest')); yield chips[1]().opacity(1, 0.5);
  yield* go(at('P3', 'eight')); yield cards[1]().opacity(1, 0.5);
  yield* go(at('P4', 'paths')); yield chips[2]().opacity(1, 0.5);
  yield* go(at('P4', 'big')); yield cards[2]().opacity(1, 0.5);
  yield* go(at('P4', 'narrow')); yield chips[3]().opacity(1, 0.5);
  yield* go(at('P4', "seven's")); yield cards[3]().opacity(1, 0.5);
  yield* go(at('P5', 'look')); yield chips[4]().opacity(1, 0.5);
  yield* go(at('P5', 'eight')); yield cards[4]().opacity(1, 0.5);
  yield* go(at('P6', 'skills')); yield chips[5]().opacity(1, 0.5);
  yield* go(at('P6', 'later')); yield cards[5]().opacity(1, 0.5);
  yield* go(TL.end);
});
