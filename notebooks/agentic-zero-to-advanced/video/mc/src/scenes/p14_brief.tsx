import {makeScene2D, Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef} from '@motion-canvas/core';
import tl from '../p14_brief/timeline.json';
import facts from '../facts_p14.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const fmt = (n: number) => n.toLocaleString('en-US');
const TOK_SCALE = 1000 / facts.t6.pad100;       // px per token
const t6n = facts.t6.narrow, t6p = facts.t6.pad100;

function bar(y: number, color: string, label: string, valueText: string, widthPx: number) {
  const r = createRef<Rect>(), v = createRef<Txt>();
  const node = (
    <Node y={y}>
      <Txt text={label} x={-880} y={-46} offsetX={-1} fontFamily={SANS} fontSize={28} fill={GREY}/>
      <Rect ref={r} x={-880} offsetX={-1} width={0} height={52} radius={8} fill={color}/>
      <Txt ref={v} text={valueText} offsetX={-1} x={-880 + widthPx + 24} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff" opacity={0}/>
    </Node>);
  return {node, r, v, widthPx};
}

export default makeScene2D(function* (view) {
  view.fill(BG);
  const narrow = bar(-410, GREEN, 'brief: the one report', `${fmt(t6n)} tokens`, Math.max(8, t6n * TOK_SCALE));
  const padded = bar(-300, AMBER, 'brief: that report + 99 others', `${fmt(t6p)} tokens`, t6p * TOK_SCALE);
  const near = bar(-110, ACCENT, 'DINOv3: nearest 4 share the part class', `${(facts.dino.near * 100).toFixed(1)}%`, facts.dino.near * 1000);
  const rand = bar(0, GREY, 'random neighbours', `${(facts.dino.random * 100).toFixed(1)}%`, facts.dino.random * 1000);
  [narrow, padded, near, rand].forEach(b => view.add(b.node));

  const okRef = createRef<Rect>(), xRef = createRef<Rect>(), payRef = createRef<Rect>();
  view.add(
    <Rect ref={okRef} x={700} y={-405} width={460} height={80} radius={16} fill={PANEL} stroke={GREEN} lineWidth={5} opacity={0} layout alignItems="center" justifyContent="center">
      <Txt text={`${facts.t6.correct} / ${facts.t6.runs} exact`} fontFamily={SANS} fontSize={40} fontWeight={700} fill={GREEN}/>
    </Rect>);
  view.add(
    <Rect ref={xRef} x={700} y={-300} width={460} height={80} radius={16} fill={PANEL} stroke={AMBER} lineWidth={5} opacity={0} layout alignItems="center" justifyContent="center">
      <Txt text={`${Math.round(t6p / t6n)}x fewer tokens`} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
    </Rect>);
  view.add(
    <Rect ref={payRef} y={170} width={1500} height={150} radius={22} fill={PANEL} stroke={ACCENT} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={10}>
      <Txt text="20 small records (visual search task)" fontFamily={SANS} fontSize={30} fill={GREY}/>
      <Txt text={`plain payload ${fmt(facts.t5.payload)} tokens   |   full harness ${fmt(facts.t5.harness)} tokens`} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
    </Rect>);

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(at('B1', 'sub-agent')); yield narrow.r().width(Math.max(8, t6n * TOK_SCALE), 0.6);
  yield* go(at('B1', 'sixty-four')); yield narrow.v().opacity(1, 0.4);
  yield* go(at('B1', 'ninety-nine')); yield padded.r().width(t6p * TOK_SCALE, 1.2);
  yield* go(at('B1', 'thousand')); yield padded.v().opacity(1, 0.4);
  yield* go(at('B2', 'eighty-four')); yield okRef().opacity(1, 0.5);
  yield* go(at('B2', 'thirty-eight')); yield xRef().opacity(1, 0.5);
  yield* go(at('B3', 'visual')); yield near.r().width(near.widthPx, 0.9);
  yield* go(at('B3', 'seventy-three')); yield near.v().opacity(1, 0.4);
  yield* go(at('B3', 'random')); yield rand.r().width(rand.widthPx, 0.6);
  yield* go(at('B3', 'ten')); yield rand.v().opacity(1, 0.4);
  yield* go(at('B4', 'payload')); yield payRef().opacity(1, 0.5);
  yield* go(TL.end);
});
