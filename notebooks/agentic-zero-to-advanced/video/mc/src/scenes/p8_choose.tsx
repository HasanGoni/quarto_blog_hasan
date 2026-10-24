import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {createRef, all, Reference} from '@motion-canvas/core';
import tl from '../p8_choose/timeline.json';
import facts from '../facts.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, MONO, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const f = facts;
const pc = (x: number) => `${Math.round(x * 100)}%`;
const LANE_Y = -90, LANE_H = 600, LANE_W = 590;

function lane(x: number, color: string, ifText: string, tool: string) {
  const root = createRef<Rect>(), ev = createRef<Rect>();
  const node = (
    <Rect ref={root} x={x} y={LANE_Y} width={LANE_W} height={LANE_H} radius={26} fill={PANEL} stroke={color} lineWidth={6} opacity={0}>
      <Txt text="if the step is" fontFamily={SANS} fontSize={28} fill={GREY} y={-245}/>
      <Txt text={ifText} fontFamily={SANS} fontSize={56} fontWeight={700} fill="#fff" y={-190}/>
      <Txt text={`use ${tool}`} fontFamily={SANS} fontSize={36} fontWeight={700} fill={color} y={-120}/>
      <Rect ref={ev} y={95} width={LANE_W - 60} height={380} opacity={0} layout direction="column" gap={14} alignItems="center"/>
    </Rect>);
  return {node, root, ev};
}

export default makeScene2D(function* (view) {
  view.fill(BG);
  const L1 = lane(-640, GREEN, 'numbers', 'a rule'), L2 = lane(0, ACCENT, 'reading text', 'a small typed model'), L3 = lane(640, AMBER, 'writing', 'the language model');
  view.add(L1.node); view.add(L2.node); view.add(L3.node);

  // lane 1: ranking score on the held-out boxes, three judges
  const auc = f.auc;
  const bars = [['rule', auc['band'].guided, GREEN], ['LLM (number)', auc['llm-verbalized'].guided, AMBER], ['typed', auc['laya-english'].guided, ACCENT]] as [string, number, string][];
  const barRefs: Reference<Rect>[] = bars.map(() => createRef<Rect>());
  L1.ev().add(<Txt text="ranking score, held-out boxes" fontFamily={SANS} fontSize={30} fill={GREY}/>);
  bars.forEach(([name, v, col], i) => L1.ev().add(
    <Rect layout gap={12} alignItems="center" width={520}>
      <Txt text={name} fontFamily={SANS} fontSize={28} fill="#fff" width={250}/>
      <Rect ref={barRefs[i]} width={0} height={38} radius={8} fill={col}/>
      <Txt text={v.toFixed(2)} fontFamily={SANS} fontSize={36} fontWeight={700} fill="#fff"/>
    </Rect>));
  L1.ev().add(<Txt text={`the rule: ${f.costs.rule_ns} ns, 0 tokens`} fontFamily={SANS} fontSize={32} fill={GREEN} fontWeight={700}/>);

  // lane 2: reading the real notes
  const names: [string, string][] = [['typed', 'laya-english'], ['LLM (number)', 'llm-verbalized'], ['LLM (token)', 'llm-logprob']];
  const countCell: Reference<Txt> = createRef<Txt>();
  L2.ev().add(<Txt text="reading 520 real notes, accuracy" fontFamily={SANS} fontSize={28} fill={GREY}/>);
  L2.ev().add(
    <Rect layout gap={8} width={520}>
      <Txt text="" width={196}/>
      {['escalated?', 'uncertain?', 'count'].map(h => <Txt text={h} fontFamily={SANS} fontSize={22} fill={GREY} width={104}/>)}
    </Rect>);
  names.forEach(([n, key], r) => {
    const a = (f.notes_acc as any)[key];
    L2.ev().add(
      <Rect layout gap={8} width={520} alignItems="center">
        <Txt text={n} fontFamily={SANS} fontSize={27} fill="#fff" width={196}/>
        {[a.escalate, a.uncertain, a.n_kept].map((v: number, c: number) => (
          <Txt ref={r === 0 && c === 2 ? countCell : undefined} text={pc(v)} fontFamily={SANS} fontSize={34} fontWeight={700} fill={r === 0 ? ACCENT : '#fff'} width={104}/>))}
      </Rect>);
  });
  L2.ev().add(<Txt text={`typed model: ${f.costs.laya_gpu_ms} ms, 0 tokens`} fontFamily={SANS} fontSize={32} fill={ACCENT} fontWeight={700}/>);

  // lane 3: writing
  L3.ev().add(<Txt text="one note, measured" fontFamily={SANS} fontSize={28} fill={GREY}/>);
  L3.ev().add(<Txt text={`${(f.costs.llm_note_ms / 1000).toFixed(1)} s`} fontFamily={SANS} fontSize={84} fontWeight={700} fill="#fff"/>);
  L3.ev().add(<Txt text={`${f.costs.llm_note_tokens} tokens`} fontFamily={SANS} fontSize={48} fontWeight={700} fill="#fff"/>);
  L3.ev().add(<Txt text={'code owns the facts,\nthe model only writes the words'} fontFamily={SANS} fontSize={30} fill={AMBER} fontWeight={700} textAlign="center"/>);

  const warn = (
    <Rect y={252} width={1720} height={78} radius={16} fill={PANEL} stroke={AMBER} lineWidth={5} opacity={0} ref={createRef<Rect>()}>
      <Txt text={`typed model + guidance: calibration error ${f.laya_guided.ece.toFixed(3)}, yet it flagged ${f.laya_guided['flagged_at_0.5']} boxes at 50%`}
           fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
    </Rect>);
  view.add(warn);

  yield addSubtitles(view, TL);
  const go = clock();

  // C1 the three kinds of step
  yield* go(at('C1', 'which'));
  yield all(L1.root().opacity(1, 0.6), L2.root().opacity(1, 0.6), L3.root().opacity(1, 0.6));
  // C2 numbers
  yield* go(at('C2', 'numbers')); yield L1.ev().opacity(1, 0.5);
  yield all(...barRefs.map((r, i) => r().width(250 * bars[i][1], 0.8)));
  yield* go(at('C2', 'beat')); yield all(barRefs[0]().scale(1.0, 0.01), L1.root().scale(1.04, 0.4));
  yield* go(at('C2', 'cost')); yield L1.root().scale(1, 0.4);
  // C3 text
  yield* go(at('C3', 'text')); yield L2.ev().opacity(1, 0.5);
  yield* go(at('C3', 'counting')); yield all(countCell().fill(AMBER, 0.4), countCell().scale(1.3, 0.4));
  // C4 writing
  yield* go(at('C4', 'write')); yield L3.ev().opacity(1, 0.5);
  // C5 measure, and beware flattering metrics
  yield* go(at("C5", "measure")); yield warn.opacity(1, 0.6);
  yield* go(TL.end);
});
