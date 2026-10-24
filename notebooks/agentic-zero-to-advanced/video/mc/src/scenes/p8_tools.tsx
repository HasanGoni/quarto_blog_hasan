import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {createRef, all, Reference} from '@motion-canvas/core';
import tl from '../p8_tools/timeline.json';
import facts from '../facts.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));   // absolute spoken time of a word
const c = facts.costs;
const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);

type Row = [string, string];
function kindCard(x: number, color: string, title: string, sub: string, rows: Row[]) {
  const root = createRef<Rect>();
  const vals: Reference<Txt>[] = rows.map(() => createRef<Txt>());
  const node = (
    <Rect ref={root} x={x} y={-130} width={580} height={600} radius={26} fill={PANEL} stroke={color} lineWidth={6} opacity={0}
          layout direction="column" padding={34} gap={18} alignItems="center">
      <Txt text={title} fontFamily={SANS} fontSize={50} fontWeight={700} fill="#fff"/>
      <Txt text={sub} fontFamily={SANS} fontSize={30} fill={color}/>
      {rows.map(([k, v], i) => (
        <Rect layout direction="column" width={500} gap={4}>
          <Txt text={k} fontFamily={SANS} fontSize={26} fill={GREY}/>
          <Txt ref={vals[i]} text={v} fontFamily={SANS} fontSize={38} fontWeight={700} fill="#fff" opacity={0}/>
        </Rect>))}
    </Rect>);
  return {node, root, vals};
}

export default makeScene2D(function* (view) {
  view.fill(BG);
  const code = kindCard(-620, GREEN, 'plain code', 'a rule you wrote', [['it gives you', 'an exact answer'], ['one call', `about ${c.rule_ns} ns`], ['tokens', '0']]);
  const typed = kindCard(0, ACCENT, 'typed model', 'small, reads text', [['it gives you', 'a typed answer\n+ a probability'], ['one call', `${c.laya_gpu_ms} ms on a GPU\n${c.laya_cpu_ms} ms on a CPU`], ['tokens', '0']]);
  const llm = kindCard(620, AMBER, 'language model', 'writes text', [['it gives you', 'words, about\nanything'], ['one note', `${fmtMs(c.llm_note_ms)}`], ['tokens', `${c.llm_note_tokens} per note`]]);
  view.add(code.node); view.add(typed.node); view.add(llm.node);

  // the six pipeline steps, shown at the end: which tool does each one need?
  const steps: [string, string][] = [['detect', GREEN], ['measure', GREEN], ['classify', '#ffffff'], ['rules', GREEN], ['decide', GREEN], ['report', AMBER]];
  const chips: Reference<Rect>[] = steps.map(() => createRef<Rect>());
  const marks: Reference<Txt>[] = steps.map(() => createRef<Txt>());
  const row = (
    <Rect y={250} layout gap={22} opacity={0} ref={createRef<Rect>()}>
      {steps.map(([name, col], i) => (
        <Rect ref={chips[i]} width={270} height={96} radius={18} stroke={col} lineWidth={5} fill={PANEL} layout direction="column" alignItems="center" justifyContent="center">
          <Txt text={name} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
          <Txt ref={marks[i]} text={name === 'classify' ? 'which tool?' : name === 'report' ? 'language model' : 'plain code'} fontFamily={SANS} fontSize={26} fill={col} opacity={0}/>
        </Rect>))}
    </Rect>);
  view.add(row);
  const rowRef = row;

  yield addSubtitles(view, TL);
  const go = clock();

  // T1: code and model
  yield* go(at('T1', 'code')); yield code.root().opacity(1, 0.6);
  yield* go(at('T1', 'model')); yield llm.root().opacity(1, 0.6);
  // T2: the third kind arrives in the middle
  yield* go(at('T2', 'typed')); typed.root().y(-90); yield all(typed.root().opacity(1, 0.7), typed.root().y(-130, 0.7));
  // T3: code numbers
  yield* go(at('T3', 'forty')); yield code.vals[1]().opacity(1, 0.4);
  yield* go(at('T3', 'nothing')); yield code.vals[2]().opacity(1, 0.4); yield code.vals[0]().opacity(1, 0.4);
  // T4: typed model numbers
  yield* go(at('T4', 'twenty')); yield typed.vals[1]().opacity(1, 0.4); yield typed.vals[0]().opacity(1, 0.4);
  yield* go(at('T4', 'tokens')); yield typed.vals[2]().opacity(1, 0.4);
  // T5: the language model
  yield* go(at('T5', 'write')); yield llm.vals[0]().opacity(1, 0.4);
  yield* go(at('T5', 'two')); yield llm.vals[1]().opacity(1, 0.4);
  yield* go(at('T5', 'tokens')); yield llm.vals[2]().opacity(1, 0.4);
  // T6: which tool does THIS step need?
  yield* go(at('T6', 'question')); yield all(code.root().opacity(0.18, 0.6), typed.root().opacity(0.18, 0.6), llm.root().opacity(0.18, 0.6));
  yield rowRef.opacity(1, 0.6);
  yield* go(at('T6', 'step'));
  for (let i = 0; i < steps.length; i++) { yield marks[i]().opacity(1, 0.25); yield* go(at('T6', 'step') + 0.25 * (i + 1)); }
  yield* go(TL.end);
});
