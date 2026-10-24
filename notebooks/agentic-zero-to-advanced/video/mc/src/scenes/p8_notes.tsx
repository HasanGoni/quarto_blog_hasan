import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {createRef, all, delay, Reference} from '@motion-canvas/core';
import tl from '../p8_notes/timeline.json';
import facts from '../facts.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, RED, GREY, SANS, MONO, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const g = facts.guard, ok = facts.example_ok, bad = facts.example_bad, am = facts.example_ambiguous;
const kind = (i: number) => g.flag_idx.laya_called_a_fine_note_uncertain.includes(i) ? 'wrongU' : g.flag_idx.laya_misread_an_explicit_verdict.includes(i) ? 'wrongE'
  : g.flag_idx.ambiguous.includes(i) ? 'amb' : g.flag_idx.real_note_omissions.includes(i) ? 'omit' : 'ok';
const pct = (p: number) => `${Math.round(p * 100)}%`;

function card(x: number, y: number, w: number, h: number, color: string, header: string) {
  const root = createRef<Rect>();
  const node = (
    <Rect ref={root} x={x} y={y} width={w} height={h} radius={22} fill={PANEL} stroke={color} lineWidth={5} opacity={0}>
      <Txt text={header} fontFamily={SANS} fontSize={28} fill={color} x={-w / 2 + 28} y={-h / 2 + 34} offsetX={-1}/>
    </Rect>);
  return {node, root};
}

export default makeScene2D(function* (view) {
  view.fill(BG);

  // --- the note, the record, and the two typed questions
  const noteCard = card(-470, -175, 900, 450, ACCENT, 'the note the model wrote in Part 7');
  noteCard.root().add(<Txt text={ok.note} textWrap width={840} fontFamily={SANS} fontSize={33} fill="#fff" y={30} lineHeight={46}/>);
  const recCard = card(520, -175, 760, 450, GREEN, 'the record behind it (plain code)');
  const k0 = ok.kept[0];
  recCard.root().add(<Txt text={`verdict: "${ok.verdict}"\nkept[0].id: ${k0.id}\nkept[0].confidence: ${k0.confidence}\nkept[0].uncertain: ${k0.uncertain}`}
    fontFamily={MONO} fontSize={34} fill="#fff" lineHeight={56} y={30}/>);
  view.add(noteCard.node); view.add(recCard.node);

  const qRows = [['was this escalated?', ok.p_escalate], ['is anything uncertain?', ok.p_uncertain]] as [string, number][];
  const qBg: Reference<Rect>[] = qRows.map(() => createRef<Rect>());
  const qFill: Reference<Rect>[] = qRows.map(() => createRef<Rect>());
  const qTxt: Reference<Txt>[] = qRows.map(() => createRef<Txt>());
  const qMark: Reference<Txt>[] = qRows.map(() => createRef<Txt>());
  qRows.forEach(([q, p], i) => view.add(
    <Rect ref={qBg[i]} y={110 + i * 95} width={1700} height={78} radius={16} fill={PANEL} stroke={ACCENT} lineWidth={3} opacity={0}>
      <Txt text={q} fontFamily={SANS} fontSize={38} fontWeight={700} fill="#fff" offsetX={-1} x={-820}/>
      <Rect width={520} height={30} radius={15} fill="#303034" offsetX={-1} x={110}/>
      <Rect ref={qFill[i]} width={0} height={30} radius={15} fill={ACCENT} offsetX={-1} x={110}/>
      <Txt ref={qTxt[i]} text={pct(p)} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff" x={730} opacity={0}/>
      <Txt ref={qMark[i]} text="matches the record" fontFamily={SANS} fontSize={28} fill={GREEN} x={-300} opacity={0} offsetX={-1}/>
    </Rect>));

  // --- the 520-note grid
  const sq: Reference<Rect>[] = Array.from({length: g.n}, () => createRef<Rect>());
  const grid = (
    <Rect ref={createRef<Rect>()} x={-420} y={-60} layout wrap="wrap" width={26 * 28} gap={4} opacity={0}>
      {sq.map((r, i) => <Rect ref={r} width={24} height={24} radius={5} fill="#4a4a4f"/>)}
    </Rect>);
  view.add(grid);
  const gridRect = grid;
  const legend = (
    <Rect x={520} y={-130} layout direction="column" gap={26} opacity={0} ref={createRef<Rect>()}>
      <Txt text={`${g.agree} matched the record`} fontFamily={SANS} fontSize={40} fontWeight={700} fill={GREEN} offsetX={-1}/>
      <Txt text={`${g.laya_wrong} typed model wrong`} fontFamily={SANS} fontSize={40} fontWeight={700} fill={RED} offsetX={-1}/>
      <Txt text={`${g.ambiguous} arguable, ${g.real_omissions} real flaws`} fontFamily={SANS} fontSize={40} fontWeight={700} fill={AMBER} offsetX={-1}/>
    </Rect>);
  view.add(legend);
  const badCard = card(520, -90, 820, 470, RED, `${bad.image.replace('.png', '')}: reads "uncertain" ${pct(bad.p_uncertain)}`);
  badCard.root().add(<Txt text={bad.note} textWrap width={760} fontFamily={SANS} fontSize={30} fill="#fff" y={30} lineHeight={42}/>);
  badCard.root().add(<Txt text="the record: no uncertain box" fontFamily={SANS} fontSize={30} fill={GREEN} y={200} fontWeight={700}/>);
  view.add(badCard.node);
  const omCard = card(520, -90, 820, 470, AMBER, `${am.image.replace('.png', '')}: arguable, not a flaw`);
  omCard.root().add(<Txt text={am.note} textWrap width={760} fontFamily={SANS} fontSize={30} fill="#fff" y={40} lineHeight={42}/>);
  view.add(omCard.node);
  const verdict = (
    <Rect x={0} y={-90} width={1500} height={300} radius={26} fill={PANEL} stroke={AMBER} lineWidth={6} opacity={0} layout direction="column" alignItems="center" justifyContent="center" gap={18} ref={createRef<Rect>()}>
      <Txt text={`${g.disagree} flags raised  ·  ${g.real_omissions} were real flaws`} fontFamily={SANS} fontSize={64} fontWeight={700} fill="#fff"/>
      <Txt text="most flags on good notes: a guard that costs more than it finds" fontFamily={SANS} fontSize={34} fill={AMBER}/>
    </Rect>);
  view.add(verdict);

  yield addSubtitles(view, TL);
  const go = clock();

  // N1 the note
  yield* go(at('N1', 'note')); yield noteCard.root().opacity(1, 0.7);
  // N2 the two questions
  yield* go(at('N2', 'escalated')); yield qBg[0]().opacity(1, 0.5);
  yield* go(at('N2', 'uncertain')); yield qBg[1]().opacity(1, 0.5);
  // N3 the answers (probability bars) and the record
  yield* go(at('N3', 'ninety-seven')); yield all(qFill[0]().width(520 * qRows[0][1], 0.8), qTxt[0]().opacity(1, 0.5));
  yield* go(at('N3', 'ninety-three')); yield all(qFill[1]().width(520 * qRows[1][1], 0.8), qTxt[1]().opacity(1, 0.5));
  yield* go(at('N3', 'record')); yield recCard.root().opacity(1, 0.6); yield qMark[0]().opacity(1, 0.5); yield qMark[1]().opacity(1, 0.5);
  // N4 all 190 notes
  yield* go(at('N4', 'Across'));
  yield all(noteCard.root().opacity(0, 0.5), recCard.root().opacity(0, 0.5), qBg[0]().opacity(0, 0.5), qBg[1]().opacity(0, 0.5));
  yield gridRect.opacity(1, 0.5);
  yield* go(at('N4', 'matched'));
  yield all(...sq.map((r, i) => (kind(i) === 'ok' ? delay(i * 0.006, r().fill(GREEN, 0.3)) : delay(0, r().fill('#4a4a4f', 0.01)))));
  yield legend.opacity(1, 0.5);
  // N5 the other 22
  yield* go(at('N5', 'audited'));
  yield all(...sq.map((r, i) => (kind(i) !== 'ok' ? r().fill(AMBER, 0.5) : r().opacity(0.5, 0.5))));
  yield* go(at('N5', '70'));
  yield all(...sq.map((r, i) => ((kind(i) === 'wrongU' || kind(i) === 'wrongE') ? r().fill(RED, 0.5) : r().scale(1, 0.01))));
  // N6 fifteen: a fine note called uncertain
  yield* go(at('N6', 'Sixty-five'));
  yield legend.opacity(0, 0.4);
  yield all(...sq.map((r, i) => (kind(i) === 'wrongU' ? r().scale(1.3, 0.5) : r().scale(1, 0.01))));
  yield badCard.root().opacity(1, 0.7);
  // N7 two real flaws
  yield* go(at('N7', 'two'));
  yield badCard.root().opacity(0, 0.4);
  yield all(...sq.map((r, i) => (kind(i) === 'amb' ? r().scale(1.9, 0.5) : kind(i) === 'wrongU' ? r().scale(1, 0.4) : r().scale(1, 0.01))));
  yield omCard.root().opacity(1, 0.7);
  // N8 the guard is not worth running
  yield* go(at('N8', 'guard'));
  yield all(omCard.root().opacity(0, 0.4), gridRect.opacity(0, 0.5));
  yield verdict.opacity(1, 0.7);
  yield* go(TL.end);
});
