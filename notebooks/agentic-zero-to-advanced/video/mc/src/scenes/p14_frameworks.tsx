import {makeScene2D, Rect, Txt, Node, Circle} from '@motion-canvas/2d';
import {createRef, Reference} from '@motion-canvas/core';
import tl from '../p14_frameworks/timeline.json';
import facts from '../facts_p14.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, RED, GREY, SANS, MONO, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const FW = facts.fw;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const start = (k: string) => T(TL, k, 0);
const fmt = (n: number) => n.toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(BG);
  const stage = (): Reference<Node> => createRef<Node>();
  const S = Array.from({length: 8}, stage);
  S.forEach(r => view.add(<Node ref={r} opacity={0}/>));
  const show = function* (i: number) {   // clear the previous stage completely, then bring the next one in (no crossfades)
    if (i > 0) yield* S[i - 1]().opacity(0, 0.3);
    S[i]().opacity(1);
  };

  // ---- W1: three frameworks
  const cards: [string, string, string][] = [['my own loop', 'plain Python, 150 lines', GREEN], ['Deep Agents', 'LangChain library', ACCENT], ['Google ADK', 'google-adk 2.13', ACCENT]];
  const c1: Reference<Rect>[] = cards.map(() => createRef<Rect>());
  S[0]().add(<Txt text="same model · same 17 questions" y={-440} fontFamily={SANS} fontSize={38} fill={GREY}/>);
  cards.forEach(([a, b, col], i) => S[0]().add(
    <Rect ref={c1[i]} x={-560 + i * 560} y={-170} width={500} height={260} radius={26} fill={PANEL} stroke={col} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={18}>
      <Txt text={a} fontFamily={SANS} fontSize={48} fontWeight={700} fill="#fff"/>
      <Txt text={b} fontFamily={SANS} fontSize={28} fill={col}/>
    </Rect>));

  // ---- W2: tokens per correct needle answer
  const bars: [string, number, string][] = [['my loop', FW.t1_tokens.harness, GREEN], ['ADK, my tools', FW.t1_tokens.adk, ACCENT],
                                           ['ADK, its own tools', FW.t1_tokens.adk_env, ACCENT], ['Deep Agents', FW.t1_tokens.da, ACCENT]];
  const X0 = -430, SC = 900 / FW.t1_tokens.da;
  const bw: Reference<Rect>[] = bars.map(() => createRef<Rect>());
  const bl: Reference<Rect>[] = bars.map(() => createRef<Rect>());
  S[1]().add(<Txt text="prompt tokens per correct answer, needle in the big table" y={-450} fontFamily={SANS} fontSize={34} fill={GREY}/>);
  bars.forEach(([name, tok, col], i) => {
    const y = -300 + i * 140;
    S[1]().add(
      <Rect ref={bl[i]} x={X0 - 20 - 260} y={y} width={520} height={90} layout alignItems="center" justifyContent="end" opacity={0}>
        <Txt text={name} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
      </Rect>);
    S[1]().add(
      <Rect ref={bw[i]} x={X0 + 2} y={y} width={tok * SC} height={80} offsetX={-1} radius={10} fill={col} opacity={0}>
        <Txt text={fmt(tok)} x={tok * SC / 2 + 90} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
      </Rect>);
  });

  // ---- W3: totals out of 17
  const rows: [string, number, string, string][] = [
    ['Deep Agents', FW.totals.da, ACCENT, 'six'], ['+ sub-agent', FW.totals.da_chunk, ACCENT, 'ten'],
    ['ADK, my tools', FW.totals.adk, ACCENT, 'nine'], ['+ sub-agent', FW.totals.adk_chunk, ACCENT, 'eight'],
    ['ADK, its own tools', FW.totals.adk_env, ACCENT, 'twelve'], ['my loop', FW.totals.harness, GREEN, 'loop']];
  const TX0 = -380, TSC = 800 / FW.n;
  const tr: Reference<Rect>[] = rows.map(() => createRef<Rect>());
  const tlb: Reference<Rect>[] = rows.map(() => createRef<Rect>());
  S[2]().add(<Txt text={`correct answers out of ${FW.n}`} y={-450} fontFamily={SANS} fontSize={34} fill={GREY}/>);
  rows.forEach(([name, v, col], i) => {
    const y = -380 + i * 108;
    S[2]().add(
      <Rect ref={tlb[i]} x={TX0 - 20 - 260} y={y} width={520} height={80} layout alignItems="center" justifyContent="end" opacity={0}>
        <Txt text={name} fontFamily={SANS} fontSize={32} fontWeight={700} fill="#fff"/>
      </Rect>);
    S[2]().add(
      <Rect ref={tr[i]} x={TX0 + 2} y={y} width={v * TSC} height={70} offsetX={-1} radius={10} fill={col} opacity={0}>
        <Txt text={String(v)} x={v * TSC / 2 + 50} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
      </Rect>);
  });

  // ---- W4: runs that hit the 40-call cap (Deep Agents, 34 runs)
  const dots: Reference<Circle>[] = Array.from({length: FW.da_runs}, () => createRef<Circle>());
  S[3]().add(<Txt text="Deep Agents: 34 runs" y={-450} fontFamily={SANS} fontSize={36} fill={GREY}/>);
  dots.forEach((d, i) => S[3]().add(
    <Circle ref={d} x={-720 + (i % 17) * 90} y={-250 + Math.floor(i / 17) * 110} size={64} fill={GREY} opacity={0}/>));
  const capLab = createRef<Txt>();
  S[3]().add(<Txt ref={capLab} text={`${FW.da_cap} of ${FW.da_runs} ended on the 40-call cap, with no answer`} y={0} fontFamily={SANS}
                  fontSize={42} fontWeight={700} fill={RED} opacity={0}/>);

  // ---- W5: same line, again and again
  const row = createRef<Rect>();
  const calls = createRef<Node>();
  const rep: Reference<Txt>[] = Array.from({length: 6}, () => createRef<Txt>());
  S[4]().add(
    <Rect ref={row} y={-380} width={1500} height={110} radius={16} fill={PANEL} stroke={GREEN} lineWidth={5} opacity={0} layout alignItems="center" justifyContent="center">
      <Txt text="test_00404, …, False, 1, 93, 30907, …    ← n_voids 1, void_area 93" fontFamily={MONO} fontSize={32} fill="#fff"/>
    </Rect>);
  S[4]().add(<Node ref={calls}>{rep.map((r, i) =>
    <Txt ref={r} text={`read_file(offset=3805, limit=${1 + (i % 2)})`} y={-230 + i * 78} fontFamily={MONO} fontSize={34} fill={i % 2 ? AMBER : ACCENT} opacity={0}/>)}</Node>);
  S[4]().add(<Txt text="…the answer was already in the context" y={270} fontFamily={SANS} fontSize={34} fill={GREY}/>);

  // ---- W6: the guard
  const guardChips = createRef<Node>();
  const gcol: Record<string, string> = {stopped: AMBER, answered: GREEN, looped: RED};
  const gtxt: Record<string, string> = {stopped: 'stopped after 7 calls, no answer', answered: 'answered in 4 calls', looped: 'still looped (33 calls)'};
  const gref: Reference<Rect>[] = FW.guard.map(() => createRef<Rect>());
  S[5]().add(<Txt text="repeat guard: stop at the 3rd identical call · 3 looping runs" y={-450} fontFamily={SANS} fontSize={34} fill={GREY}/>);
  FW.guard.forEach((g: string, i: number) => S[5]().add(
    <Rect ref={gref[i]} x={-560 + i * 560} y={-130} width={520} height={200} radius={22} fill={PANEL} stroke={gcol[g]} lineWidth={6} opacity={0}
          layout alignItems="center" justifyContent="center" padding={[0, 20]}>
      <Txt text={gtxt[g]} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff" textAlign="center" textWrap width={470}/>
    </Rect>));

  // ---- W7: the shell in a box
  const envCard = createRef<Rect>(), box = createRef<Rect>(), nonet = createRef<Txt>();
  S[6]().add(
    <Rect ref={envCard} x={-560} y={-170} width={560} height={240} radius={22} fill={PANEL} stroke={RED} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={14}>
      <Txt text="Execute  (a real shell)" fontFamily={MONO} fontSize={34} fill="#fff"/>
      <Txt text="sees your API keys" fontFamily={SANS} fontSize={32} fill={RED}/>
    </Rect>);
  S[6]().add(
    <Rect ref={box} x={280} y={-170} width={760} height={420} radius={30} fill={'rgba(0,0,0,0)'} stroke={GREEN} lineWidth={8} lineDash={[26, 16]} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={16}>
      <Txt text="Docker container" fontFamily={SANS} fontSize={40} fontWeight={700} fill={GREEN}/>
      <Txt text="read-only · no keys · only a copy of the data" fontFamily={SANS} fontSize={28} fill="#fff"/>
    </Rect>);
  S[6]().add(<Txt ref={nonet} text="no network" x={280} y={110} fontFamily={SANS} fontSize={40} fontWeight={700} fill={GREEN} opacity={0}/>);

  // ---- W8: takeaways
  const take = ['set a cap', 'read the loops', 'box the shell'];
  const tk: Reference<Rect>[] = take.map(() => createRef<Rect>());
  const note = createRef<Txt>();
  take.forEach((t, i) => S[7]().add(
    <Rect ref={tk[i]} x={-560 + i * 560} y={-200} width={500} height={160} radius={26} fill={PANEL} stroke={GREEN} lineWidth={6} opacity={0}
          layout alignItems="center" justifyContent="center">
      <Txt text={t} fontFamily={SANS} fontSize={44} fontWeight={700} fill="#fff"/>
    </Rect>));
  S[7]().add(<Txt ref={note} text="one local model (Qwen3.8-27B), not Gemini" y={40} fontFamily={SANS} fontSize={36} fill={AMBER} opacity={0}/>);

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(start('W1')); yield* show(0);
  yield* go(at('W1', 'loop')); yield c1[0]().opacity(1, 0.5);
  yield* go(at('W1', 'Agents')); yield c1[1]().opacity(1, 0.5);
  yield* go(at('W1', 'ADK')); yield c1[2]().opacity(1, 0.5);

  const barKey: [string, string][] = [['W2', 'twenty-eight'], ['W3', 'five'], ['W3', 'eight'], ['W4', 'thirteen']];
  yield* go(start('W2')); yield* show(1);
  for (let i = 0; i < 4; i++) { yield* go(at(barKey[i][0], barKey[i][1])); yield bl[i]().opacity(1, 0.3); yield bw[i]().opacity(1, 0.5); }

  const rowKey: [string, string][] = [['W5', 'six'], ['W5', 'ten'], ['W5', 'nine'], ['W5', 'eight'], ['W6', 'twelve'], ['W6', 'loop']];
  yield* go(start('W5')); yield* show(2);
  for (let i = 0; i < rows.length; i++) { yield* go(at(rowKey[i][0], rowKey[i][1])); yield tlb[i]().opacity(1, 0.3); yield tr[i]().opacity(1, 0.5); }

  yield* go(start('W7')); yield* show(3);
  yield* go(at('W7', 'Deep')); for (const d of dots) yield d().opacity(1, 0.4);
  yield* go(at('W7', 'eighteen'));
  for (let i = 0; i < FW.da_cap; i++) yield dots[i]().fill(RED, 0.3);
  yield capLab().opacity(1, 0.5);

  yield* go(start('W8')); yield* show(4);
  yield* go(at('W8', 'answer')); yield row().opacity(1, 0.5);
  yield* go(at('W8', 'reread'));
  for (let i = 0; i < rep.length; i++) { yield rep[i]().opacity(1, 0.15); yield* go(at('W8', 'reread') + 0.2 * (i + 1)); }

  yield* go(start('W9')); yield* show(5);
  yield* go(at('W9', 'stops')); yield gref[0]().opacity(1, 0.5);
  yield* go(at('W9', 'saved')); yield gref[1]().opacity(1, 0.5);
  yield* go(at('W9', 'did')); yield gref[2]().opacity(1, 0.5);

  yield* go(start('W10')); yield* show(6);
  yield* go(at('W10', 'shell')); yield envCard().opacity(1, 0.5);
  yield* go(at('W10', 'Docker')); yield box().opacity(1, 0.5);
  yield* go(at('W10', 'no')); yield nonet().opacity(1, 0.4);

  yield* go(start('W11')); yield* show(7);
  yield* go(at('W11', 'cap')); yield tk[0]().opacity(1, 0.4);
  yield* go(at('W11', 'loops')); yield tk[1]().opacity(1, 0.4);
  yield* go(at('W11', 'box')); yield tk[2]().opacity(1, 0.4);
  yield* go(at('W11', 'local')); yield note().opacity(1, 0.5);
  yield* go(TL.end);
});
