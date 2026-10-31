import {makeScene2D, Rect, Txt, Node, Line} from '@motion-canvas/2d';
import {createRef, all} from '@motion-canvas/core';
import tl from '../sb_sam/timeline.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, MONO, PANEL, RED} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const end = (k: string) => { const b = TL.beats.find(x => x.key === k)!; return b.t0 + b.dur; };
const GLOBAL = [3, 6, 9, 12];

export default makeScene2D(function* (view) {
  view.fill(BG);
  // ---- the 12 blocks (persistent for A1-A4)
  const row = createRef<Node>(); const blk: any[] = [];
  const X0 = -((12 * 130) / 2) + 65;
  view.add(<Node ref={row} opacity={0}>
    <Txt text="SAM image encoder (ViT-B)  ·  12 blocks  ·  768 channels" y={-440} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
    {Array.from({length: 12}, (_, i) => {
      const r = createRef<Rect>(); blk.push(r); const g = GLOBAL.includes(i + 1);
      return (<Rect ref={r} x={X0 + i * 130} y={-340} width={112} height={96} radius={14} fill={PANEL} stroke={g ? ACCENT : GREY} lineWidth={g ? 6 : 3}>
        <Txt text={`${i + 1}`} fontFamily={SANS} fontSize={44} fontWeight={700} fill="#fff"/></Rect>);
    })}
  </Node>);
  // ---- A2: tokens
  const tok = createRef<Node>();
  const GS = 22;
  view.add(<Node ref={tok} x={-480} y={-20} opacity={0}>
    <Rect width={GS * 16} height={GS * 16} fill="#2c2c30" stroke={GREY} lineWidth={3}/>
    {Array.from({length: 15}, (_, i) => <Line points={[[-GS * 8 + (i + 1) * GS, -GS * 8], [-GS * 8 + (i + 1) * GS, GS * 8]]} stroke="#555" lineWidth={1}/>)}
    {Array.from({length: 15}, (_, i) => <Line points={[[-GS * 8, -GS * 8 + (i + 1) * GS], [GS * 8, -GS * 8 + (i + 1) * GS]]} stroke="#555" lineWidth={1}/>)}
    <Txt text="256 px image → 16 × 16 = 256 tokens" y={GS * 8 + 40} fontFamily={SANS} fontSize={30} fill={ACCENT}/>
    <Txt text="one token = one 16 px patch" y={GS * 8 + 80} fontFamily={SANS} fontSize={28} fill={GREY}/>
  </Node>);
  // ---- A3: window vs global
  const win = createRef<Node>(), glo = createRef<Node>();
  view.add(<Node ref={win} x={0} y={-20} opacity={0}>
    <Rect width={300} height={300} fill="#2c2c30" stroke={GREY} lineWidth={3}/>
    <Rect x={-60} y={-60} width={120} height={120} fill={AMBER} opacity={0.35} stroke={AMBER} lineWidth={5}/>
    <Txt text="window blocks" y={200} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
    <Txt text="look inside a 14 × 14 token window" y={246} fontFamily={SANS} fontSize={28} fill={GREY}/>
  </Node>);
  view.add(<Node ref={glo} x={520} y={-20} opacity={0}>
    <Rect width={300} height={300} fill={ACCENT} opacity={0.35} stroke={ACCENT} lineWidth={5}/>
    <Txt text="blocks 3, 6, 9, 12" y={200} fontFamily={SANS} fontSize={34} fontWeight={700} fill="#fff"/>
    <Txt text="global: every token sees all tokens" y={246} fontFamily={SANS} fontSize={28} fill={GREY}/>
  </Node>);
  // ---- A4: taps
  const taps: any[] = [];
  [6, 9, 12].forEach((b, j) => { const r = createRef<Node>(); taps.push(r);
    view.add(<Node ref={r} x={X0 + (b - 1) * 130} y={-255} opacity={0}>
      <Line points={[[0, -10], [0, 40]]} stroke={GREEN} lineWidth={6} endArrow arrowSize={16}/>
      <Txt text={`S${b}`} y={80} fontFamily={SANS} fontSize={46} fontWeight={700} fill={GREEN}/></Node>); });
  const tapNote = createRef<Txt>();
  view.add(<Txt ref={tapNote} y={60} text="three feature maps, each 768 × 16 × 16, go to the Mamba decoder" fontFamily={SANS} fontSize={36} fill={GREEN} opacity={0}/>);

  // ---- A5: what is kept
  const keep = createRef<Node>();
  view.add(<Node ref={keep} opacity={0}>
    <Rect x={-560} y={-170} width={420} height={110} radius={18} fill={PANEL} stroke={RED} lineWidth={5}><Txt text="prompt encoder" fontFamily={SANS} fontSize={40} fill={GREY}/></Rect>
    <Line points={[[-760, -230], [-360, -110]]} stroke={RED} lineWidth={8}/><Line points={[[-760, -110], [-360, -230]]} stroke={RED} lineWidth={8}/>
    <Rect x={-560} y={0} width={420} height={110} radius={18} fill={PANEL} stroke={RED} lineWidth={5}><Txt text="mask decoder (optional)" fontFamily={SANS} fontSize={36} fill={GREY}/></Rect>
    <Rect x={330} y={-80} width={700} height={260} radius={24} fill={PANEL} stroke={GREEN} lineWidth={8} layout direction="column" alignItems="center" justifyContent="center" gap={14}>
      <Txt text="image encoder only" fontFamily={SANS} fontSize={58} fontWeight={700} fill="#fff"/>
      <Txt text="prompt-free: no clicks, no boxes" fontFamily={SANS} fontSize={36} fill={GREEN}/></Rect>
  </Node>);

  // ---- A6: one block opened
  const one = createRef<Node>(), lora = createRef<Rect>(), ad1 = createRef<Rect>(), ad2 = createRef<Rect>();
  const BX = (y: number, t: string, c = GREY, ref?: any, w = 480) => (<Rect ref={ref} y={y} width={w} height={84} radius={14} fill={PANEL} stroke={c} lineWidth={4}><Txt text={t} fontFamily={SANS} fontSize={34} fill="#fff"/></Rect>);
  view.add(<Node ref={one} x={-300} opacity={0}>
    <Txt text="inside one ViT block" y={-465} fontFamily={SANS} fontSize={34} fill={GREY}/>
    {BX(-400, 'LayerNorm')}{BX(-300, 'attention (Q, K, V)', ACCENT)}
    <Rect ref={ad1} y={-200} width={480} height={74} radius={14} fill={PANEL} stroke={AMBER} lineWidth={5} opacity={0}><Txt text="Adapter 1" fontFamily={SANS} fontSize={34} fill={AMBER}/></Rect>
    {BX(-110, 'LayerNorm')}{BX(-10, 'MLP', ACCENT)}
    <Rect ref={ad2} y={90} width={480} height={74} radius={14} fill={PANEL} stroke={AMBER} lineWidth={5} opacity={0}><Txt text="Adapter 2 (in the MLP path)" fontFamily={SANS} fontSize={32} fill={AMBER}/></Rect>
  </Node>);
  view.add(<Rect ref={lora} x={170} y={-300} width={440} height={110} radius={14} fill={PANEL} stroke={GREEN} lineWidth={5} opacity={0} layout direction="column" alignItems="center" justifyContent="center">
    <Txt text="LoRA on Q and V" fontFamily={SANS} fontSize={34} fontWeight={700} fill={GREEN}/><Txt text="small low-rank update" fontFamily={SANS} fontSize={28} fill={GREY}/></Rect>);

  // ---- A7: numbers
  const nums = createRef<Node>(), barF = createRef<Rect>(), barT = createRef<Rect>();
  view.add(<Node ref={nums} opacity={0}>
    <Txt text="frozen SAM encoder" x={-860} y={-250} offsetX={-1} fontFamily={SANS} fontSize={34} fill={GREY}/>
    <Rect ref={barF} x={-860} y={-190} offsetX={-1} width={1400} height={64} radius={10} fill={GREY}/>
    <Txt text="89.67 M frozen" x={560} y={-190} offsetX={-1} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
    <Txt text="LoRA + Adapters (our implementation)" x={-860} y={-80} offsetX={-1} fontFamily={SANS} fontSize={34} fill={GREY}/>
    <Rect ref={barT} x={-860} y={-20} offsetX={-1} width={1400 * 2.53 / 89.67} height={64} radius={6} fill={GREEN}/>
    <Txt text="2.53 M trainable" x={-860 + 1400 * 2.53 / 89.67 + 24} y={-20} offsetX={-1} fontFamily={SANS} fontSize={40} fontWeight={700} fill={GREEN}/>
    <Txt text="bars to scale: 2.8 % of the encoder" x={-860} y={90} offsetX={-1} fontFamily={SANS} fontSize={32} fill={AMBER}/>
  </Node>);

  yield addSubtitles(view, TL);
  const go = clock();
  yield* go(at('A1', 'encoder')); yield row().opacity(1, 0.5);
  yield* go(end('A1')); yield* go(at('A2', 'sixteen')); yield tok().opacity(1, 0.5);
  yield* go(end('A2')); yield* tok().opacity(0, 0.35);
  yield* go(at('A3', 'windows')); yield win().opacity(1, 0.4);
  yield* go(at('A3', 'whole')); yield glo().opacity(1, 0.4);
  yield all(...GLOBAL.map(g => blk[g - 1]().fill(ACCENT, 0.4)));
  yield* go(end('A3')); yield* all(win().opacity(0, 0.35), glo().opacity(0, 0.35));
  yield* go(at('A4', 'six')); yield taps[0]().opacity(1, 0.4);
  yield* go(at('A4', 'nine')); yield taps[1]().opacity(1, 0.4);
  yield* go(at('A4', 'twelve')); yield taps[2]().opacity(1, 0.4);
  yield* go(at('A4', 'global')); yield tapNote().opacity(1, 0.4);
  yield* go(end('A4')); yield* all(row().opacity(0, 0.35), ...taps.map(t => t().opacity(0, 0.35)), tapNote().opacity(0, 0.35));
  yield* go(at('A5', 'prompt')); yield keep().opacity(1, 0.5);
  yield* go(end('A5')); yield* keep().opacity(0, 0.35);
  yield* go(at('A6', 'two')); yield one().opacity(1, 0.5);
  yield* go(at('A6', 'low-rank')); yield lora().opacity(1, 0.4);
  yield* go(at('A6', 'adapters')); yield all(ad1().opacity(1, 0.4), ad2().opacity(1, 0.4));
  yield* go(end('A6')); yield* all(one().opacity(0, 0.35), lora().opacity(0, 0.35));
  yield* go(at('A7', 'frozen')); yield nums().opacity(1, 0.5);
  yield* go(TL.end);
});
