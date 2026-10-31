import {makeScene2D, Rect, Txt, Node, Line} from '@motion-canvas/2d';
import {createRef, all} from '@motion-canvas/core';
import tl from '../sb_arch/timeline.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL, RED} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const end = (k: string) => { const b = TL.beats.find(x => x.key === k)!; return b.t0 + b.dur; };
const ROW = [-330, -190, -50, 90];            // y of the four resolution levels (top = finest)
const EX = -720, SX = -280, HX = 140, DX = 640;

function box(x: number, y: number, w: number, h: number, c: string, l1: string, l2 = '', f1 = 34) {
  const r = createRef<Rect>();
  const n = (<Rect ref={r} x={x} y={y} width={w} height={h} radius={14} fill={PANEL} stroke={c} lineWidth={4} opacity={0} layout direction="column" alignItems="center" justifyContent="center" gap={4}>
    <Txt text={l1} fontFamily={SANS} fontSize={f1} fontWeight={700} fill="#fff"/>{l2 ? <Txt text={l2} fontFamily={SANS} fontSize={26} fill={GREY}/> : null}</Rect>);
  return {n, r};
}
function arrow(pts: number[][], c = GREY) {
  const r = createRef<Line>();
  return {n: <Line ref={r} points={pts} stroke={c} lineWidth={5} endArrow arrowSize={14} opacity={0}/>, r};
}
function chip(x: number, y: number, t: string, c: string) {
  const r = createRef<Rect>();
  return {n: <Rect ref={r} x={x} y={y} padding={[4, 12]} radius={10} fill={BG} stroke={c} lineWidth={3} opacity={0} layout><Txt text={t} fontFamily={SANS} fontSize={28} fontWeight={700} fill={c}/></Rect>, r};
}

export default makeScene2D(function* (view) {
  view.fill(BG);
  const inp = box(-500, -455, 380, 58, GREY, 'image 3 × 256 × 256', '', 28);
  const enc = [['96 × 64 × 64'], ['192 × 32 × 32'], ['384 × 16 × 16'], ['768 × 8 × 8']].map((s, i) => box(EX, ROW[i], 360, 100, GREEN, `Mamba (VSS ×2)`, s[0]));
  const sam = box(SX, -165, 250, 470, ACCENT, 'SAM', 'frozen + CoSE');
  const aIn1 = arrow([[-560, -425], [EX, -382]]), aIn2 = arrow([[-440, -425], [SX, -400]]);
  const aDown = [0, 1, 2].map(i => arrow([[EX, ROW[i] + 50], [EX, ROW[i + 1] - 50]], GREEN));
  const M = [0, 1, 2].map(i => chip(EX + 250, ROW[i] + 38, `M${i + 1}`, AMBER));
  const Mh = [0, 1, 2].map(i => chip(HX - 175, ROW[i] + 38, `M${i + 1}`, AMBER));
  const S = [0, 1, 2].map(i => chip(SX + 170, ROW[i] - 30, `S${[6, 9, 12][i]}`, ACCENT));
  const Sh = [0, 1, 2].map(i => chip(HX - 175, ROW[i] - 30, `S${[6, 9, 12][i]}`, ACCENT));
  const sShape = chip(SX, 120, '768 × 16 × 16 each', ACCENT);
  const hc = [0, 1, 2].map(i => box(HX, ROW[i], 200, 100, AMBER, 'HCEM', ''));
  const dec = [0, 1, 2, 3].map(i => box(DX, ROW[i], 360, 100, GREEN, 'Mamba (VSS ×2)', ['96 × 64 × 64', '192 × 32 × 32', '384 × 16 × 16', '768 × 8 × 8 (bottleneck)'][i]));
  const aBot = arrow([[EX + 180, ROW[3]], [DX - 180, ROW[3]]], GREEN);
  const aUp = [3, 2, 1].map(i => arrow([[DX, ROW[i] - 50], [DX, ROW[i - 1] + 50]], GREEN));
  const aHc = [0, 1, 2].map(i => arrow([[HX + 100, ROW[i]], [DX - 180, ROW[i]]], AMBER));
  const out = box(DX, -455, 360, 58, GREEN, 'mask 1 × 256 × 256', '', 28);
  const aOut = arrow([[DX, ROW[0] - 50], [DX, -420]], GREEN);
  const cap = createRef<Rect>(), cap1 = createRef<Txt>(), cap2 = createRef<Txt>();
  view.add(<Rect ref={cap} y={212} width={1560} height={104} radius={18} fill={PANEL} stroke={AMBER} lineWidth={5} opacity={0} layout direction="column" alignItems="center" justifyContent="center" gap={6}>
    <Txt ref={cap1} text="" fontFamily={SANS} fontSize={38} fontWeight={700} fill="#fff"/>
    <Txt ref={cap2} text="" fontFamily={SANS} fontSize={30} fill={GREY}/></Rect>);
  const all_ = [inp, ...enc, sam, ...hc, ...dec, out];
  [inp, ...enc, sam, ...hc, ...dec, out].forEach(b => view.add(b.n));
  [aIn1, aIn2, ...aDown, ...M, ...Mh, ...S, ...Sh, sShape, aBot, ...aUp, ...aHc, aOut].forEach(a => view.add(a.n));
  const show = (xs: any[], t = 0.4) => all(...xs.map(x => x.r().opacity(1, t)));
  const hide = (xs: any[], t = 0.3) => all(...xs.map(x => x.r().opacity(0, t)));

  yield addSubtitles(view, TL);
  const go = clock();
  yield* go(at('N1', 'whole')); yield show([inp]);
  yield* go(at('N1', 'both')); yield show([...enc, sam, aIn1, aIn2, ...aDown], 0.5);
  yield* go(at('N2', 'sixty-four')); yield show([M[0]]);
  yield* go(at('N2', 'thirty-two')); yield show([M[1]]);
  yield* go(at('N2', 'sixteen')); yield show([M[2]]);
  yield* go(at('N3', 'three')); yield show(S);
  yield* go(at('N3', 'sixteen')); yield show([sShape]);
  yield* go(at('N4', 'upsamples')); yield show([...hc, ...Mh, ...Sh]);
  yield cap().opacity(1, 0.4); cap1().text('HCEM:  upsample S  →  concatenate with M  →  1 × 1 conv'); cap2().text('Eq. 3 · (S6, M1) · (S9, M2) · (S12, M3)');
  yield* go(at('N5', 'gate')); cap1().text('gate = sigmoid( average-pool + max-pool )  per channel'); cap2().text('Eq. 4 · a volume knob for every channel · added back residually');
  yield* go(at('N6', 'concatenated')); cap1().text('concatenate with M again  →  1 × 1 conv  =  guided skip'); cap2().text('handed to the decoder at the same resolution');
  yield show([...aHc, ...dec.slice(0, 0)]);
  yield* go(at('N6', 'decoder')); yield show([dec[3], aBot, ...dec.slice(0, 3)], 0.5);
  yield* go(at('N7', 'three')); yield show(aUp, 0.6);
  yield* go(at('N7', 'mask')); yield show([out, aOut]);
  yield* go(TL.end);
});
