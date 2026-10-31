import {makeScene2D, Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef, all} from '@motion-canvas/core';
import tl from '../sb_story/timeline.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL, RED} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));

// one expert card: title + two lines, a coloured edge
function card(x: number, title: string, l1: string, l2: string, color: string) {
  const r = createRef<Rect>();
  const node = (
    <Rect ref={r} x={x} y={-110} width={420} height={330} radius={22} fill={PANEL} stroke={color} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={18} padding={24}>
      <Txt text={title} fontFamily={SANS} fontSize={50} fontWeight={700} fill="#fff"/>
      <Txt text={l1} fontFamily={SANS} fontSize={34} fill={color} textAlign="center"/>
      <Txt text={l2} fontFamily={SANS} fontSize={30} fill={GREY} textAlign="center"/>
    </Rect>);
  return {node, r};
}

export default makeScene2D(function* (view) {
  view.fill(BG);

  // S1: the two needs
  const big = createRef<Rect>(), edge = createRef<Rect>();
  view.add(
    <Rect ref={big} x={-420} y={-130} width={700} height={190} radius={26} fill={PANEL} stroke={ACCENT} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={8}>
      <Txt text="the big picture" fontFamily={SANS} fontSize={54} fontWeight={700} fill="#fff"/>
      <Txt text="which organ is this? where does it sit?" fontFamily={SANS} fontSize={30} fill={GREY}/>
    </Rect>);
  view.add(
    <Rect ref={edge} x={420} y={-130} width={700} height={190} radius={26} fill={PANEL} stroke={GREEN} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={8}>
      <Txt text="the exact edge" fontFamily={SANS} fontSize={54} fontWeight={700} fill="#fff"/>
      <Txt text="where does the lesion stop, to the pixel?" fontFamily={SANS} fontSize={30} fill={GREY}/>
    </Rect>);

  // S2-S4: four experts
  const unet = card(-690, 'U-Net', 'sees edges', 'but only locally', GREEN);
  const trf = card(-230, 'Transformer', 'sees everything', 'cost grows with N²', AMBER);
  const mam = card(230, 'Mamba', 'sees everything', 'cost grows with N', ACCENT);
  const sam = card(690, 'SAM', 'knows objects', 'but not skin or CT', RED);
  [unet, trf, mam, sam].forEach(c => view.add(c.node));
  const gap = createRef<Txt>(), gap2 = createRef<Txt>();
  view.add(<Txt ref={gap} x={230} y={120} width={400} textWrap textAlign="center" text="no idea what an object is" fontFamily={SANS} fontSize={34} fill={AMBER} opacity={0}/>);
  view.add(<Txt ref={gap2} x={690} y={120} width={400} textWrap textAlign="center" text="trained on everyday photos" fontFamily={SANS} fontSize={34} fill={AMBER} opacity={0}/>);

  // S5: combine
  const both = createRef<Rect>();
  view.add(
    <Rect ref={both} y={-20} width={1500} height={300} radius={28} fill={PANEL} stroke={ACCENT} lineWidth={8} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={16}>
      <Txt text="SAMBA-UNet" fontFamily={SANS} fontSize={76} fontWeight={700} fill="#fff"/>
      <Txt text="Mamba: where exactly   +   SAM: what is it" fontFamily={SANS} fontSize={44} fill={ACCENT}/>
      <Txt text="SAM's features guide the Mamba decoder · no prompts needed" fontFamily={SANS} fontSize={32} fill={GREY}/>
    </Rect>);

  // S6-S7: the claim
  const oursC = createRef<Rect>(), vmC = createRef<Rect>(), delta = createRef<Rect>(), note = createRef<Txt>();
  view.add(
    <Rect ref={oursC} x={-480} y={-130} width={560} height={300} radius={26} fill={PANEL} stroke={ACCENT} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={10}>
      <Txt text="SAMBA-UNet" fontFamily={SANS} fontSize={40} fill={GREY}/>
      <Txt text="81.94" fontFamily={SANS} fontSize={120} fontWeight={700} fill="#fff"/>
      <Txt text="Dice, Synapse (8 organs)" fontFamily={SANS} fontSize={30} fill={GREY}/>
    </Rect>);
  view.add(
    <Rect ref={vmC} x={480} y={-130} width={560} height={300} radius={26} fill={PANEL} stroke={GREY} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={10}>
      <Txt text="VM-UNet" fontFamily={SANS} fontSize={40} fill={GREY}/>
      <Txt text="81.08" fontFamily={SANS} fontSize={120} fontWeight={700} fill="#fff"/>
      <Txt text="Dice, Synapse (8 organs)" fontFamily={SANS} fontSize={30} fill={GREY}/>
    </Rect>);
  view.add(
    <Rect ref={delta} y={150} width={1180} height={110} radius={22} fill={PANEL} stroke={AMBER} lineWidth={6} opacity={0}
          layout alignItems="center" justifyContent="center">
      <Txt text="+0.86 Dice · one run · no error bars" fontFamily={SANS} fontSize={44} fontWeight={700} fill={AMBER}/>
    </Rect>);

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(at('S1', 'picture')); yield big().opacity(1, 0.4);
  yield* go(at('S1', 'edge')); yield edge().opacity(1, 0.4);
  // clear, then the four experts
  yield* go(TL.beats.find(b => b.key === 'S1')!.t0 + TL.beats.find(b => b.key === 'S1')!.dur);
  yield* all(big().opacity(0, 0.35), edge().opacity(0, 0.35));
  yield* go(at('S2', 'U-Net')); yield unet.r().opacity(1, 0.4);
  yield* go(at('S2', 'transformer')); yield trf.r().opacity(1, 0.4);
  yield* go(at('S3', 'Mamba')); yield mam.r().opacity(1, 0.4);
  yield* go(at('S3', 'objects')); yield gap().opacity(1, 0.4);
  yield* go(at('S4', 'SAM')); yield sam.r().opacity(1, 0.4);
  yield* go(at('S4', 'nothing')); yield gap2().opacity(1, 0.4);
  yield* go(TL.beats.find(b => b.key === 'S4')!.t0 + TL.beats.find(b => b.key === 'S4')!.dur);
  yield* all(...[unet, trf, mam, sam].map(c => c.r().opacity(0, 0.35)), gap().opacity(0, 0.35), gap2().opacity(0, 0.35));
  yield* go(at('S5', 'both')); yield both().opacity(1, 0.5);
  yield* go(TL.beats.find(b => b.key === 'S5')!.t0 + TL.beats.find(b => b.key === 'S5')!.dur);
  yield* both().opacity(0, 0.35);
  yield* go(at('S6', '81.94')); yield oursC().opacity(1, 0.4);
  yield* go(at('S6', '81.08')); yield vmC().opacity(1, 0.4);
  yield* go(at('S7', 'gain')); yield delta().opacity(1, 0.5);
  yield* go(TL.end);
});
