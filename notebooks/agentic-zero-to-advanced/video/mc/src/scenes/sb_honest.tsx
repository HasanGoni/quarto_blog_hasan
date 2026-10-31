import {makeScene2D, Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef, all} from '@motion-canvas/core';
import tl from '../sb_honest/timeline.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, PANEL, RED} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const end = (k: string) => { const b = TL.beats.find(x => x.key === k)!; return b.t0 + b.dur; };

function card(x: number, y: number, w: number, h: number, c: string, lines: [string, number, string, boolean][]) {
  const r = createRef<Rect>();
  const n = (<Rect ref={r} x={x} y={y} width={w} height={h} radius={22} fill={PANEL} stroke={c} lineWidth={6} opacity={0} layout direction="column" alignItems="center" justifyContent="center" gap={14} padding={24}>
    {lines.map(l => <Txt text={l[0]} fontFamily={SANS} fontSize={l[1]} fontWeight={l[3] ? 700 : 400} fill={l[2]} textAlign="center" width={w - 60} textWrap/>)}</Rect>);
  return {n, r};
}
export default makeScene2D(function* (view) {
  view.fill(BG);
  const W_ = '#fff';
  const h1 = card(0, -130, 1300, 360, RED, [['Code: not released', 64, W_, true], ['"code will be released upon acceptance"', 40, GREY, false], ['everything on our page is our own reading of the equations', 34, AMBER, false]]);
  const h2a = card(-460, -170, 560, 280, AMBER, [['ablation baseline (a)', 34, GREY, false], ['79.05', 110, W_, true], ['Dice on Synapse', 30, GREY, false]]);
  const h2b = card(460, -170, 560, 280, GREY, [['VM-UNet (Table 2)', 34, GREY, false], ['81.08', 110, W_, true], ['Dice on Synapse', 30, GREY, false]]);
  const h2c = card(0, 150, 1400, 100, RED, [['no row for HCEM alone: SAM features and HCEM are not separated', 38, W_, true]]);
  const h3a = card(-560, -170, 480, 240, AMBER, [['Synapse', 32, GREY, false], ['+0.86 Dice', 56, W_, true]]);
  const h3b = card(0, -170, 480, 240, AMBER, [['ISIC17', 32, GREY, false], ['+0.84 mIoU', 56, W_, true]]);
  const h3c = card(560, -170, 480, 240, AMBER, [['ISIC18', 32, GREY, false], ['+0.34 mIoU', 56, W_, true]]);
  const h3d = card(0, 120, 1500, 100, RED, [['no seeds · no error bars · single numbers', 42, W_, true]]);
  const h4a = card(0, -250, 1500, 110, ACCENT, [['abstract: "balancing generalization, segmentation accuracy, and computational efficiency"', 34, W_, false]]);
  const h4b = card(-500, -20, 460, 200, RED, [['parameters', 34, GREY, false], ['not reported', 46, W_, true]]);
  const h4c = card(0, -20, 460, 200, RED, [['FLOPs', 34, GREY, false], ['not reported', 46, W_, true]]);
  const h4d = card(500, -20, 460, 200, RED, [['speed', 34, GREY, false], ['not reported', 46, W_, true]]);
  const h5a = card(-440, -190, 760, 260, GREEN, [['a promising recipe', 54, W_, true], ['SAM features guiding a Mamba decoder is a sensible idea', 32, GREY, false]]);
  const h5b = card(440, -190, 760, 260, AMBER, [['not yet a proof', 54, W_, true], ['one number per dataset, undefined baseline', 32, GREY, false]]);
  const h5c = card(0, 140, 1700, 120, ACCENT, [['what would settle it: official code · mean ± std over seeds · HCEM-only ablation · params and FLOPs', 34, W_, false]]);
  
  [h1, h2a, h2b, h2c, h3a, h3b, h3c, h3d, h4a, h4b, h4c, h4d, h5a, h5b, h5c].forEach(c => view.add(c.n));
  const show = (xs: any[], t = 0.45) => all(...xs.map(x => x.r().opacity(1, t)));
  const hide = (xs: any[], t = 0.35) => all(...xs.map(x => x.r().opacity(0, t)));
  yield addSubtitles(view, TL);
  const go = clock();
  yield* go(at('H1', 'not')); yield show([h1]);
  yield* go(end('H1')); yield* hide([h1]);
  yield* go(at('H2', 'seventy-nine,')); yield show([h2a]);
  yield* go(at('H2', 'V')); yield show([h2b]);
  yield* go(at('H2', 'row')); yield show([h2c]);
  yield* go(end('H2')); yield* hide([h2a, h2b, h2c]);
  yield* go(at('H3', 'seeds')); yield show([h3d]);
  yield* go(at('H3', 'under')); yield show([h3a, h3b, h3c]);
  yield* go(end('H3')); yield* hide([h3a, h3b, h3c, h3d]);
  yield* go(at('H4', 'efficiency,')); yield show([h4a]);
  yield* go(at('H4', 'parameter')); yield show([h4b]);
  yield* go(at('H4', 'flops,')); yield show([h4c]);
  yield* go(at('H4', 'speed.')); yield show([h4d]);
  yield* go(end('H4')); yield* hide([h4a, h4b, h4c, h4d]);
  yield* go(at('H5', 'recipe.')); yield show([h5a]);
  yield* go(at('H5', 'proof.')); yield show([h5b]);
  yield* go(end('H5') - 0.2); yield show([h5c]);
  yield* go(TL.end);
});
