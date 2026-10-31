import {makeScene2D, Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef, all} from '@motion-canvas/core';
import tl from '../sb_impl/timeline.json';
import facts from '../facts_sb.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, GREY, SANS, MONO, PANEL, RED} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const end = (k: string) => { const b = TL.beats.find(x => x.key === k)!; return b.t0 + b.dur; };

// one panel per sentence: title tag, code lines (ours), real output lines (captured from smoke_test.py)
function panel(tag: string, code: string[], out: string[], outColor = GREEN, outTag = 'real output') {
  const p = createRef<Node>(), o = createRef<Rect>();
  const n = (<Node ref={p} opacity={0}>
    <Txt text={tag} x={-900} y={-470} offsetX={-1} fontFamily={SANS} fontSize={34} fontWeight={700} fill={ACCENT}/>
    <Txt text="code written by us (not the authors')" x={900} y={-470} offsetX={1} fontFamily={SANS} fontSize={26} fill={AMBER}/>
    <Rect x={0} y={-290} width={1800} height={60 + code.length * 52} radius={16} fill="#101114" stroke="#33363d" lineWidth={3} layout direction="column" padding={[22, 34]} gap={6}>
      {code.map(l => <Txt text={l} fontFamily={MONO} fontSize={32} fill="#e6e6e6"/>)}</Rect>
    <Rect ref={o} x={0} y={-290 + (60 + code.length * 52) / 2 + 40 + (30 + out.length * 46) / 2} width={1800} height={50 + out.length * 46} radius={16} fill="#0b140f" stroke={outColor} lineWidth={4} opacity={0} layout direction="column" padding={[18, 34]} gap={4}>
      <Txt text={outTag} fontFamily={SANS} fontSize={24} fill={outColor}/>
      {out.map(l => <Txt text={l} fontFamily={MONO} fontSize={32} fill={outColor}/>)}</Rect>
  </Node>);
  return {n, p, o};
}
export default makeScene2D(function* (view) {
  view.fill(BG);
  const P = [
    panel('1 · load SAM, freeze it', ['sam = SamModel.from_pretrained("facebook/sam-vit-base")', 'enc = sam.vision_encoder          # prompt-free: encoder only', 'for p in enc.parameters():', '    p.requires_grad_(False)'], facts.frozen),
    panel('2 · LoRA + Adapters that start at zero', ['nn.init.zeros_(self.up_q.weight)       # LoRA up-projection', 'nn.init.zeros_(self.up.weight)         # Adapter up-projection', '# => CoSE(x) == SAM(x) at step 0'], facts.ident),
    panel('3 · feed 256 px: HF refuses, call the conv directly', ['h = enc.patch_embed(x256)     # HF: only 1024 px allowed', 'h = enc.patch_embed.projection(x256).permute(0, 2, 3, 1)', 'pe = F.interpolate(enc.pos_embed.permute(0,3,1,2), size=h.shape[1:3], mode="bicubic")', 'h = h + pe.permute(0, 2, 3, 1)       # position embedding 64x64 -> 16x16'], ["ValueError: Input image size (256*256) doesn't match model (1024*1024).", "(fixed: h is now (B, 16, 16, 768))"], RED, 'real error, then the fix'),
    panel('4 · tap blocks 6, 9, 12 (the global-attention blocks)', ['for i, layer in enumerate(enc.layers, start=1):', '    h = layer(h)', '    if i in (6, 9, 12):', '        feats[i] = h.permute(0, 3, 1, 2)'], facts.taps),
    panel('5 · the Mamba scan: chunked PyTorch instead of the CUDA kernel', ['# 64 sequential steps per chunk + 64 across chunks, not 4096', 'y = selective_scan_ref(u, delta, A, B, C, D)'], [...facts.scan, 'backward: 281.94 s (plain loop)  ->  11.54 s (chunked)']),
    panel('6 · check the gradients before you train', ['loss = bce_dice_loss(model(xb), yb)', 'loss.backward()'], facts.grads),
    panel('7 · overfit one image first', ['for step in range(30):', '    loss = bce_dice_loss(model(xb), yb)', '    loss.backward(); opt.step(); opt.zero_grad()'], facts.train.map((l: string) => l.replace('  (memorisation check only, not generalisation)', '')), AMBER, 'real output (memorisation, not accuracy)'),
  ];
  P.forEach(p => view.add(p.n));
  yield addSubtitles(view, TL);
  const go = clock();
  const K = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7'];
  const O = ['transformers', 'zero', 'insists', 'six', 'matched', 'zero', 'point'];       // the word at which the real output appears
  for (let i = 0; i < 7; i++) {
    yield* go(TL.beats.find(b => b.key === K[i])!.t0 + 0.1); yield P[i].p().opacity(1, 0.4);
    yield* go(at(K[i], O[i])); yield P[i].o().opacity(1, 0.4);
    yield* go(end(K[i])); if (i < 6) yield* P[i].p().opacity(0, 0.35);
  }
  yield* go(TL.end);
});
