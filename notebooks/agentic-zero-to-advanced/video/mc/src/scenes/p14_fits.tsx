import {makeScene2D, Rect, Txt} from '@motion-canvas/2d';
import {createRef} from '@motion-canvas/core';
import tl from '../p14_fits/timeline.json';
import facts from '../facts_p14.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, AMBER, RED, GREY, SANS, MONO, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const WIN_W = 1500;
const used = Math.round(WIN_W * facts.t2.stuff.tokens / facts.window);
const fmt = (n: number) => n.toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(BG);
  const winRef = createRef<Rect>(), barRef = createRef<Rect>(), fitsRef = createRef<Txt>();
  view.add(
    <Rect ref={winRef} y={-400} width={WIN_W} height={76} radius={14} stroke={ACCENT} lineWidth={6} opacity={0}>
      <Txt text={`context window: ${fmt(facts.window)} tokens`} x={0} y={-62} fontFamily={SANS} fontSize={30} fill={ACCENT}/>
      <Rect ref={barRef} x={-WIN_W / 2 + 4} offsetX={-1} width={0} height={64} radius={10} fill={GREEN} opacity={0.9}/>
      <Txt ref={fitsRef} text="FITS" x={430} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff" opacity={0}/>
    </Rect>);
  view.add(<Txt text={`200 reports = about ${fmt(facts.t2.stuff.tokens)} tokens with the question`} y={-318} fontFamily={SANS} fontSize={30} fill={GREY}/>);

  const pasteRef = createRef<Rect>(), toolsRef = createRef<Rect>();
  view.add(
    <Rect ref={pasteRef} x={-470} y={-190} width={880} height={170} radius={22} fill={PANEL} stroke={GREEN} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={6}>
      <Txt text="paste them all in" fontFamily={SANS} fontSize={34} fill={GREY}/>
      <Txt text={`${facts.t2.stuff.correct} out of ${facts.t2.stuff.n} right`} fontFamily={SANS} fontSize={64} fontWeight={700} fill={GREEN}/>
    </Rect>);
  view.add(
    <Rect ref={toolsRef} x={470} y={-190} width={880} height={170} radius={22} fill={PANEL} stroke={RED} lineWidth={6} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={6}>
      <Txt text="every tool setup" fontFamily={SANS} fontSize={34} fill={GREY}/>
      <Txt text="0 out of 3 right" fontFamily={SANS} fontSize={64} fontWeight={700} fill={RED}/>
    </Rect>);

  const bugRef = createRef<Rect>(), cutRef = createRef<Rect>();
  view.add(
    <Rect ref={bugRef} y={30} width={1500} height={190} radius={22} fill={PANEL} stroke={AMBER} lineWidth={6} opacity={0}>
      <Txt text="first run: answer capped at 1,200 tokens" x={-250} y={-62} fontFamily={SANS} fontSize={32} fill={AMBER}/>
      <Txt text={'report 21: no match  report 22: no match  report 23:'} x={-250} y={16} fontFamily={MONO} fontSize={26} fill="#fff"/>
      <Rect ref={cutRef} x={140} y={16} width={8} height={56} fill={RED} opacity={0}/>
      <Txt text="cut off mid-count" x={470} y={16} fontFamily={SANS} fontSize={34} fontWeight={700} fill={RED} opacity={1}/>
      <Txt text="that run scored 0 of 3" x={470} y={-62} fontFamily={SANS} fontSize={28} fill={GREY}/>
    </Rect>);

  const ruleRef = createRef<Rect>();
  view.add(
    <Rect ref={ruleRef} y={225} width={1500} height={104} radius={20} fill={PANEL} stroke={ACCENT} lineWidth={6} opacity={0}
          layout alignItems="center" justifyContent="center" gap={50}>
      <Txt text="fits in the window?  paste it" fontFamily={SANS} fontSize={38} fontWeight={700} fill={GREEN}/>
      <Txt text="|" fontFamily={SANS} fontSize={38} fill={GREY}/>
      <Txt text="stops fitting?  use the harness" fontFamily={SANS} fontSize={38} fontWeight={700} fill={ACCENT}/>
    </Rect>);

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(at('F1', 'reports')); yield winRef().opacity(1, 0.5); yield barRef().width(used, 1.0);
  yield* go(at('F1', 'fits')); yield fitsRef().opacity(1, 0.4);
  yield* go(at('F2', 'pasting')); yield pasteRef().opacity(1, 0.5);
  yield* go(at('F2', 'zero')); yield toolsRef().opacity(1, 0.5);
  yield* go(at('F3', 'capped')); yield bugRef().opacity(1, 0.5);
  yield* go(at('F3', 'cut')); yield cutRef().opacity(1, 0.3);
  yield* go(at('F4', 'fits')); yield ruleRef().opacity(1, 0.5);
  yield* go(TL.end);
});
