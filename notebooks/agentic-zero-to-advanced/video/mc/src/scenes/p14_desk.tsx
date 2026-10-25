import {makeScene2D, Rect, Txt, Node} from '@motion-canvas/2d';
import {createRef, all, Reference} from '@motion-canvas/core';
import tl from '../p14_desk/timeline.json';
import facts from '../facts_p14.json';
import {T, W, clock, Timeline} from '../lib/clock';
import {addSubtitles} from '../lib/subtitles';
import {BG, ACCENT, GREEN, RED, GREY, SANS, PANEL} from '../lib/theme';

const TL = tl as Timeline;
const at = (k: string, w: string) => T(TL, k, W(TL, k, w));
const win = facts.window.toLocaleString('en-US');

function chip(x: number, y: number, color: string, text: string) {
  const r = createRef<Rect>();
  const node = (
    <Rect ref={r} x={x} y={y} width={560} height={96} radius={20} fill={PANEL} stroke={color} lineWidth={6} opacity={0}
          layout alignItems="center" justifyContent="center">
      <Txt text={text} fontFamily={SANS} fontSize={36} fontWeight={700} fill="#fff"/>
    </Rect>);
  return {node, r};
}

export default makeScene2D(function* (view) {
  view.fill(BG);

  // the library: a wall of small books, far bigger than the desk
  const libRef = createRef<Rect>();
  const books: any[] = [];
  const cols = ['#4b5d73', '#5b6e84', '#3f5166', '#6a7d93'];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 12; c++)
    books.push(<Rect x={-330 + c * 60} y={-99 + r * 66} width={44} height={54} radius={5} fill={cols[(r + c) % 4]}/>);
  const library = (
    <Rect ref={libRef} x={430} y={-250} width={820} height={400} radius={26} fill={PANEL} stroke={GREY} lineWidth={5} opacity={0}>
      <Txt text="THE LIBRARY" y={-165} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
      <Node y={5}>{books}</Node>
      <Txt text="4,000-row table, 200 reports, policies" y={165} fontFamily={SANS} fontSize={28} fill={GREY}/>
    </Rect>);
  const deskRef = createRef<Rect>();
  const desk = (
    <Rect ref={deskRef} x={-560} y={-250} width={440} height={300} radius={26} fill={PANEL} stroke={ACCENT} lineWidth={8} opacity={0}
          layout direction="column" alignItems="center" justifyContent="center" gap={14}>
      <Txt text="THE DESK" fontFamily={SANS} fontSize={44} fontWeight={700} fill="#fff"/>
      <Txt text="the context window" fontFamily={SANS} fontSize={30} fill={ACCENT}/>
      <Txt text={`${win} tokens`} fontFamily={SANS} fontSize={40} fontWeight={700} fill="#fff"/>
    </Rect>);
  view.add(desk); view.add(library);

  const f1 = chip(-600, 70, RED, 'searches badly');
  const f2 = chip(0, 70, RED, 'carries too much');
  const f3 = chip(600, 70, RED, 'carries the wrong pages');
  const x1 = chip(-600, 200, GREEN, 'search by text');
  const x2 = chip(0, 200, GREEN, 'read in pages');
  const x3 = chip(600, 200, GREEN, 'pass paths, not files');
  [f1, f2, f3, x1, x2, x3].forEach(c => view.add(c.node));

  yield addSubtitles(view, TL);
  const go = clock();

  yield* go(at('D1', 'desk')); yield deskRef().opacity(1, 0.6);
  yield* go(at('D2', 'library')); yield libRef().opacity(1, 0.7);
  yield* go(at('D2', 'bigger')); yield all(libRef().stroke(RED, 0.5));
  yield* go(at('D3', 'searches')); yield f1.r().opacity(1, 0.5);
  yield* go(at('D3', 'much')); yield f2.r().opacity(1, 0.5);
  yield* go(at('D3', 'wrong')); yield f3.r().opacity(1, 0.5);
  yield* go(at('D4', 'text')); yield x1.r().opacity(1, 0.5);
  yield* go(at('D4', 'pages')); yield x2.r().opacity(1, 0.5);
  yield* go(at('D4', 'paths')); yield x3.r().opacity(1, 0.5);
  yield* go(TL.end);
});
