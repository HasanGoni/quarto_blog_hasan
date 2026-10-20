"""SpatialClaw for Inspection, Part 1 video: four narrated chapters (Scene classes Sc1..Sc4).

Real material: the numbers come from the paper's Table 2 (Gemma4-31B, average of 20 benchmarks, BLINK row) and from
my own run (outputs/smoke_kernel.txt). Labelled on screen as a stand-in: the agent is hard-coded cells, the pad is
synthetic, the 'segmenter' is a threshold.

Build (from notebooks/paper-explainers-2026-10-09/videos, with this file symlinked there as spatialclaw_video.py):
    ./make_chapter.sh spatialclaw_video.py Sc1 sc1 clone warm      # -> chapters/sc1.warm.mp4
Colours: yellow is the subtitle highlighter only; content uses white, cyan #6EC6FF and green/red accents.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))          # vidbase.py lives next to the symlink
sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from vidbase import NarratedScene

config.background_color = "#181818"
FONT, MONO = "DejaVu Sans", "DejaVu Sans Mono"
CYAN = "#6EC6FF"
ASSETS = Path(__file__).resolve().parent / "assets"
if not ASSETS.exists():                                             # when run through the symlink in videos/
    ASSETS = Path.home() / "workspace/projects/git_data/quarto_blog_hasan/notebooks/spatialclaw-inspection/video/assets"


def lab(s, size=26, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


def code(s, size=24, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def card(w, h, color=GREY_B, fill=0.0):
    return RoundedRectangle(corner_radius=0.2, width=w, height=h, color=color, stroke_width=4, fill_color=color, fill_opacity=fill)


# --------------------------------------------------------------------------------------------------------------- scene 1
class Sc1(NarratedScene, Scene):
    SLUG = "sc1"
    LINES = {
        "A1": "Most agents with tools get a **menu**: pick a tool, get an answer, pick another.",
        "A2": "Spatial Claw gives the model a **notebook** instead, one that remembers everything.",
        "A3": "The model writes **one cell**, and looks at what came out.",
        "A4": "Then it writes the next cell using the **same variables**. Nothing is forgotten between steps, and that is the **whole trick**.",
    }

    def construct(self):
        # A1: the menu
        menu = card(4.6, 3.0, GREY_B).move_to([-3.6, 1.4, 0])
        mt = lab("MENU", 26, CYAN, weight="BOLD").move_to(menu.get_top() + DOWN * 0.45)
        rows = VGroup(*[code(t, 24) for t in ["segment(image)", "measure(mask)", "crop(box)"]]).arrange(DOWN, buff=0.3).move_to(menu.get_center() + DOWN * 0.25)
        ans = card(3.2, 1.2, GREY_B).move_to([4.0, 1.4, 0])
        ans_t = lab("JSON answer", 24).move_to(ans)
        arr = Arrow(menu.get_right(), ans.get_left(), buff=0.15, color=GREY_B, stroke_width=5)
        self.beat("A1", FadeIn(menu), FadeIn(mt), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), rt=1.6, wait=False)
        self.play(GrowArrow(arr), FadeIn(ans), FadeIn(ans_t), run_time=1.0)
        self.hold()

        # A2: the notebook
        self.play(FadeOut(VGroup(menu, mt, rows, ans, ans_t, arr)), run_time=0.6)
        nb = card(10.0, 3.9, CYAN, 0.06).move_to([0, 1.3, 0])
        nbt = lab("a notebook that remembers everything", 28, CYAN, weight="BOLD").move_to(nb.get_top() + DOWN * 0.45)
        self.beat("A2", FadeIn(nb), FadeIn(nbt), rt=1.0)

        # A3: one cell, look at the output
        c1 = code("In [1]:  mask = segment(img)", 26).move_to([-1.2, 1.95, 0]).align_to(nb, LEFT).shift(RIGHT * 0.6)
        o1 = code("Out:      mask  (200, 200) bool", 24, GREEN_C).next_to(c1, DOWN, buff=0.22).align_to(c1, LEFT)
        self.beat("A3", FadeIn(c1), rt=0.8, wait=False)
        self.play(FadeIn(o1), run_time=0.8)
        self.hold()

        # A4: next cell reuses the variable
        c2 = code("In [2]:  area = mask.sum()", 26).next_to(o1, DOWN, buff=0.45).align_to(c1, LEFT)
        o2 = code("Out:      area  int", 24, GREEN_C).next_to(c2, DOWN, buff=0.22).align_to(c1, LEFT)
        link = CurvedArrow(c1.get_right() + LEFT * 2.0 + UP * 0.1, c2.get_right() + LEFT * 1.9 + UP * 0.05, angle=-TAU / 4, color=CYAN, stroke_width=5).shift(RIGHT * 2.4)
        keep = lab("same mask,\nstill alive", 22, CYAN, line_spacing=0.9).next_to(link, RIGHT, buff=0.3)
        self.beat("A4", FadeIn(c2), rt=0.8, wait=False)
        self.play(Create(link), FadeIn(keep), FadeIn(o2), run_time=1.2)
        self.hold()
        self.wait(1.0)
        self.save_timeline()


# --------------------------------------------------------------------------------------------------------------- scene 2
class Sc2(NarratedScene, Scene):
    SLUG = "sc2"
    LINES = {
        "B1": "Here is a real run, with a **stand-in** agent and a synthetic solder pad.",
        "B2": "Cell one finds the dark voids. Cell two reuses them and counts **three voids**.",
        "B3": "Together they cover **6.43 percent** of the pad.",
        "B4": "Cell three makes a mistake, and the notebook answers with a **traceback**.",
        "B5": "Cell four fixes it, because the earlier variables are still **alive**.",
        "B6": "A failed step costs **one cell**, not the whole run.",
    }

    def construct(self):
        pad = ImageMobject(str(ASSETS / "pad.png")).set_height(3.7).move_to([-4.2, 1.35, 0])
        frame = Rectangle(width=pad.width, height=pad.height, color=GREY_B, stroke_width=3).move_to(pad)
        cap = lab("stand-in: synthetic pad,\nthreshold instead of a segmenter", 19, GREY_B, line_spacing=0.9).move_to([-4.2, -0.95, 0])
        self.beat("B1", FadeIn(pad), FadeIn(frame), FadeIn(cap), rt=1.2)

        ov = ImageMobject(str(ASSETS / "voids_overlay.png")).set_height(3.7).move_to(pad)
        r1 = VGroup(code("voids = (img < 130) & pad", 22), code("lab, n = ndimage.label(voids)", 22)).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        o1 = code("void count: 3   sizes: [437, 1005, 193]", 21, GREEN_C)
        right1 = VGroup(r1, o1).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([3.1, 2.1, 0])
        self.beat("B2", FadeIn(ov), FadeIn(right1), rt=1.2)

        big = lab("void ratio:  6.43 %", 40, CYAN, weight="BOLD").move_to([3.1, 0.55, 0])
        self.beat("B3", FadeIn(big), rt=0.8)

        self.play(FadeOut(VGroup(right1, big)), run_time=0.5)
        c3 = code("print(largest_void_area)", 24).move_to([3.1, 2.3, 0])
        tb = VGroup(card(6.4, 1.25, RED_C, 0.1), lab("NameError: name 'largest_void_area'\nis not defined", 22, RED_C, line_spacing=0.9)).move_to([3.1, 0.9, 0])
        self.beat("B4", FadeIn(c3), FadeIn(tb), rt=0.9)

        self.play(FadeOut(VGroup(c3, tb)), run_time=0.5)
        c4 = code("largest_void_area = int(max(sizes))", 22).move_to([3.1, 2.3, 0])
        o4 = code("largest void px: 1005", 24, GREEN_C).next_to(c4, DOWN, buff=0.3).align_to(c4, LEFT)
        alive = lab("still alive:  sizes   voids   pad   lab", 22, CYAN).next_to(o4, DOWN, buff=0.45).align_to(c4, LEFT)
        self.beat("B5", FadeIn(c4), FadeIn(o4), FadeIn(alive), rt=1.0)

        self.play(FadeOut(VGroup(c4, o4, alive)), run_time=0.5)
        msg = VGroup(card(6.6, 1.5, CYAN, 0.08), lab("a failed step costs ONE cell", 32, WHITE, weight="BOLD")).move_to([3.1, 1.4, 0])
        self.beat("B6", FadeIn(msg), rt=0.9)
        self.wait(1.0)
        self.save_timeline()


# --------------------------------------------------------------------------------------------------------------- scene 3
def bar(x, base, h, w, color):
    return Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to([x, base + h / 2, 0])


class Sc3(NarratedScene, Scene):
    SLUG = "sc3"
    LINES = {
        "C1": "The paper reports **59.9 percent** on average, with the same model throughout.",
        "C2": "With no tools the score is **53.4**, and a plain tool-call agent reaches **56.7**.",
        "C3": "So the notebook idea is worth about **three points**, and the tools themselves about **three more**.",
        "C4": "And on the Blink benchmark, the no-tool model scored **75.7** against 73.4 with tools.",
        "C5": "When the question is mostly recognition, tools can **hurt**.",
    }

    def construct(self):
        head = lab("SpatialClaw paper, Table 2: Gemma4-31B, average of 20 benchmarks   (axis starts at 50)", 20, GREY_B).move_to([0, 3.55, 0])
        base, w = -1.0, 2.2
        xs = [-3.8, 0.0, 3.8]
        vals = [53.4, 56.7, 59.9]
        names = ["no tools", "tool-call agent", "SpatialClaw"]
        cols = [GREY_B, BLUE_C, GREEN_C]
        hs = [(v - 50) * 0.35 for v in vals]
        bars = [bar(x, base, h, w, c) for x, h, c in zip(xs, hs, cols)]
        vl = [lab(f"{v}", 30, WHITE, weight="BOLD").next_to(b, UP, buff=0.12) for v, b in zip(vals, bars)]
        nl = [lab(n, 22, WHITE).move_to([x, base - 0.4, 0]) for n, x in zip(names, xs)]
        axis = Line([-5.6, base, 0], [5.6, base, 0], color=GREY_B, stroke_width=3)

        self.add(head, axis)
        self.beat("C1", GrowFromEdge(bars[2], DOWN), FadeIn(vl[2]), FadeIn(nl[2]), rt=1.2)

        self.beat("C2", GrowFromEdge(bars[0], DOWN), FadeIn(vl[0]), FadeIn(nl[0]), rt=0.9, wait=False)
        self.play(GrowFromEdge(bars[1], DOWN), FadeIn(vl[1]), FadeIn(nl[1]), run_time=0.9)
        self.hold()

        def gap(i, j, text, color):
            y1, y2 = base + hs[i], base + hs[j]
            xm = (xs[i] + xs[j]) / 2
            a = Arrow([xm, y1 + 0.05, 0], [xm, y2 - 0.05, 0], buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.3)
            t = lab(text, 24, color).next_to(a, UP, buff=0.35)
            return VGroup(a, t)
        g1, g2 = gap(0, 1, "+3.3  tools", CYAN), gap(1, 2, "+3.2  notebook", GREEN_C)
        g2.shift(UP * 0.0)
        self.beat("C3", FadeIn(g1), rt=0.8, wait=False)
        self.play(FadeIn(g2), run_time=0.8)
        self.hold()

        self.play(FadeOut(VGroup(*bars, *vl, *nl, g1, g2)), run_time=0.6)
        head2 = lab("BLINK benchmark (same paper, Table 2)   (axis starts at 70)", 20, GREY_B)
        head2.move_to(head)
        self.play(Transform(head, head2), run_time=0.3)
        bh = [(75.7 - 70) * 0.5, (73.4 - 70) * 0.5]
        bx = [-2.6, 2.6]
        b2 = [bar(bx[0], base, bh[0], 2.6, GREY_B), bar(bx[1], base, bh[1], 2.6, GREEN_C)]
        v2 = [lab("75.7", 32, WHITE, weight="BOLD").next_to(b2[0], UP, buff=0.12), lab("73.4", 32, WHITE, weight="BOLD").next_to(b2[1], UP, buff=0.12)]
        n2 = [lab("no tools", 24).move_to([bx[0], base - 0.4, 0]), lab("SpatialClaw", 24).move_to([bx[1], base - 0.4, 0])]
        self.beat("C4", *[GrowFromEdge(b, DOWN) for b in b2], *[FadeIn(x) for x in v2 + n2], rt=1.2)

        hurt = VGroup(card(4.2, 1.0, RED_C, 0.1), lab("tools lower the score", 26, RED_C, weight="BOLD")).move_to([0, 2.6, 0])
        self.beat("C5", FadeIn(hurt), rt=0.9)
        self.wait(1.0)
        self.save_timeline()


# --------------------------------------------------------------------------------------------------------------- scene 4
class Sc4(NarratedScene, Scene):
    SLUG = "sc4"
    LINES = {
        "D1": "For inspection we **keep** the notebook, the loop, and the images shown back to the model.",
        "D2": "We **drop** depth, video and the cluster, because an X-ray is not a scene.",
        "D3": "We swap the free-form answer for a **typed verdict**.",
        "D4": "And when the agent runs out of steps, it must say **review**, never a guess of pass.",
        "D5": "Next time we write the smallest loop, and we **run it** before we claim anything.",
    }

    def construct(self):
        kh = lab("KEEP", 34, GREEN_C, weight="BOLD").move_to([-3.6, 3.2, 0])
        keeps = VGroup(*[lab(t, 26) for t in ["persistent notebook", "plan, cell, feedback loop", "images shown back to the model"]]).arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(kh, DOWN, buff=0.5).align_to(kh, LEFT).shift(LEFT * 1.2)
        self.beat("D1", FadeIn(kh), LaggedStart(*[FadeIn(k) for k in keeps], lag_ratio=0.35), rt=1.6)

        dh = lab("DROP", 34, RED_C, weight="BOLD").move_to([3.8, 3.2, 0])
        drops = VGroup(*[lab(t, 26) for t in ["depth and 3D", "video code", "the cluster (SLURM)"]]).arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(dh, DOWN, buff=0.5).align_to(dh, LEFT).shift(LEFT * 0.6)
        self.beat("D2", FadeIn(dh), LaggedStart(*[FadeIn(d) for d in drops], lag_ratio=0.35), rt=1.6)

        self.play(FadeOut(VGroup(kh, keeps, dh, drops)), run_time=0.6)
        old = VGroup(card(3.6, 1.0, GREY_B), code('ReturnAnswer("B")', 24)).move_to([-3.8, 2.4, 0])
        vl = VGroup(*[code(t, 22) for t in ['Verdict(status="FAIL",', '        void_ratio=6.43,', '        evidence=[img_3])']]).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        new = VGroup(card(7.2, 1.9, CYAN, 0.06), vl).move_to([2.8, 2.0, 0])
        arr = Arrow(old.get_right(), new.get_left(), buff=0.15, color=GREY_B, stroke_width=5)
        tag = lab("illustrative: values from the stand-in run", 18, GREY_B).next_to(new, DOWN, buff=0.2)
        self.beat("D3", FadeIn(old), rt=0.6, wait=False)
        self.play(GrowArrow(arr), FadeIn(new), FadeIn(tag), run_time=1.0)
        self.hold()

        self.play(FadeOut(VGroup(old, new, arr, tag)), run_time=0.6)
        out = VGroup(card(3.8, 1.1, GREY_B), lab("out of steps", 28)).move_to([-4.2, 2.0, 0])
        rev = VGroup(card(3.4, 1.1, GREEN_C, 0.12), lab("REVIEW", 34, GREEN_C, weight="BOLD")).move_to([4.0, 2.7, 0])
        guess = VGroup(card(3.4, 1.1, RED_C, 0.08), lab("guess: PASS", 30, RED_C)).move_to([4.0, 0.9, 0])
        cross = Line(guess.get_left() + RIGHT * 0.3, guess.get_right() + LEFT * 0.3, color=RED_C, stroke_width=7).move_to(guess)
        a1 = Arrow(out.get_right(), rev.get_left(), buff=0.15, color=GREEN_C, stroke_width=6)
        a2 = Arrow(out.get_right(), guess.get_left(), buff=0.15, color=RED_C, stroke_width=4)
        self.beat("D4", FadeIn(out), rt=0.6, wait=False)
        self.play(GrowArrow(a1), FadeIn(rev), run_time=0.9)
        self.play(GrowArrow(a2), FadeIn(guess), Create(cross), run_time=0.9)
        self.hold()

        self.play(FadeOut(VGroup(out, rev, guess, cross, a1, a2)), run_time=0.6)
        nxt = VGroup(card(10.0, 1.8, CYAN, 0.08), lab("Part 3: the smallest loop, run for real", 36, WHITE, weight="BOLD")).move_to([0, 1.7, 0])
        self.beat("D5", FadeIn(nxt), rt=1.0)
        self.wait(1.0)
        self.save_timeline()
