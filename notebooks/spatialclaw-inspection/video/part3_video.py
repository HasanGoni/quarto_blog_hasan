"""SpatialClaw for Inspection, Part 3 video: three narrated chapters (Scene classes P3a, P3b, P3c).

Every number on screen comes from the real run (notebooks/spatialclaw-inspection/outputs/run_dev_output.txt,
trivial_baselines.txt, test_paths.txt). 'submit(1.56)' is the real val_00000 episode. Labelled on screen:
the scripted stand-in model used for the path tests.

Build (from notebooks/paper-explainers-2026-10-09/videos, module symlinked there with an ABSOLUTE target):
    VC=<repo>/notebooks/voice-clone ./make_chapter.sh part3_video.py P3a p3a clone warm
Colours: yellow is the subtitle highlighter only; content uses white, cyan and green/red accents.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from vidbase import NarratedScene

config.background_color = "#181818"
FONT, MONO = "DejaVu Sans", "DejaVu Sans Mono"
CYAN = "#6EC6FF"


def lab(s, size=26, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


def code(s, size=24, color=WHITE):
    return Text(s, font=MONO, font_size=size, color=color)


def card(w, h, color=GREY_B, fill=0.0):
    return RoundedRectangle(corner_radius=0.2, width=w, height=h, color=color, stroke_width=4, fill_color=color, fill_opacity=fill)


def tile(text, w, h, color, size=24, fill=0.08):
    c = card(w, h, color, fill)
    return VGroup(c, lab(text, size, WHITE, line_spacing=0.9).move_to(c))


# ------------------------------------------------------------------------------------------------------------- chapter a
class P3a(NarratedScene, Scene):
    SLUG = "p3a"
    LINES = {
        "A1": "The smallest loop has just **four moving parts**: the model, a kernel, the printed result, and a **budget**.",
        "A2": "The model writes **one cell**, the kernel runs it, and the output goes **back**.",
        "A3": "The variables stay **alive**, so the next cell can build on the last.",
        "A4": "The loop stops when a number is **submitted**, or when the budget is **spent**.",
    }

    def construct(self):
        model = tile("model", 2.6, 1.2, BLUE_C).move_to([-5.0, 2.6, 0])
        kernel = tile("kernel", 2.6, 1.2, GREY_B).move_to([-1.0, 2.6, 0])
        result = tile("printed\nresult", 2.6, 1.2, GREEN_C).move_to([3.0, 2.6, 0])
        budget = tile("budget:\n8 turns", 2.2, 1.2, CYAN).move_to([5.6, 0.4, 0]).scale(0.9)
        self.beat("A1", LaggedStart(*[FadeIn(m) for m in (model, kernel, result, budget)], lag_ratio=0.35), rt=2.2)

        a1 = Arrow(model.get_right(), kernel.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        a2 = Arrow(kernel.get_right(), result.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        back = CurvedArrow(result.get_bottom() + DOWN * 0.05, model.get_bottom() + DOWN * 0.05, angle=-TAU / 5, color=CYAN, stroke_width=5)
        cell_l = lab("one cell", 22, WHITE).next_to(a1, UP, buff=0.15)
        self.beat("A2", GrowArrow(a1), FadeIn(cell_l), rt=0.8, wait=False)
        self.play(GrowArrow(a2), run_time=0.7)
        self.play(Create(back), run_time=1.0)
        self.hold()

        shelf = VGroup(card(7.2, 1.0, CYAN, 0.06), code("img    mask    sizes    part_pixels", 22, CYAN)).move_to([-1.0, -0.9, 0])
        shelf_l = lab("variables stay alive between cells", 22, GREY_B).next_to(shelf, UP, buff=0.12)
        self.beat("A3", FadeIn(shelf), FadeIn(shelf_l), rt=1.0)

        sub = tile("submit(number)", 3.4, 1.0, GREEN_C).move_to([5.0, -0.9, 0]).scale(0.85)
        newb = tile("budget:\n0 turns left", 2.2, 1.2, RED_C).move_to(budget).scale(0.9)
        self.beat("A4", FadeIn(sub), Transform(budget, newb), rt=1.2)
        self.wait(1.0)
        self.save_timeline()


# ------------------------------------------------------------------------------------------------------------- chapter b
class P3b(NarratedScene, Scene):
    SLUG = "p3b"
    LINES = {
        "B1": "The model never says pass or fail. It only submits a **number**.",
        "B2": "A plain rule compares it with the limit, here **one point five percent**.",
        "B3": "No number means **review**, never a guess of pass.",
        "B4": "A scripted stand-in model tested **nine** ways an episode can end, on the real kernel.",
    }

    def construct(self):
        sub = VGroup(card(5.4, 1.5, BLUE_C, 0.08), VGroup(code("submit(1.56)", 32), lab("real episode: image val_00000", 19, GREY_B)).arrange(DOWN, buff=0.15)).move_to([-3.8, 2.4, 0])
        sub[1].move_to(sub[0])
        self.beat("B1", FadeIn(sub), rt=1.0)

        rule = VGroup(card(4.2, 1.5, GREY_B), lab("decide()\nordinary code", 26, WHITE, line_spacing=0.9)).move_to([2.4, 2.4, 0])
        rule[1].move_to(rule[0])
        arr = Arrow(sub.get_right(), rule.get_left(), buff=0.15, color=GREY_B, stroke_width=5)
        res = VGroup(card(5.0, 1.0, RED_C, 0.1), lab("FAIL   1.56 > 1.50", 28, RED_C, weight="BOLD")).move_to([2.4, 0.5, 0])
        res[1].move_to(res[0])
        self.beat("B2", GrowArrow(arr), FadeIn(rule), rt=0.9, wait=False)
        self.play(FadeIn(res), run_time=0.8)
        self.hold()

        self.play(FadeOut(VGroup(sub, rule, arr, res)), run_time=0.6)
        nonum = VGroup(card(4.2, 1.2, GREY_B), lab("no number\n(out of turns)", 26, WHITE, line_spacing=0.9)).move_to([-3.8, 2.2, 0])
        nonum[1].move_to(nonum[0])
        rev = VGroup(card(3.6, 1.2, GREEN_C, 0.12), lab("REVIEW", 36, GREEN_C, weight="BOLD")).move_to([3.6, 2.9, 0])
        rev[1].move_to(rev[0])
        guess = VGroup(card(3.6, 1.2, RED_C, 0.08), lab("guess: PASS", 30, RED_C)).move_to([3.6, 0.9, 0])
        guess[1].move_to(guess[0])
        strike = Line(guess.get_left() + RIGHT * 0.35, guess.get_right() + LEFT * 0.35, color=RED_C, stroke_width=7).move_to(guess)
        g1 = Arrow(nonum.get_right(), rev.get_left(), buff=0.15, color=GREEN_C, stroke_width=6)
        g2 = Arrow(nonum.get_right(), guess.get_left(), buff=0.15, color=RED_C, stroke_width=4)
        self.beat("B3", FadeIn(nonum), rt=0.6, wait=False)
        self.play(GrowArrow(g1), FadeIn(rev), run_time=0.8)
        self.play(GrowArrow(g2), FadeIn(guess), Create(strike), run_time=0.8)
        self.hold()

        self.play(FadeOut(VGroup(nonum, rev, guess, strike, g1, g2)), run_time=0.6)
        big = lab("9 / 9 paths ok", 60, GREEN_C, weight="BOLD").move_to([0, 2.5, 0])
        note = lab("scripted stand-in model  ·  real kernel, real timeout, real rule layer", 24, GREY_B).move_to([0, 1.4, 0])
        self.beat("B4", FadeIn(big), FadeIn(note), rt=1.0)
        self.wait(1.0)
        self.save_timeline()


# ------------------------------------------------------------------------------------------------------------- chapter c
def hbar(x0, y, value, total, w_full, color, h=0.55):
    return Rectangle(width=max(w_full * value / total, 0.02), height=h, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to([x0 + w_full * value / total / 2, y, 0])


class P3c(NarratedScene, Scene):
    SLUG = "p3c"
    LINES = {
        "C1": "Then twelve real images, with a local **twenty seven billion** parameter model.",
        "C2": "The agent matched the truth on **nine of twelve** decisions.",
        "C3": "But saying pass to everything also scores **nine of twelve**.",
        "C4": "The agent missed **one** failure, where that rule misses **three**.",
        "C5": "And every episode used nearly the **whole budget**, so eight turns is too tight.",
    }

    def construct(self):
        head = lab("12 real images  ·  DEV split  ·  local Qwen3.8-27B  ·  8 turns each", 24, GREY_B).move_to([0, 3.5, 0])
        self.beat("C1", FadeIn(head), rt=1.0)

        x0, wf = -3.3, 7.0
        t1 = lab("verdicts that match the truth", 26, WHITE, weight="BOLD").move_to([-2.0, 2.7, 0])
        n1 = lab("agent", 24, CYAN).move_to([-5.3, 1.9, 0]); n2 = lab("always PASS", 24, GREY_B).move_to([-5.3, 1.1, 0])
        b1 = hbar(x0, 1.9, 9, 12, wf, CYAN); v1 = lab("9 of 12", 24).next_to(b1, RIGHT, buff=0.2)
        b2 = hbar(x0, 1.1, 9, 12, wf, GREY_B); v2 = lab("9 of 12", 24).next_to(b2, RIGHT, buff=0.2)
        frame1 = Rectangle(width=wf, height=0.55, stroke_color=GREY_B, stroke_width=1).move_to([x0 + wf / 2, 1.9, 0])
        frame2 = Rectangle(width=wf, height=0.55, stroke_color=GREY_B, stroke_width=1).move_to([x0 + wf / 2, 1.1, 0])
        self.beat("C2", FadeIn(t1), FadeIn(n1), FadeIn(frame1), GrowFromEdge(b1, LEFT), FadeIn(v1), rt=1.2)
        self.beat("C3", FadeIn(n2), FadeIn(frame2), GrowFromEdge(b2, LEFT), FadeIn(v2), rt=1.2)

        t2 = lab("real failures called PASS (false PASS)", 26, WHITE, weight="BOLD").move_to([-1.6, 0.2, 0])
        m1 = lab("agent", 24, CYAN).move_to([-5.3, -0.6, 0]); m2 = lab("always PASS", 24, GREY_B).move_to([-5.3, -1.3, 0])
        c1 = hbar(x0, -0.6, 1, 3, 3.0, RED_C); c2 = hbar(x0, -1.3, 3, 3, 3.0, RED_C)
        cv1 = lab("1", 28, RED_C, weight="BOLD").next_to(c1, RIGHT, buff=0.2); cv2 = lab("3", 28, RED_C, weight="BOLD").next_to(c2, RIGHT, buff=0.2)
        self.beat("C4", FadeIn(t2), FadeIn(m1), FadeIn(m2), GrowFromEdge(c1, LEFT), GrowFromEdge(c2, LEFT), FadeIn(cv1), FadeIn(cv2), rt=1.4)

        self.play(FadeOut(VGroup(t1, n1, n2, b1, b2, v1, v2, frame1, frame2, t2, m1, m2, c1, c2, cv1, cv2)), run_time=0.7)
        base = -1.2
        ax = Line([-4.5, base, 0], [4.5, base, 0], color=GREY_B, stroke_width=3)
        bar8 = Rectangle(width=1.6, height=9 * 0.3, stroke_width=0, fill_color=RED_C, fill_opacity=0.85).move_to([2.2, base + 9 * 0.15, 0])
        bar7 = Rectangle(width=1.6, height=3 * 0.3, stroke_width=0, fill_color=CYAN, fill_opacity=0.85).move_to([-0.6, base + 3 * 0.15, 0])
        l8 = lab("turn 8:  9 episodes", 22).next_to(bar8, UP, buff=0.12); l7 = lab("turn 7:  3 episodes", 22).next_to(bar7, UP, buff=0.12)
        x8 = lab("submitted on turn 8", 20, GREY_B).move_to([2.2, base - 0.35, 0]); x7 = lab("turn 7", 20, GREY_B).move_to([-0.6, base - 0.35, 0])
        cap = lab("0 episodes submitted before turn 7", 24, WHITE).move_to([0, 3.0, 0])
        self.beat("C5", FadeIn(ax), GrowFromEdge(bar8, DOWN), GrowFromEdge(bar7, DOWN), FadeIn(l8), FadeIn(l7), FadeIn(x8), FadeIn(x7), FadeIn(cap), rt=1.4)
        self.wait(1.0)
        self.save_timeline()
