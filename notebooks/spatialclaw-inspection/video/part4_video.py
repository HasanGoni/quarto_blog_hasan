"""SpatialClaw for Inspection, Part 4 video: three narrated chapters (Scene classes P4a, P4b, P4c).

Every number on screen comes from outputs/part4_summary.txt (real runs, same 12 DEV images, each set-up run twice).

Build (from notebooks/paper-explainers-2026-10-09/videos, module symlinked there with an ABSOLUTE target):
    VC=<repo>/notebooks/voice-clone ./make_chapter.sh part4_video.py P4a p4a clone warm
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



def hbar(x0, y, value, total, w_full, color, h=0.45):
    return Rectangle(width=max(w_full * value / total, 0.02), height=h, stroke_width=0, fill_color=color, fill_opacity=0.9).move_to([x0 + w_full * value / total / 2, y, 0])


class P4a(NarratedScene, Scene):
    SLUG = "p4a"
    LINES = {
        "A1": "Part four gives the loop **eyes** and **hands**.",
        "A2": "A new function, **show**, puts a picture in front of the model.",
        "A3": "Four small helper tools do the **boring** steps, like outlining the part.",
        "A4": "The rule that decides pass or fail is **unchanged**.",
    }

    def construct(self):
        model = tile("model", 2.4, 1.1, BLUE_C).move_to([-4.6, 2.6, 0])
        kernel = tile("kernel", 2.4, 1.1, GREY_B).move_to([0.2, 2.6, 0])
        back = tile("text", 2.0, 1.1, GREEN_C).move_to([4.6, 2.6, 0])
        a1 = Arrow(model.get_right(), kernel.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        a2 = Arrow(kernel.get_right(), back.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        self.beat("A1", LaggedStart(*[FadeIn(m) for m in (model, kernel, back, a1, a2)], lag_ratio=0.25), rt=1.8)

        pic = tile("picture\nfrom show()", 2.6, 1.2, CYAN, fill=0.15).move_to([4.6, 1.0, 0])
        pa = Arrow(kernel.get_bottom(), pic.get_left(), buff=0.1, color=CYAN, stroke_width=5)
        ret = CurvedArrow(back.get_top() + UP * 0.05, model.get_top() + UP * 0.05, angle=TAU / 20, color=CYAN, stroke_width=4)
        self.beat("A2", GrowArrow(pa), FadeIn(pic), rt=1.0, wait=False)
        self.play(Create(ret), run_time=1.0)
        self.hold()

        tools = VGroup(card(10.4, 1.2, BLUE_C, 0.06), code("part_mask   label_blobs   overlay   crop", 24, WHITE)).move_to([-0.5, -0.45, 0])
        tools[1].move_to(tools[0])
        tl = lab("numpy and scipy only  ·  building blocks, not a detector", 21, GREY_B).next_to(tools, DOWN, buff=0.12)
        self.beat("A3", FadeIn(tools), FadeIn(tl), rt=1.2)

        rule = tile("decide(): ordinary code\nPASS / FAIL / REVIEW", 6.4, 1.6, RED_C, size=28).move_to([0, 0.8, 0])
        self.play(FadeOut(VGroup(model, kernel, back, a1, a2, pic, pa, ret, tools, tl)), run_time=0.6)
        self.beat("A4", FadeIn(rule), rt=1.0)
        self.wait(1.0)
        self.save_timeline()


class P4b(NarratedScene, Scene):
    SLUG = "p4b"
    LINES = {
        "B1": "Three set-ups on the same twelve images: **text only**, plus **pictures**, and plus pictures and **tools**.",
        "B2": "Then I ran each one **twice**.",
        "B3": "With tools, the mean error was **two point six** points in the first run.",
        "B4": "In the second run, the same set-up scored **zero point two six**.",
        "B5": "Same model, same images. The noise is as big as the **differences** I wanted to measure.",
    }

    def construct(self):
        title = lab("mean error of the void ratio, percentage points", 24, GREY_B).move_to([0, 3.4, 0])
        x0, wf, tot = -3.2, 7.4, 3.0
        names = [("A  text only", GREY_B, 0.66, 0.77), ("B  + show()", GREY_B, 0.90, 0.91), ("C  + show() + tools", CYAN, 2.63, 0.26)]
        ys = [2.4, 1.0, -0.4]
        labels = [lab(n, 22, c).move_to([-5.6, y - 0.3, 0]).align_to([-6.6, 0, 0], LEFT) for (n, c, _, _), y in zip(names, ys)]
        self.beat("B1", FadeIn(title), *[FadeIn(l) for l in labels], rt=1.6)

        r1 = [hbar(x0, y, v1, tot, wf, c) for (n, c, v1, v2), y in zip(names, ys)]
        v1l = [lab(f"{v1:.2f}", 21).next_to(b, RIGHT, buff=0.15) for b, (_, _, v1, _) in zip(r1, names)]
        self.play(*[GrowFromEdge(b, LEFT) for b in r1], *[FadeIn(t) for t in v1l], run_time=1.0)
        r2 = [hbar(x0, y - 0.55, v2, tot, wf, c, h=0.35).set_opacity(0.55) for (n, c, v1, v2), y in zip(names, ys)]
        v2l = [lab(f"{v2:.2f}", 19, GREY_B).next_to(b, RIGHT, buff=0.15) for b, (_, _, _, v2) in zip(r2, names)]
        k = lab("solid = run 1   ·   faded = run 2", 20, GREY_B).move_to([0, -1.55, 0])
        self.beat("B2", *[GrowFromEdge(b, LEFT) for b in r2], *[FadeIn(t) for t in v2l], FadeIn(k), rt=1.4)

        ring = SurroundingRectangle(VGroup(r1[2], v1l[2]), color=WHITE, buff=0.12, stroke_width=3)
        self.beat("B3", Create(ring), rt=0.8)
        ring2 = SurroundingRectangle(VGroup(r2[2], v2l[2]), color=WHITE, buff=0.12, stroke_width=3)
        self.beat("B4", ReplacementTransform(ring, ring2), rt=0.8)
        self.beat("B5", FadeOut(ring2), rt=0.8)
        self.wait(1.0)
        self.save_timeline()


class P4c(NarratedScene, Scene):
    SLUG = "p4c"
    LINES = {
        "C1": "Over both runs, tools gave the **lowest** typical error, and about **forty percent** fewer tokens.",
        "C2": "But one run submitted **twenty nine percent**, for a part whose true value is **zero point six five**.",
        "C3": "The model had printed a warning, and still submitted on its **last** turn.",
        "C4": "So I cannot say tools are **more accurate**. Only that the typical case got better.",
        "C5": "Next: a plausibility check in the rule layer, and a real **sandbox**.",
    }

    def construct(self):
        title = lab("median error over both runs, percentage points", 24, GREY_B).move_to([0, 3.4, 0])
        x0, wf, tot = -3.0, 7.0, 0.30
        rows = [("A  text only", GREY_B, 0.21, 2.2), ("B  + show()", GREY_B, 0.25, 1.2), ("C  + tools", CYAN, 0.16, 0.2)]
        objs = []
        for n, c, v, y in rows:
            b = hbar(x0, y, v, tot, wf, c, h=0.5)
            objs += [lab(n, 22, c).move_to([-5.5, y, 0]), b, lab(f"{v:.2f}", 22).next_to(b, RIGHT, buff=0.15)]
        self.beat("C1", FadeIn(title), *[FadeIn(o) for o in objs], rt=2.0)

        self.play(FadeOut(VGroup(title, *objs)), run_time=0.6)
        big = lab("28.99 %", 84, RED_C, weight="BOLD").move_to([-3.2, 1.8, 0])
        small = lab("true value: 0.65 %", 34, WHITE).move_to([-3.2, 0.4, 0])
        tag = lab("one run, one image, condition C", 22, GREY_B).move_to([-3.2, -0.5, 0])
        self.beat("C2", FadeIn(big), FadeIn(small), FadeIn(tag), rt=1.4)

        turn = tile("turn 14 of 14", 3.8, 1.0, RED_C, size=28).move_to([3.6, 1.8, 0])
        warn = code("frac part diff>20: 0.317", 22, GREY_B).move_to([3.6, 0.4, 0])
        self.beat("C3", FadeIn(warn), FadeIn(turn), rt=1.4)

        self.play(FadeOut(VGroup(big, small, tag, turn, warn)), run_time=0.6)
        self.beat("C4", FadeIn(lab("typical case: better\nworst case: not shown to be safe", 34, WHITE, line_spacing=1.1).move_to([0, 1.6, 0])), rt=1.4)
        nxt = VGroup(tile("plausibility check", 4.6, 1.0, GREEN_C, size=24), tile("real sandbox", 3.6, 1.0, BLUE_C, size=24)).arrange(RIGHT, buff=0.6).move_to([0, -0.4, 0])
        self.beat("C5", FadeIn(nxt), rt=1.0)
        self.wait(1.0)
        self.save_timeline()
