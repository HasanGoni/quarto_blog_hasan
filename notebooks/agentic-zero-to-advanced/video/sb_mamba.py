"""Chapter 4 (Manim): how Mamba reads an image, on the REAL lesion (paper Fig.1 image, 16x16 cells). The memory decay values are illustrative (a = 0.8, and a made-up
'selective' rule), labelled on screen.   Build:  ./make_chapter.sh sb_mamba.py MambaChapter sb_mamba clone warm"""
from pathlib import Path
import numpy as np
from PIL import Image
from manim import *  # noqa: F403
from vidbase import NarratedScene
from sb_voice import MAMBA

config.background_color = "#181818"
A = Path(__file__).resolve().parent / "assets"
CY, GR, AM = "#6EC6FF", "#5BD18B", "#F2A93B"
RGB = np.asarray(Image.open(A / "lesion_256.png").convert("RGB").resize((16, 16), Image.BOX))
DARK = 1 - RGB.mean(2) / 255.0
ROW = 7                                                         # an image row that crosses the lesion
T_ = lambda s, **k: Text(s, font="DejaVu Sans", **k)


def grid_img(h, pos):
    m = ImageMobject(RGB).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).set(height=h).move_to(pos)
    return m


class MambaChapter(NarratedScene, Scene):
    SLUG, LINES = "sb_mamba", MAMBA

    def bars(self, hs, color, x0=0.0, w=0.34, base=0.0, k=0.42):
        return VGroup(*[Rectangle(width=w, height=max(h * k, 0.02), stroke_width=0, fill_color=color, fill_opacity=0.9).move_to([x0 + w * 1.12 * (i + 0.5), base + max(h * k, 0.02) / 2, 0]) for i, h in enumerate(hs)])

    def construct(self):
        # ---- real data for the toy memory
        x = DARK[ROW]                                            # 16 values: darkness of each cell along the row
        h_plain = []; h = 0.0
        for v in x:
            h = 0.8 * h + v; h_plain.append(h)
        a_sel = [0.9] + [float(np.clip(0.9 - 2.5 * abs(x[i] - x[i - 1]), 0.1, 0.95)) for i in range(1, 16)]
        h_sel = []; h = 0.0
        for a, v in zip(a_sel, x):
            h = a * h + v; h_sel.append(h)

        title = T_("Mamba: a memory that walks along a sequence", font_size=24, color=CY).move_to([2.3, 3.7, 0])
        grid = grid_img(3.7, [-4.3, 1.2, 0])
        gcap = T_("real lesion image, 16 × 16 cells", font_size=22, color=GREY_B).next_to(grid, DOWN, buff=0.15)
        cell = 3.7 / 16
        rowbox = Rectangle(width=3.7, height=cell, stroke_color=CY, stroke_width=4).move_to(grid.get_center() + UP * (3.7 / 2 - cell * (ROW + 0.5)))
        strip = VGroup(*[Square(0.38, stroke_width=1, stroke_color=GREY_B, fill_color=rgb_to_color(RGB[ROW, i] / 255.0), fill_opacity=1).move_to([0.19 + 0.38 * i + 0.0, 2.45, 0]) for i in range(16)])
        s_arrow = Arrow(rowbox.get_right(), strip.get_left() + LEFT * 0.1, buff=0.1, color=CY, stroke_width=4)
        base = Line([0.0, 0.0, 0], [6.5, 0.0, 0], color=GREY_B, stroke_width=2)
        mlab = T_("memory h  (bar height = how much is remembered)", font_size=20, color=GREY_B).move_to([3.0, -0.35, 0])
        slab = T_("one image row, read left to right", font_size=20, color=GREY_B).move_to([3.1, 2.85, 0])

        # M1
        self.beat("M1", FadeIn(title), FadeIn(grid), FadeIn(gcap), rt=1.0, wait=False)
        self.wait_word("sequence")
        self.play(Create(rowbox), FadeIn(strip, lag_ratio=0.05), GrowArrow(s_arrow), FadeIn(slab), run_time=1.6)
        self.wait_word("memory")
        self.play(Create(base), FadeIn(mlab), run_time=0.8)
        self.hold()
        # M2
        f1 = T_("h(t) = a · h(t−1) + x(t)    a = 0.8 < 1: the past fades", font_size=22, color=WHITE).move_to([2.9, 3.25, 0])
        pointer = Triangle(fill_color=CY, fill_opacity=1, stroke_width=0).scale(0.12).rotate(PI).move_to([0.19, 2.1, 0])
        b1 = self.bars(h_plain, CY)
        self.beat("M2", wait=False)
        self.wait_word("fades")
        self.play(FadeIn(f1), FadeIn(pointer), run_time=0.8)
        self.wait_word("added")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in b1], lag_ratio=0.3), pointer.animate.move_to([0.19 + 0.38 * 15, 2.1, 0]), run_time=3.2)
        self.hold()
        # M3
        f2 = T_("h(t) = a(x)·h(t−1) + b(x)·x(t)    a, b: from the content", font_size=22, color=WHITE).move_to([2.9, 3.25, 0])
        note = T_("illustrative rule: forget more when the content changes", font_size=20, color=AM).move_to([3.0, -0.8, 0])
        b2 = self.bars(h_sel, AM)
        self.beat("M3", wait=False)
        self.wait_word("content")
        self.play(Transform(f1, f2), Transform(b1, b2), FadeIn(note), run_time=1.6)
        self.hold()
        # M4: four orders
        h4, y4 = 2.0, 2.45
        xs = [-5.0, -1.65, 1.7, 5.05]
        mini = [grid_img(h4, [x_, y4, 0]) for x_ in xs]
        names = ["→ rows", "↓ columns", "← rows reversed", "↑ columns reversed"]
        labs = [T_(n, font_size=24, color=WHITE).move_to([x_, y4 - h4 / 2 - 0.3, 0]) for n, x_ in zip(names, xs)]
        cols = [CY, GR, AM, "#E58FFF"]

        def paths(i, x_):
            out = VGroup(); s = h4 / 16 * 2.5
            for k in range(5):
                f = -h4 / 2 + 0.2 + k * (h4 - 0.4) / 4
                if i in (0, 2):
                    a_, b_ = [x_ - h4 / 2 + 0.1, y4 + f, 0], [x_ + h4 / 2 - 0.1, y4 + f, 0]
                else:
                    a_, b_ = [x_ + f, y4 - h4 / 2 + 0.1, 0], [x_ + f, y4 + h4 / 2 - 0.1, 0]
                if i in (2, 3):
                    a_, b_ = b_, a_
                out.add(Arrow(a_, b_, buff=0, color=cols[i], stroke_width=4, max_tip_length_to_length_ratio=0.12))
            return out
        P4 = [paths(i, x_) for i, x_ in enumerate(xs)]
        self.beat("M4", FadeOut(Group(title, grid, gcap, rowbox, strip, s_arrow, base, mlab, slab, f1, pointer, b1, note)), rt=0.8, wait=False)
        self.wait_word("four")
        self.play(*[FadeIn(m) for m in mini], *[FadeIn(l) for l in labs], run_time=0.8)
        self.wait_word("rows")
        self.play(FadeIn(P4[0]), run_time=0.5)
        self.wait_word("columns")
        self.play(FadeIn(P4[1]), run_time=0.5)
        self.wait_word("reversed")
        self.play(FadeIn(P4[2]), FadeIn(P4[3]), run_time=0.6)
        self.hold()
        # M5: merge
        merged = grid_img(1.7, [0, -0.55, 0])
        arrows = VGroup(*[Arrow([x_, y4 - h4 / 2 - 0.55, 0], merged.get_top() + UP * 0.05 + RIGHT * (x_ - 0) * 0.06, buff=0.05, color=c, stroke_width=4) for x_, c in zip(xs, cols)])
        mlab2 = T_("each cell has now been read from 4 sides; the 4 results are merged", font_size=26, color=CY).move_to([0, -1.65, 0])
        self.beat("M5", wait=False)
        self.wait_word("four")
        self.play(FadeIn(arrows), FadeIn(merged), run_time=1.2)
        self.wait_word("merged")
        self.play(FadeIn(mlab2), run_time=0.6)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
