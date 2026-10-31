"""Chapter 6 (Manim): what happens to ONE real image. Real: the lesion image (paper Fig.1), real pre-trained SAM ViT-B features (PCA colours), the mask printed in the paper,
and our model's output after 30 steps on this single image (memorisation). Stand-in: the Mamba weights are random, so only shapes are shown.
Build:  ./make_chapter.sh sb_journey.py JourneyChapter sb_journey clone warm"""
from pathlib import Path
from manim import *  # noqa: F403
from vidbase import NarratedScene
from sb_voice import JOURNEY

config.background_color = "#181818"
A = Path(__file__).resolve().parent / "assets"
CY, GR, AM = "#6EC6FF", "#5BD18B", "#F2A93B"
T_ = lambda s, **k: Text(s, font="DejaVu Sans", **k)


def img(name, h, pos, nearest=False):
    m = ImageMobject(str(A / name)).set(height=h).move_to(pos)
    return m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]) if nearest else m


class JourneyChapter(NarratedScene, Scene):
    SLUG, LINES = "sb_journey", JOURNEY

    def construct(self):
        lesion = img("lesion_256.png", 3.6, [0, 1.2, 0])
        cap = T_("the lesion printed in the paper's Figure 1  ·  256 × 256", font_size=24, color=GREY_B).move_to([0, -0.95, 0])
        self.beat("J1", FadeIn(lesion), FadeIn(cap), rt=1.0)
        # J2: SAM features
        xs = [-1.2, 2.0, 5.2]
        maps = [img(f"sam_S{b}_pca.png", 2.7, [x, 1.4, 0], nearest=True) for b, x in zip((6, 9, 12), xs)]
        labs = [T_(f"block {b}", font_size=26, color=WHITE).move_to([x, 1.4 - 1.35 - 0.3, 0]) for b, x in zip((6, 9, 12), xs)]
        note = T_("real SAM ViT-B · 16 × 16 tokens · shared PCA colours", font_size=22, color=GREY_B).move_to([2.0, -0.95, 0])
        arr = Arrow([-3.3, 1.4, 0], [-2.7, 1.4, 0], buff=0, color=CY, stroke_width=4)
        self.beat("J2", FadeOut(cap), lesion.animate.set(height=3.0).move_to([-5.0, 1.4, 0]), rt=1.0, wait=False)
        self.wait_word("six")
        self.play(FadeIn(maps[0]), FadeIn(labs[0]), GrowArrow(arr), run_time=0.7)
        self.wait_word("nine")
        self.play(FadeIn(maps[1]), FadeIn(labs[1]), run_time=0.7)
        self.wait_word("twelve")
        self.play(FadeIn(maps[2]), FadeIn(labs[2]), FadeIn(note), run_time=0.7)
        self.hold()
        # J3: ring on block 6 then block 12
        r6 = SurroundingRectangle(maps[0], color=CY, buff=0.08, stroke_width=6)
        r12 = SurroundingRectangle(maps[2], color=CY, buff=0.08, stroke_width=6)
        self.beat("J3", wait=False)
        self.wait_word("six")
        self.play(Create(r6), run_time=0.6)
        self.wait_word("twelve")
        self.play(ReplacementTransform(r6, r12), run_time=0.8)
        self.hold()
        # J4: Mamba shapes (random weights)
        group = Group(lesion, arr, *maps, *labs, note, r12)
        sizes = [2.2, 1.1, 0.55]; xs4 = [-4.6, -0.6, 3.2]
        boxes = VGroup(*[Square(s, stroke_color=GR, stroke_width=4, fill_color=GR, fill_opacity=0.18).move_to([x, 1.4, 0]) for s, x in zip(sizes, xs4)])
        tl = [T_(t, font_size=24, color=WHITE).move_to([x, 0.05, 0]) for t, s, x in zip(("M1  96 × 64 × 64", "M2  192 × 32 × 32", "M3  384 × 16 × 16"), sizes, xs4)]
        rnd = T_("random weights (stand-in): shapes only", font_size=24, color=AM).move_to([0, -0.8, 0])
        self.beat("J4", FadeOut(group), rt=0.7, wait=False)
        self.wait_word("sixty-four")
        self.play(FadeIn(boxes[0]), FadeIn(tl[0]), run_time=0.6)
        self.wait_word("thirty-two")
        self.play(FadeIn(boxes[1]), FadeIn(tl[1]), run_time=0.6)
        self.wait_word("sixteen")
        self.play(FadeIn(boxes[2]), FadeIn(tl[2]), run_time=0.6)
        self.wait_word("random")
        self.play(FadeIn(rnd), run_time=0.6)
        self.hold()
        # J5: input | paper mask | our output
        xs5 = [-4.8, 0, 4.8]
        a = img("lesion_256.png", 3.0, [xs5[0], 1.3, 0]); b = img("paper_mask_256.png", 3.0, [xs5[1], 1.3, 0]); c = img("pred_mask.png", 3.0, [xs5[2], 1.3, 0])
        t5 = [T_("input", font_size=24, color=WHITE), T_("mask printed in the paper\n(ground truth or\nprediction: unknown)", font_size=20, color=WHITE), T_("our model after 30 steps\non this one image", font_size=20, color=WHITE)]
        for t, x in zip(t5, xs5):
            t.move_to([x, 1.3 - 1.5 - 0.45, 0])
        warn = T_("Dice 0.993 vs that mask, on the training image: memorising, not a result", font_size=24, color=AM).move_to([0, -1.65, 0])
        self.beat("J5", FadeOut(VGroup(boxes, *tl, rnd)), rt=0.7, wait=False)
        self.wait_word("our")
        self.play(FadeIn(a), FadeIn(b), FadeIn(c), *[FadeIn(t) for t in t5], run_time=1.0)
        self.wait_word("memorising,")
        self.play(FadeIn(warn), run_time=0.6)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
