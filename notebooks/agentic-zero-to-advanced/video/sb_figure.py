"""Chapter 2 (Manim): zoom tour of the paper's REAL Figure 1(a), arXiv 2610.40029. Image = the paper's own figure, rendered from the PDF at 900 dpi.
Build:  ./make_chapter.sh sb_figure.py FigureChapter sb_figure clone warm"""
from pathlib import Path
from manim import *  # noqa: F403
from vidbase import NarratedScene, ZoomLens
from manim import YELLOW_C
from sb_voice import FIGURE

config.background_color = "#181818"
IMG = Path(__file__).resolve().parents[2] / "paper-sambaunet" / "assets" / "fig1a_hi-07.png"
PW, PH = 1060, 1074                                   # reference pixel grid used to write the boxes below (taken from the 1060x1074 crop)
BOXES = {                                            # x0, y0, x1, y1 on that grid
    "top": (185, 0, 995, 215), "left": (8, 272, 306, 1052), "sam": (318, 272, 542, 832),
    "taps": (518, 270, 600, 830), "cpem": (578, 285, 745, 830), "right": (748, 272, 1048, 1052), "left_top": (8, 272, 306, 700), "left_bot": (8, 640, 306, 1052), "right_top": (748, 272, 1048, 700), "right_bot": (748, 640, 1048, 1052), "maskdec": (318, 878, 618, 975),
}


class FigureChapter(ZoomLens, NarratedScene, MovingCameraScene):
    SLUG, LINES = "sb_figure", FIGURE

    def box(self, name):
        x0, y0, x1, y1 = BOXES[name]
        sc = self.img.height / PH
        cx = self.img.get_center()[0] + ((x0 + x1) / 2 - PW / 2) * sc
        cy = self.img.get_center()[1] - ((y0 + y1) / 2 - PH / 2) * sc
        return Rectangle(width=(x1 - x0) * sc, height=(y1 - y0) * sc, stroke_width=0, fill_opacity=0).move_to([cx, cy, 0])

    def _view(self, name, fill, usable):
        tgt = self.box(name)
        w = max(tgt.width / fill, tgt.height / usable * 16 / 9, 1.0)
        H = w * 9 / 16
        cx, cy = tgt.get_center()[:2]
        if w >= self.img.width * 1.05:                         # target narrower than the frame: keep the picture centred, no empty side
            cx = self.img.get_center()[0]
        return tgt, w, [cx, cy - (0.5 - usable / 2) * H, 0]

    def zoom_box(self, name, fill=0.85, usable=0.6):
        tgt, w, c = self._view(name, fill, usable)
        ring = Rectangle(width=tgt.width + 0.2, height=tgt.height + 0.2, stroke_color=YELLOW_C, stroke_width=8).move_to(tgt.get_center())
        ring.set_z_index(400)
        self.play(Create(ring), run_time=0.5)
        self.play(self.camera.frame.animate.set(width=w).move_to(c), FadeOut(ring), run_time=1.5)
        return w

    def tour(self, key, name, fill=0.85, then=None):
        self.beat(key, wait=False)
        if self.camera.frame.width < 14:
            self.zoom_out(rt=1.1)
        w = self.zoom_box(name, fill)
        if then:                                                 # tall region: look at the top half first, pan to the bottom half mid-sentence
            mid = (self._t0 + self._beat_end) / 2
            if self.time < mid:
                self.wait(mid - self.time)
            _, _, c = self._view(then, fill, 0.6)
            self.play(self.camera.frame.animate.move_to(c), run_time=1.4)
        self.hold()

    def ring_only(self, key, name):
        """Short sentence: keep the camera where it is and just pulse a ring around the part being named."""
        self.beat(key, wait=False)
        tgt = self.box(name)
        ring = Rectangle(width=tgt.width + 0.5, height=tgt.height + 0.2, stroke_color=YELLOW_C, stroke_width=3).move_to(tgt.get_center())
        ring.set_z_index(400)
        self.play(Create(ring), run_time=0.6)
        self.hold()
        self.play(FadeOut(ring), run_time=0.3)

    def construct(self):
        self.img = ImageMobject(str(IMG)).set(height=5.75).move_to([0, 1.0, 0])
        bg = Rectangle(width=self.img.width + 0.1, height=self.img.height + 0.1, stroke_width=0, fill_color=WHITE, fill_opacity=1).move_to(self.img)
        self.add(bg, self.img)
        self.setup_lens([0, 0], 14.222, 8.0, tiles=[bg])
        self.beat("F1", wait=True)
        self.tour("F2", "top", fill=0.8)
        self.tour("F3", "left_top", fill=0.9, then="left_bot")
        self.tour("F4", "sam", fill=0.9)
        self.ring_only("F5", "taps")
        self.tour("F6", "cpem", fill=0.9)
        self.tour("F7", "right_top", fill=0.9, then="right_bot")
        self.tour("F8", "maskdec", fill=0.55)
        self.zoom_out()
        self.wait(0.8)
        self.save_timeline()
