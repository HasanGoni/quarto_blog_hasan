"""SpatialClaw for Inspection, Part 5 video: three narrated chapters (Scene classes P5a, P5b, P5c).

Every number on screen comes from outputs/part5_attacks.txt, part5_replay.txt, test_paths5.txt (real runs).

Build (from notebooks/paper-explainers-2026-10-09/videos, module symlinked there with an ABSOLUTE target):
    VC=<repo>/notebooks/voice-clone ./make_chapter.sh part5_video.py P5a p5a clone warm
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




class P5a(NarratedScene, Scene):
    SLUG = "p5a"
    LINES = {
        "A1": "In parts three and four, the model's code ran as **me**.",
        "A2": "It could **read** my files and **write** anywhere I can.",
        "A3": "And the docker **socket** was writable, which means root on this machine.",
        "A4": "I audited the cells, but that was **evidence**, not a guarantee.",
    }

    def construct(self):
        code_t = tile("model's code", 3.4, 1.2, BLUE_C).move_to([-4.2, 2.4, 0])
        me = tile("runs as: me", 3.0, 1.2, RED_C).move_to([0.6, 2.4, 0])
        a0 = Arrow(code_t.get_right(), me.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        self.beat("A1", FadeIn(code_t), FadeIn(me), GrowArrow(a0), rt=1.8)

        files = tile("my files\nread + write", 3.0, 1.2, RED_C, size=22).move_to([-4.2, 0.4, 0])
        net = tile("my network", 3.0, 1.2, RED_C, size=22).move_to([-0.6, 0.4, 0])
        self.beat("A2", FadeIn(files), FadeIn(net), rt=1.4)

        sock = tile("docker socket\nwritable = root", 3.6, 1.2, RED_C, size=22).move_to([3.4, 0.4, 0])
        self.beat("A3", FadeIn(sock), rt=1.2)

        note = lab("93 cells audited: only numpy and scipy imports", 24, GREY_B).move_to([0, -1.1, 0])
        self.beat("A4", FadeIn(note), rt=1.2)
        self.wait(1.0)
        self.save_timeline()


class P5b(NarratedScene, Scene):
    SLUG = "p5b"
    LINES = {
        "B1": "So the code now runs in a **container**.",
        "B2": "**No network**. A **read-only** filesystem. No special powers.",
        "B3": "One gigabyte of memory, one processor, sixty four processes.",
        "B4": "And the host **never trusts** what comes back.",
    }

    def construct(self):
        box = DashedVMobject(RoundedRectangle(corner_radius=0.2, width=6.0, height=4.4, color=RED_C, stroke_width=4)).move_to([3.4, 0.9, 0])
        title = lab("container", 26, RED_C, weight="BOLD").move_to([3.4, 2.7, 0])
        runner = tile("model's code", 3.4, 0.9, BLUE_C, size=22).move_to([3.4, 1.8, 0])
        hcard = card(5.0, 3.6, BLUE_C, 0.04).move_to([-3.9, 0.9, 0])
        host = VGroup(hcard, lab("host", 26, BLUE_C, weight="BOLD").next_to(hcard.get_top(), DOWN, buff=0.25))
        self.beat("B1", Create(box), FadeIn(title), FadeIn(runner), FadeIn(host), rt=1.6)

        l1 = lab("no network", 22, WHITE).move_to([3.4, 0.9, 0]); l2 = lab("read-only filesystem", 22, WHITE).move_to([3.4, 0.4, 0]); l3 = lab("no capabilities", 22, WHITE).move_to([3.4, -0.1, 0])
        self.beat("B2", LaggedStart(FadeIn(l1), FadeIn(l2), FadeIn(l3), lag_ratio=0.4), rt=2.0)
        l4 = lab("1 GB  ·  1 CPU  ·  64 processes", 22, CYAN).move_to([3.4, -0.7, 0])
        self.beat("B3", FadeIn(l4), rt=1.2)

        chk = tile("checks every answer:\nid, shape, hard timeout", 4.2, 1.1, GREEN_C, size=20).move_to([-3.9, 0.6, 0])
        arr = Arrow([0.3, 0.6, 0], [-1.3, 0.6, 0], buff=0.0, color=GREEN_C, stroke_width=5)
        self.beat("B4", FadeIn(chk), GrowArrow(arr), rt=1.4)
        self.wait(1.0)
        self.save_timeline()


class P5c(NarratedScene, Scene):
    SLUG = "p5c"
    LINES = {
        "C1": "I wrote **fifteen** attacks.",
        "C2": "The old kernel let the **reads, writes and connections** through.",
        "C3": "The container **blocked** them.",
        "C4": "A cell that turns off its own alarm gets the **whole container** killed, and the episode ends as review.",
        "C5": "All six hundred thirty four recorded cells **replay identically**.",
        "C6": "What it cannot stop: a kernel **escape**, or a wrong **number**.",
    }

    def construct(self):
        big = lab("15 attacks", 64, WHITE, weight="BOLD").move_to([0, 2.6, 0])
        self.beat("C1", FadeIn(big), rt=1.0)

        rows = ["read a host file", "write outside scratch", "connect to a host service"]
        left = VGroup(*[tile(r + "\nran", 4.2, 0.9, RED_C, size=19) for r in rows]).arrange(DOWN, buff=0.2).move_to([-3.4, 0.2, 0])
        self.beat("C2", big.animate.scale(0.5).move_to([0, 3.4, 0]), LaggedStart(*[FadeIn(t) for t in left], lag_ratio=0.3), rt=1.6)
        right = VGroup(*[tile(r + "\nblocked", 4.2, 0.9, GREEN_C, size=19) for r in rows]).arrange(DOWN, buff=0.2).move_to([3.4, 0.2, 0])
        self.beat("C3", LaggedStart(*[FadeIn(t) for t in right], lag_ratio=0.3), rt=1.4)

        self.play(FadeOut(VGroup(left, right, big)), run_time=0.6)
        alarm = tile("cell turns off its alarm,\nloops forever", 5.2, 1.2, RED_C, size=22).move_to([-3.6, 1.8, 0])
        kill = tile("host kills the container\n-> REVIEW", 5.0, 1.2, GREEN_C, size=22).move_to([3.4, 1.8, 0])
        ar = Arrow(alarm.get_right(), kill.get_left(), buff=0.1, color=GREY_B, stroke_width=5)
        self.beat("C4", FadeIn(alarm), GrowArrow(ar), FadeIn(kill), rt=1.6)
        rep = lab("634 / 634 recorded cells replay identically", 30, GREEN_C, weight="BOLD").move_to([0, 0.2, 0])
        self.beat("C5", FadeIn(rep), rt=1.0)
        lim = lab("not covered: kernel escape  ·  wrong numbers", 26, GREY_B).move_to([0, -0.9, 0])
        self.beat("C6", FadeIn(lim), rt=1.0)
        self.wait(1.0)
        self.save_timeline()
