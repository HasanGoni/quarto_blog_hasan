"""Word-by-word subtitle overlay for a narrated Manim video, rendered as its own TRANSPARENT video and composited on top with ffmpeg.

Why separate: the base video then keeps cheap static waits (a subtitle that changes every word inside the base scene makes Manim render every frame).
Input (env SUBTITLE_TIMELINE): JSON {"beats": [{"text": "... **marked** words ...", "t0": audio start (s), "dur": seconds, "times": [reveal time of every word]}], "end": total seconds}
Each sentence appears on a dark panel at the bottom, building up word by word at the spoken times; **marked** words get a yellow highlighter background.
Render:  manim render --disable_caching -t --format=webm -r 1280,720 --fps 30 subtitle_overlay.py SubtitleOverlay   then composite (see composite_subtitles.sh)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
import numpy as np
from manim import *  # noqa: F403
from video_speech import strip_markup, to_markup

config.background_color = "#000000"
config.transparent = True
TL = json.load(open(os.environ["SUBTITLE_TIMELINE"]))
FONT, SIZE, WIDTH = "DejaVu Sans", 26, 66
TAIL = 0.15                     # subtitle stays this long after the sentence has been spoken


class SubtitleOverlay(Scene):
    def construct(self):
        empty = np.zeros((0, 3))
        beats = []
        holder = VGroup()
        holder.set_z_index(100)
        for b in TL["beats"]:
            n = len(strip_markup(b["text"]).split())
            variants = [MarkupText(to_markup(b["text"], width=WIDTH, n_visible=k), font=FONT, font_size=SIZE, color=WHITE, line_spacing=1.1) for k in range(1, n + 1)]
            full = variants[-1].move_to(ORIGIN)
            for v in variants[:-1]:
                v.align_to(full, UP).align_to(full, LEFT)
            panel = BackgroundRectangle(full, color="#101010", fill_opacity=0.82, buff=0.18)
            g = VGroup(panel, *variants).shift(DOWN * (config.frame_height / 2 - 0.45 - panel.height / 2))
            g.set_z_index(100)
            parts = [[(sm, sm.points.copy()) for sm in layer.get_family() if sm.has_points()] for layer in g.submobjects]   # [panel, variant1, variant2, ...]
            for layer_parts in parts:
                for sm, _ in layer_parts:
                    sm.points = empty                                  # hidden until its moment (shapes without points are skipped by the renderer)
            beats.append(dict(b=b, g=g, parts=parts, n=n))
        self.add(holder)                                               # only the ACTIVE sentence is ever inside `holder`, so every frame stays cheap
        state = {"beat": None, "k": None}

        def show(bi, k):
            """bi None: nothing; else the panel plus the layer with the first k words visible (k = 0: panel only)."""
            for old in list(holder.submobjects):
                holder.remove(old)
            if bi is None:
                return
            bt = beats[bi]
            holder.add(bt["g"])
            for li, layer_parts in enumerate(bt["parts"]):
                vis = li == 0 or li == k
                for sm, pts in layer_parts:
                    sm.points = pts if vis else empty

        def upd(m, dt=0):                                              # time-based updater (dt argument): runs for every frame of the wait below
            t = self.time
            bi = k = None
            if t < 0.04:                       # the holder must be EMPTY when Manim starts the wait, else its children are baked into the static background for the whole video
                return
            for i, bt in enumerate(beats):
                b = bt["b"]
                if b["t0"] <= t < b["t0"] + b["dur"] + TAIL:
                    bi = i
                    k = sum(1 for x in b["times"] if b["t0"] + x <= t)
                    break
            if (bi, k) != (state["beat"], state["k"]):
                state["beat"], state["k"] = bi, k
                show(bi, k)
        holder.add_updater(upd)
        self.wait(TL["end"] + 0.1)
