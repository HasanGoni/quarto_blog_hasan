"""Shared base for narrated Manim CHAPTERS (one chapter = one scene file = one small video the user can review on its own).

NarratedScene gives a scene:  seconds(text) / beat(key, *anims, rt, wait) / hold() / save_timeline(slug)
 - the sentence audio is played at the start of the beat (local Kokoro voice, files in assets/um/narration),
 - every beat is recorded in assets/um/timeline_<slug>.json (text, audio start, duration, word reveal times) for subtitle_overlay.py,
 - nothing about subtitles is drawn in the scene itself (they are composited afterwards, see make_chapter.sh).
A chapter file defines  LINES = {key: "sentence with **marked** words"}  (a plain dict literal: make_chapter.sh reads it to synthesise the voice)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from video_speech import NARRATION_DIRNAME, spoken, strip_markup, tts_key, word_times  # noqa: E402

HERE = Path(__file__).resolve().parent
A = HERE / "assets" / "um"
NAR = A / NARRATION_DIRNAME                         # "narration" (Kokoro) or "narration_clone" (the user's cloned voice), chosen by VIDEO_VOICE_ENGINE
MAN = json.load(open(NAR / "manifest.json")) if (NAR / "manifest.json").exists() else {}


class NarratedScene:
    LINES = {}
    SLUG = "chapter"

    def _init_narration(self):
        if not hasattr(self, "_timeline"):
            self._timeline, self._chapters, self._missing, self._beat_end, self._t0, self._times = [], [], [], 0.0, 0.0, []

    def seconds(self, text):
        self._init_narration()
        k = tts_key(spoken(text))
        self._t0 = self.time
        if k in MAN:
            self.add_sound(str(NAR / MAN[k]["file"]))
            e = MAN[k]
            self._times = word_times(text, spoken(text), e.get("ts"), e["seconds"])
            return e["seconds"]
        self._missing.append(spoken(text))
        dur = max(2.8, 0.40 * len(strip_markup(text).split()))
        self._times = word_times(text, spoken(text), None, dur)
        return dur

    def beat(self, key, *anims, rt=1.0, extra_wait=0.0, wait=True):
        """One narration sentence: play its audio, record it for the subtitle overlay, run the visuals, hold until it has been spoken.
        wait=False returns right after the animations so the scene can add more steps; call self.hold() afterwards."""
        self._init_narration()
        text = self.LINES.get(key, key) if key else ""
        self._cur_text = text
        self._chapters.append((key or "title", round(self.time, 2)))
        start = self.time
        dur = self.seconds(text) if text else 0
        if text:
            self._timeline.append(dict(key=key, text=text, t0=round(self._t0, 3), dur=round(dur, 3), times=[round(t, 3) for t in self._times]))
        self._beat_end = start + dur + extra_wait
        if anims:
            self.play(*anims, run_time=rt)
        if wait:
            self.hold()

    def wait_word(self, word, lead=0.0):
        """Wait (inside the current beat) until the spoken word `word` is said (first match, case/punctuation-insensitive), so a visual appears exactly then.
        Call right after beat(..., wait=False) or after earlier plays of the same sentence; does nothing if that moment has already passed."""
        import re
        norm = lambda x: re.sub(r"\W", "", x.lower())
        words = strip_markup(self._cur_text).split()
        for i, x in enumerate(words):
            if norm(x) == norm(word):
                target = self._t0 + self._times[i] - lead
                if self.time < target:
                    self.wait(target - self.time)
                return
        raise ValueError(f"word {word!r} not in sentence: {self._cur_text}")

    def hold(self):
        left = self._beat_end - self.time
        if left > 0:
            self.wait(left)

    def save_timeline(self):
        self._init_narration()
        tag = self.SLUG + os.environ.get("CHAPTER_SUFFIX", "")                  # "" (calm) or ".warm"
        json.dump(dict(chapters=self._chapters, end=round(self.time, 2)), open(A / f"chapters_{tag}.json", "w"))
        json.dump(dict(beats=self._timeline, end=round(self.time, 2)), open(A / f"timeline_{tag}.json", "w"))
        if self._missing:
            NAR.mkdir(parents=True, exist_ok=True)
            mp = NAR / "missing.json"
            prev = json.load(open(mp)) if mp.exists() else []
            json.dump(list(dict.fromkeys(prev + self._missing)), open(mp, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# VISIBLE ZOOM: the viewer must always see that we zoomed, where we are and how far in.  Mixin for a MovingCameraScene (put it BEFORE NarratedScene):
#   class MyChapter(ZoomLens, NarratedScene, MovingCameraScene): ...
#   self.setup_lens(world_center, world_width, world_height, tiles=[node_group, ...])   # once, after the whole picture exists
#   self.zoom_to(target_mobject, fill=0.8)  -> yellow ring marks the area, then the camera flies in; minimap + "x2.4" badge follow the zoom
#   self.zoom_out()                         -> back to the whole picture
# The minimap is screen-fixed (same size at every zoom); it is made of updater mobjects, so Manim does not bake it into a static image.
from manim import (BLACK, DOWN, GREY_B, ORIGIN, UP, UR, WHITE, YELLOW_C, Create, FadeOut, Group, Rectangle, Text, VGroup)  # noqa: E402

HOME_W = 14.222


class ZoomLens:
    MINI_W, MARGIN = 2.7, 0.25
    SHOW_MINIMAP = False                                                     # the user does not want the minimap / zoom-level box; the ring + focus dimming show where we zoom

    def setup_lens(self, world_center, world_w, world_h, tiles=(), home_w=HOME_W):
        self._lw, self._lc, self._lh, self._home_w = float(world_w), [float(x) for x in world_center[:2]], float(world_h), home_w
        self._lens_tiles = [(t.get_center()[:2].copy(), t.width, t.height) for t in tiles]
        self._badges, self._lens_hidden = {}, True                          # the minimap is shown only while a zoom is on screen
        self.lens = VGroup()
        self.lens.set_z_index(500)
        self.lens.add_updater(self._lens_update)
        self.add(self.lens)

    def _badge(self, z):
        k = round(z, 1)
        if k not in self._badges:
            self._badges[k] = Text(f"zoom x{k:.1f}", font="DejaVu Sans", font_size=22, color=YELLOW_C)
        return self._badges[k].copy()

    def _lens_update(self, m, dt=0):
        f = self.camera.frame
        if self._lens_hidden or not self.SHOW_MINIMAP:
            m.become(VGroup())
            return
        s = f.width / self._home_w                                           # screen-fixed: scale the whole minimap with the frame
        mw = self.MINI_W
        mh = mw * self._lh / self._lw
        corner = f.get_corner(UR) + np_vec(-self.MARGIN * s, -self.MARGIN * s)
        c = corner + np_vec(-mw * s / 2, -mh * s / 2)
        k = mw / self._lw                                                    # world units -> minimap units (before * s)
        bg = Rectangle(width=mw * s, height=mh * s, stroke_color=GREY_B, stroke_width=2, fill_color=BLACK, fill_opacity=0.85).move_to(c)
        parts = [bg]
        for (tc, tw, th) in self._lens_tiles:
            r = Rectangle(width=max(tw * k * s, 0.02), height=max(th * k * s, 0.02), stroke_width=0, fill_color=GREY_B, fill_opacity=0.55)
            parts.append(r.move_to(c + np_vec((tc[0] - self._lc[0]) * k * s, (tc[1] - self._lc[1]) * k * s)))
        fc = f.get_center()
        vw, vh = min(f.width * k * s, mw * s), min(f.height * k * s, mh * s)
        v = Rectangle(width=vw, height=vh, stroke_color=YELLOW_C, stroke_width=3, fill_opacity=0)
        parts.append(v.move_to(c + np_vec((fc[0] - self._lc[0]) * k * s, (fc[1] - self._lc[1]) * k * s)))
        z = self._home_w / f.width * (self._world_scale())
        b = self._badge(z).scale(s).next_to(bg, DOWN, buff=0.06 * s)
        b.align_to(bg, np_vec(1, 0))
        parts.append(b)
        m.become(VGroup(*parts))

    def _world_scale(self):
        return self._lw / self._home_w                                        # zoom is quoted relative to the whole picture (x1.0 = whole picture)

    def zoom_to(self, target, fill=0.8, rt=1.6, ring_rt=0.5, usable=0.72):
        """Mark `target` with a yellow ring, then fly the camera in. The target is fitted into the TOP `usable` share of the frame (the subtitle panel covers the bottom)."""
        f = self.camera.frame
        w = max(target.width / fill, target.height / usable * 16 / 9, 1.0)
        H = w * 9 / 16
        c = target.get_center() - np_vec(0, (0.5 - usable / 2) * H)
        ring = Rectangle(width=target.width + 0.3, height=target.height + 0.3, stroke_color=YELLOW_C, stroke_width=8).move_to(target.get_center())
        ring.set_z_index(400)
        self.play(Create(ring), run_time=ring_rt)
        self._lens_hidden = False                                            # the minimap appears only when the camera really starts to move
        self.play(f.animate.set(width=w).move_to(c), FadeOut(ring), run_time=rt)

    def zoom_out(self, rt=1.8):
        self.play(self.camera.frame.animate.set(width=self._lw).move_to([self._lc[0], self._lc[1], 0]), run_time=rt)
        self._lens_hidden = True


def np_vec(x, y, z=0.0):
    import numpy as _np
    return _np.array([x, y, z], dtype=float)
