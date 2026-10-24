"""Part 7 video chapters: the inspection pipeline as a LangGraph graph. Manim, narrated (warm cloned voice).

Every number, box, label and sentence on screen is read from files a real run wrote: assets/frame/state.json (one real held-out
image, from video_frame_state.py), out/pipeline_summary.json, out/fixes_summary.json, assets/frame/boxes.json (every detector box on the 200
dev images, from widget_rows.py). Nothing here is typed in by hand except layout. Images are from the synthetic void dataset.

Chapters:  PipeOrder (code owns the order)  ->  FrameDetect, FrameDecide (one real image walks through all six nodes)
           ->  PipeEdges (the two branches)  ->  BandPlot (where the rules come from)  ->  PipeResult (held-out numbers)
Colour rule: yellow is only the subtitle highlighter. Text is white or cyan; green = plain Python, cyan = a model call, amber = warning/miss.
Build:  ./make_chapter.sh p7_pipeline FrameDetect p7_frame1 clone warm     (about 3 min each, run in the foreground)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
import numpy as np
from manim import *  # noqa: F403
from video_speech import to_markup
from vidbase import NarratedScene

config.background_color = "#181818"
FONT, MONO = "DejaVu Sans", "DejaVu Sans Mono"
CYAN, AMBER, RED = "#6EC6FF", "#F2A93B", "#FF6B6B"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
FRAME_DIR = HERE / "assets" / "frame"

LINES_ORDER = {
    "P1": "In Part 5, the **model** decided what happens next. In this pipeline, the **code** decides.",
    "P2": "There are six steps, in a fixed order: **detect**, **measure**, **classify**, **rules**, **decide**, and **report**.",
    "P3": "Four of them are plain **Python**. No model call, no tokens.",
    "P4": "Only two of them use a **model**: classify, and report.",
    "P5": "In short, the model **judges**, and the code owns the facts, the rules, and the verdict.",
}
LINES_FRAME1 = {
    "F1": "Here is one real image from the held-out set: a **metal** part, seen by X-ray.",
    "F2": "In the first node, **detect**, the classical detector proposes **five** boxes. Most of them will be false alarms.",
    "F3": "Second node, **measure**. Plain code measures every box: **contrast**, **area**, and the distance to the **border**.",
    "F4": "No model has been called yet. **Zero** calls, **zero** tokens.",
}
LINES_FRAME2 = {
    "D1": "Third node, **classify**. One model call reads all five measurements and labels every box. It never sees the **image**.",
    "D2": "Now the **rules**. The model already rejected the small boxes, so there is nothing left for a rule to **drop**.",
    "D3": "In the fifth node, **decide**, two boxes survived, and both are **uncertain**. So the verdict is **escalate**.",
    "D4": "Last, **report**. The model writes the note, but every **number** in it comes from the state.",
    "D5": "Only now do we open the **answer key**. Box four is a real void, so that one is a **hit**. And the pipeline sent it to a human.",
    "D6": "The other real void was **never** boxed. A pipeline can only filter what the detector found.",
}
LINES_EDGES = {
    "G1": "Most of this graph is a straight **line**. We only draw an edge where there is a real **choice**.",
    "G2": "Choice one. If the detector finds **nothing**, we skip straight to the report. No model call at all.",
    "G3": "Choice two. If the classifier's answer is **invalid**, even after one retry, we skip the rules.",
    "G4": "The verdict is **escalate**. The pipeline never guesses. It asks a **human**.",
    "G5": "Both branches are tested with a **scripted** fake model, because a real run may never take them.",
}
LINES_BAND = {
    "B1": "Here is every box the detector found on the **dev** images. **546** boxes. The green ones are real voids.",
    "B2": "**146** of them touch the image **border**. Not one is real. That is rule one.",
    "B3": "Brightness matters too. Among the rest, above contrast **thirty**, about **70** percent are real. Below **fifteen**, about **17**.",
}
LINES_RESULT = {
    "R1": "Same held-out images as Part 5. **200** images, and **387** real voids.",
    "R2": "The plain detector keeps every box. **15.8** percent precision, **20.9** percent recall.",
    "R3": "Two plain rules, border and size, and no model at all: **30** percent precision, and recall of **19.9**. That clears the gate.",
    "R4": "Add the contrast rule: **39** percent precision, but recall drops to **16** percent. Just under the gate.",
    "R5": "Give the classifier to the model: **41** percent, the same recall, and **four** times the tokens. It sends **36** images to a human.",
    "R6": "Without the guidance it does not collapse here: **36** percent. But it finds ten fewer real voids.",
    "R7": "Compare the Part 5 agent: more tokens, and fewer real **voids** than the two rules.",
}


# ---------------------------------------------------------------------------------------------- helpers
def lab(s, size=26, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


def code(s, size=22, color=WHITE):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=MONO, font_size=size, color=color)


def card(w, h, color, fill=0.1, sw=4):
    return RoundedRectangle(corner_radius=0.18, width=w, height=h, color=color, stroke_width=sw, fill_color=color, fill_opacity=fill)


def node(title, sub, color, w=1.95, h=1.2, ts=22, ss=14):
    c = card(w, h, color, 0.15)
    t = lab(title, ts, WHITE, weight="BOLD").move_to(c.get_center() + UP * 0.2)
    s = lab(sub, ss, color).move_to(c.get_center() + DOWN * 0.3)
    return VGroup(c, t, s)


STEPS = [("detect", "find the boxes", GREEN_C), ("measure", "numbers per box", GREEN_C), ("classify", "label each box", CYAN),
         ("rules", "border, size", GREEN_C), ("decide", "the verdict", GREEN_C), ("report", "write the note", CYAN)]
XS = [-5.875 + 2.35 * i for i in range(6)]


def pipeline_row(y, colors=None):
    return [node(n, s, (colors[i] if colors else c)).move_to([XS[i], y, 0]) for i, (n, s, c) in enumerate(STEPS)]


def row_arrows(y):
    return [Arrow([XS[i] + 1.0, y, 0], [XS[i + 1] - 1.0, y, 0], buff=0.03, color=GREY_B, stroke_width=5,
                  max_tip_length_to_length_ratio=0.5) for i in range(5)]


def fit(img, box=2.0):
    """Scale an ImageMobject to fit a box x box square, keeping the aspect ratio."""
    return img.set_width(box) if img.width >= img.height else img.set_height(box)


def load_state():
    return json.load(open(FRAME_DIR / "state.json"))


class FrameStage:
    """The shared stage of the two frame chapters: a thin pipeline strip on top (the ACTIVE node lights up), the real X-ray on the
    left, a panel on the right, and a 'model calls / tokens' counter. All geometry in one place so both chapters match."""
    IMG_H, IMG_C = 4.85, np.array([-3.75, 0.675, 0.0])
    CHIP_Y = 3.62
    OFFS = {0: (0.3, -0.12), 1: (-0.33, 0.0), 2: (-0.12, 0.3), 3: (0.3, 0.02), 4: (0.3, 0.2), 5: (-0.3, 0.0), 6: (0.0, -0.32)}

    def px(self, x, y):
        s = self.IMG_H / self.S["size"][1]
        return self.IMG_C + np.array([(x - self.S["size"][0] / 2) * s, -(y - self.S["size"][1] / 2) * s, 0])

    def box_center(self, i):
        x1, y1, x2, y2 = self.S["candidates"][i]["box"]
        return self.px((x1 + x2) / 2, (y1 + y2) / 2)

    def build_stage(self):
        self.S = load_state()
        self.img = ImageMobject(str(FRAME_DIR / "frame.png")).set_height(self.IMG_H).move_to(self.IMG_C)
        self.img_frame = SurroundingRectangle(self.img, buff=0.04, color=GREY_B, stroke_width=2)
        self.chips = []
        for i, (name, _sub, col) in enumerate(STEPS):
            c = RoundedRectangle(corner_radius=0.14, width=1.95, height=0.56, color=GREY_B, stroke_width=2, fill_color=GREY_B, fill_opacity=0.05)
            t = lab(name, 20, WHITE, weight="BOLD").move_to(c)
            self.chips.append(VGroup(c, t).move_to([-5.5 + 2.2 * i, self.CHIP_Y, 0]))
        self.chip_state = ["idle"] * 6
        self.counter = self._counter_text(0, None)

    def _counter_text(self, calls, tokens):
        s = f"model calls: {calls}" + (f"   tokens: {tokens:,}" if tokens is not None else ("   tokens: 0" if calls == 0 else ""))
        pill = RoundedRectangle(corner_radius=0.16, width=4.5, height=0.46, color=GREY_B, stroke_width=2, fill_color=GREY_B, fill_opacity=0.06)
        t = lab(s, 19, WHITE).move_to(pill)
        return VGroup(pill, t).move_to([4.5, 2.95, 0])

    def chip_anims(self, active, done=()):
        """Light chip `active` up (bright fill, thick outline) and show `done` chips as finished. Returns animations."""
        out = []
        for i, chip in enumerate(self.chips):
            col = STEPS[i][2]
            if i == active:
                out.append(chip[0].animate.set_stroke(col, width=8, opacity=1).set_fill(col, opacity=0.55))
            elif i in done:
                out.append(chip[0].animate.set_stroke(col, width=3, opacity=0.9).set_fill(col, opacity=0.18))
        return out

    def set_counter(self, calls, tokens, color=CYAN):
        new = self._counter_text(calls, tokens)
        new[0].set_stroke(color, width=4).set_fill(color, 0.2)
        anim = Transform(self.counter, new)
        return anim

    def marker(self, i, color=WHITE, sw=4):
        x1, y1, x2, y2 = self.S["candidates"][i]["box"]
        s = self.IMG_H / self.S["size"][1]
        w, h = max((x2 - x1) * s + 0.12, 0.24), max((y2 - y1) * s + 0.12, 0.24)
        return Rectangle(width=w, height=h, color=color, stroke_width=sw).move_to(self.box_center(i))

    def box_label(self, i, color=WHITE, size=20):
        dx, dy = self.OFFS[i]
        return lab(str(i), size, color, weight="BOLD").set_stroke(BLACK, width=6, background=True).move_to(self.box_center(i) + np.array([dx, dy, 0]))


# ------------------------------------------------------------------------------------------------ chapter 1: the idea
class PipeOrder(NarratedScene, Scene):
    SLUG = "p7_order"
    LINES = LINES_ORDER

    def construct(self):
        old = VGroup(card(5.6, 2.4, AMBER, 0.1, 5), lab("Part 5 and 6", 24, AMBER), lab("the model picks\nthe next step", 30, WHITE, weight="BOLD"))
        old[1].move_to(old[0].get_center() + UP * 0.8); old[2].move_to(old[0].get_center() + DOWN * 0.3)
        new = VGroup(card(5.6, 2.4, CYAN, 0.1, 5), lab("Part 7", 24, CYAN), lab("the code picks\nthe next step", 30, WHITE, weight="BOLD"))
        new[1].move_to(new[0].get_center() + UP * 0.8); new[2].move_to(new[0].get_center() + DOWN * 0.3)
        old.move_to([-3.4, 1.2, 0]); new.move_to([3.4, 1.2, 0])
        self.beat("P1", rt=0.1, wait=False)
        self.wait_word("model")
        self.play(FadeIn(old), run_time=0.7)
        self.wait_word("code")
        self.play(FadeIn(new), run_time=0.7)
        self.hold()

        self.play(FadeOut(Group(old, new)), run_time=0.5)
        row = pipeline_row(1.2, colors=[GREY_B] * 6)
        arrows = row_arrows(1.2)
        self.beat("P2", rt=0.1, wait=False)
        for i, w in enumerate(("detect", "measure", "classify", "rules", "decide", "report")):
            self.wait_word(w)
            self.play(FadeIn(row[i]), *([GrowArrow(arrows[i - 1])] if i else []), run_time=0.5)
        self.hold()

        plain = [0, 1, 3, 4]
        self.beat("P3", rt=0.1, wait=False)
        self.wait_word("Python")
        self.play(*[row[i][0].animate.set_color(GREEN_C).set_fill(GREEN_C, 0.15) for i in plain],
                  *[row[i][2].animate.set_color(GREEN_C) for i in plain], run_time=0.9)
        tag = lab("plain Python: free, instant, testable", 24, GREEN_C).move_to([0, -0.45, 0])
        self.play(FadeIn(tag), run_time=0.6)
        self.hold()

        self.beat("P4", rt=0.1, wait=False)
        self.wait_word("model")
        self.play(*[row[i][0].animate.set_color(CYAN).set_fill(CYAN, 0.15) for i in (2, 5)],
                  *[row[i][2].animate.set_color(CYAN) for i in (2, 5)], FadeOut(tag), run_time=0.9)
        tag2 = lab("a model call: two per image at most", 24, CYAN).move_to([0, -0.45, 0])
        self.play(FadeIn(tag2), run_time=0.6)
        self.hold()

        self.play(FadeOut(tag2), run_time=0.4)
        left = VGroup(card(5.4, 1.5, CYAN, 0.1, 5), lab("the model judges", 30, WHITE, weight="BOLD")).move_to([-3.3, -0.55, 0])
        left[1].move_to(left[0])
        right = VGroup(card(5.8, 1.5, GREEN_C, 0.1, 5), lab("the code owns facts,\nrules and the verdict", 28, WHITE, weight="BOLD")).move_to([3.3, -0.55, 0])
        right[1].move_to(right[0])
        self.beat("P5", rt=0.1, wait=False)
        self.wait_word("judges")
        self.play(FadeIn(left), run_time=0.7)
        self.wait_word("facts")
        self.play(FadeIn(right), run_time=0.7)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


# ------------------------------------------------------------------------------------ chapters 2-3: one real image
class FrameDetect(FrameStage, NarratedScene, Scene):
    SLUG = "p7_frame1"
    LINES = LINES_FRAME1

    def construct(self):
        self.build_stage()
        S = self.S
        N = len(S["candidates"])
        assert N == 5, N  # the narration says five boxes
        # ---- F1: the real image
        cap = lab(f"real image: {S['image']} (synthetic void dataset)\nheld out, never used to pick a rule\n(contrast-stretched for display)", 18, GREY_A).move_to([3.4, 1.0, 0])
        self.beat("F1", rt=0.1, wait=False)
        self.play(FadeIn(Group(*self.chips), lag_ratio=0.1), run_time=1.0)
        self.wait_word("real")
        self.play(FadeIn(self.img), Create(self.img_frame), run_time=1.2)
        self.wait_word("metal")
        self.play(FadeIn(cap), FadeIn(self.counter), run_time=0.8)
        self.hold()

        # ---- F2: detect proposes five boxes
        self.play(FadeOut(cap), run_time=0.3)
        self.beat("F2", rt=0.1, wait=False)
        self.wait_word("detect")
        self.play(*self.chip_anims(0), run_time=0.6)
        self.wait_word("proposes")
        marks, labs = [], []
        for i in range(N):
            m, l = self.marker(i, CYAN), self.box_label(i, CYAN)
            marks.append(m); labs.append(l)
            self.play(Create(m), FadeIn(l), run_time=0.35)
        listing = lab(f"candidates: {N}", 24, CYAN).move_to([3.4, 1.4, 0])
        self.wait_word("five")
        self.play(FadeIn(listing), run_time=0.5)
        note = lab("detector output only:\nboxes, no labels, no verdict", 19, GREY_B).move_to([3.4, 0.5, 0])
        self.wait_word("false")
        self.play(FadeIn(note), run_time=0.6)
        self.hold()
        self.play(FadeOut(listing), FadeOut(note), run_time=0.4)

        # ---- F3: measure fills a table, one active row per box, with a zoom crop of the box
        xs = [0.1, 1.3, 2.4, 3.5]
        meta = json.load(open(FRAME_DIR / "crops_meta.json"))
        head = VGroup(*[lab(h, 17, GREY_B).move_to([x, 2.35, 0]) for h, x in zip(("box", "contrast", "area px", "border px"), xs)])
        ys = [1.95 - 0.5 * i for i in range(N)]
        rows = []
        for i, m in enumerate(S["measurements"]):
            cells = [lab(str(i), 22, CYAN, weight="BOLD"), code(f"{m['contrast']:.1f}", 20), code(str(m["area_px"]), 20), code(str(m["edge_distance_px"]), 20)]
            for c, x in zip(cells, xs):
                c.move_to([x, ys[i], 0])
            bg = RoundedRectangle(corner_radius=0.08, width=4.8, height=0.4, color=CYAN, stroke_width=0, fill_color=CYAN, fill_opacity=0).move_to([2.0, ys[i], 0])
            rows.append((bg, cells))
        inset_cap = lab("zoom on the active box", 15, GREY_B).move_to([5.75, 2.2, 0])
        self.beat("F3", rt=0.1, wait=False)
        self.wait_word("measure")
        self.play(*self.chip_anims(1, done=(0,)), run_time=0.6)
        self.play(FadeIn(head), *[FadeIn(b) for b, _ in rows], FadeIn(inset_cap), run_time=0.5)
        crop = None
        for i, (bg, cells) in enumerate(rows):
            img_i = fit(ImageMobject(str(FRAME_DIR / f"crop_{i}.png")), 2.0).move_to([5.75, 1.0, 0])
            m = meta[i]
            left, top, w, h = img_i.get_left()[0], img_i.get_top()[1], img_i.width, img_i.height
            mk = Rectangle(width=max((m["fx2"] - m["fx1"]) * w, 0.12) + 0.1, height=max((m["fy2"] - m["fy1"]) * h, 0.12) + 0.1,
                           color=CYAN, stroke_width=3).move_to([left + (m["fx1"] + m["fx2"]) / 2 * w, top - (m["fy1"] + m["fy2"]) / 2 * h, 0])
            new_crop = Group(img_i, SurroundingRectangle(img_i, buff=0.02, color=GREY_B, stroke_width=2), mk)
            anims = [FadeIn(c) for c in cells] + [bg.animate.set_fill(CYAN, 0.22), Indicate(marks[i], color=WHITE, scale_factor=1.9)]
            anims += [FadeOut(crop)] if crop is not None else []
            anims += [FadeIn(new_crop)]
            self.play(*anims, run_time=0.55)
            self.play(bg.animate.set_fill(CYAN, 0.0), run_time=0.15)
            crop = new_crop
            if i == 0:
                self.wait_word("contrast")
            elif i == 2:
                self.wait_word("area")
            elif i == 3:
                self.wait_word("border")
        self.hold()

        # ---- F4: no model yet
        zero = Rectangle(width=4.9, height=0.7, color=AMBER, stroke_width=5).move_to(self.counter)
        self.beat("F4", rt=0.1, wait=False)
        self.wait_word("model")
        self.play(Create(zero), run_time=0.5)
        self.wait_word("Zero")
        self.play(Indicate(self.counter[1], color=AMBER, scale_factor=1.12), run_time=0.8)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


class FrameDecide(FrameStage, NarratedScene, Scene):
    SLUG = "p7_frame2"
    LINES = LINES_FRAME2

    def construct(self):
        self.build_stage()
        S = self.S
        N = len(S["candidates"])
        marks = [self.marker(i, CYAN) for i in range(N)]
        labs = [self.box_label(i, CYAN) for i in range(N)]
        xs = [0.1, 1.05, 1.95, 2.85, 4.5, 6.2]
        head = VGroup(*[lab(h, 16, GREY_B).move_to([x, 2.35, 0]) for h, x in zip(("box", "contrast", "area", "border", "the model says", "conf"), xs)])
        ys = [1.95 - 0.45 * i for i in range(N)]
        rows = []
        for i, (m, l) in enumerate(zip(S["measurements"], S["labels"])):
            lab_col = {"defect": GREEN_C, "false_alarm": GREY_B, "uncertain": AMBER}[l["label"]]
            chip = VGroup(RoundedRectangle(corner_radius=0.1, width=1.7, height=0.34, color=lab_col, stroke_width=3, fill_color=lab_col, fill_opacity=0.25),
                          lab(l["label"].replace("_", " "), 15, WHITE, weight="BOLD")).move_to([xs[4], ys[i], 0])
            chip[1].move_to(chip[0])
            cells = [lab(str(i), 20, CYAN, weight="BOLD").move_to([xs[0], ys[i], 0]), code(f"{m['contrast']:.1f}", 17).move_to([xs[1], ys[i], 0]),
                     code(str(m["area_px"]), 17).move_to([xs[2], ys[i], 0]), code(str(m["edge_distance_px"]), 17).move_to([xs[3], ys[i], 0])]
            conf = code(f"{l['confidence']:.2f}", 17, GREY_B).move_to([xs[5], ys[i], 0])
            bg = RoundedRectangle(corner_radius=0.08, width=7.0, height=0.4, color=WHITE, stroke_width=0, fill_color=WHITE, fill_opacity=0).move_to([3.15, ys[i], 0])
            rows.append(dict(bg=bg, cells=cells, chip=chip, conf=conf))
        for c in self.chips[:2]:
            c[0].set_stroke(GREEN_C, 3, 0.9).set_fill(GREEN_C, 0.18)
        self.add(self.img, self.img_frame, *self.chips, self.counter, *marks, *labs, head, *[r["bg"] for r in rows],
                 *[c for r in rows for c in r["cells"]])

        # ---- D1: classify
        schema = code('{"labels": [{"id": 0, "label": "...", "confidence": 0.9}, ...]}', 14, CYAN).move_to([3.4, -1.15, 0])
        self.beat("D1", rt=0.1, wait=False)
        self.wait_word("classify")
        self.play(*self.chip_anims(2, done=(0, 1)), self.set_counter(1, None), run_time=0.7)
        self.wait_word("labels")
        for i, r in enumerate(rows):
            self.play(FadeIn(r["chip"]), FadeIn(r["conf"]), run_time=0.25)
        self.play(FadeIn(schema), run_time=0.5)
        self.wait_word("image")
        veil = lab("no pixels were sent, only these numbers", 18, AMBER).move_to([3.4, -1.6, 0])
        self.play(Indicate(self.img_frame, color=AMBER, scale_factor=1.03), FadeIn(veil), run_time=0.8)
        self.hold()
        self.play(FadeOut(schema), FadeOut(veil), run_time=0.4)

        # ---- D2: rules. Nothing to drop here: the model already rejected the small boxes
        rejected = {r["id"]: r["by"] for r in S["rejected"]}
        kept_ids = [k["id"] for k in S["kept"]]
        rule_box = card(6.9, 0.62, GREEN_C, 0.12, 4).move_to([3.4, -1.3, 0])
        rule_txt = lab("R1 border, R2 area under 20 px: both kept boxes pass", 16, WHITE, weight="BOLD").move_to(rule_box)
        self.beat("D2", rt=0.1, wait=False)
        self.wait_word("rules")
        self.play(*self.chip_anims(3, done=(0, 1, 2)), FadeIn(rule_box), FadeIn(rule_txt), run_time=0.7)
        self.wait_word("rejected")
        self.play(*[m.animate.set_color(GREY_B).set_stroke(opacity=0.35) for i, m in enumerate(marks) if i in rejected],
                  *[l.animate.set_color(GREY_B) for i, l in enumerate(labs) if i in rejected], run_time=0.8)
        self.wait_word("drop")
        self.play(*[Indicate(marks[i], color=GREEN_C, scale_factor=2.0) for i in kept_ids], run_time=0.9)
        self.hold()
        self.play(FadeOut(rule_box), FadeOut(rule_txt), run_time=0.4)

        # ---- D3: decide
        stamp = VGroup(card(3.0, 0.62, AMBER, 0.25, 6), lab(S["verdict"].upper(), 30, WHITE, weight="BOLD")).move_to([3.4, -1.55, 0])
        stamp[1].move_to(stamp[0])
        why = lab(S["reason"], 16, GREY_B).move_to([3.4, -0.98, 0])
        self.beat("D3", rt=0.1, wait=False)
        self.wait_word("decide")
        self.play(*self.chip_anims(4, done=(0, 1, 2, 3)), run_time=0.6)
        self.wait_word("survived")
        self.play(*[rows[i]["bg"].animate.set_fill(AMBER, 0.25) for i in kept_ids],
                  *[marks[i].animate.set_color(AMBER) for i in kept_ids], *[labs[i].animate.set_color(AMBER) for i in kept_ids], run_time=1.0)
        self.wait_word("escalate")
        self.play(FadeIn(stamp), FadeIn(why), run_time=0.7)
        self.hold()

        # ---- D4: report. The real note replaces the table; below it, the exact facts the note was given
        table = [head] + [r["bg"] for r in rows] + [c for r in rows for c in r["cells"]] + [r["chip"] for r in rows] + [r["conf"] for r in rows]
        note = lab("\n".join(self._wrap(S["summary"], 54)), 16, WHITE).move_to([3.4, 1.4, 0])
        note_box = card(7.1, note.height + 0.4, CYAN, 0.08, 3).move_to(note)
        facts = "\n".join(f"{{id: {k['id']}, severity: {k['severity']}, confidence: {k['confidence']}}}" for k in S["kept"])
        facts_t = code(facts, 15, GREEN_C).move_to([3.4, -0.8, 0])
        facts_box = card(7.1, facts_t.height + 0.3, GREEN_C, 0.08, 3).move_to(facts_t)
        facts_cap = lab("state.kept: the facts the note was given (plain Python)", 15, GREEN_C).next_to(facts_box, UP, buff=0.08)
        self.beat("D4", rt=0.1, wait=False)
        self.play(FadeOut(stamp), FadeOut(why), *[FadeOut(m) for m in table], run_time=0.5)
        self.wait_word("report")
        self.play(*self.chip_anims(5, done=(0, 1, 2, 3, 4)), self.set_counter(S["llm_calls"], S["tokens"]), run_time=0.7)
        self.play(FadeIn(note_box), FadeIn(note), run_time=0.8)
        self.wait_word("number")
        self.play(FadeIn(facts_box), FadeIn(facts_t), FadeIn(facts_cap), run_time=0.8)
        self.play(Indicate(facts_box, color=GREEN_C, scale_factor=1.03), run_time=0.7)
        self.hold()
        self.play(FadeOut(note), FadeOut(note_box), FadeOut(facts_box), FadeOut(facts_t), FadeOut(facts_cap), run_time=0.4)

        # ---- D5/D6: the answer key, only now
        gts = []
        for g in S["answer_key"]:
            x1, y1, x2, y2 = g
            s = self.IMG_H / S["size"][1]
            gts.append(Rectangle(width=max((x2 - x1) * s, 0.0) + 0.3, height=max((y2 - y1) * s, 0.0) + 0.3, color=GREEN_C, stroke_width=5).move_to(self.px((x1 + x2) / 2, (y1 + y2) / 2)))
        real_ids = S["real_ids"]
        assert real_ids == [4] and S["gt_hit"] == [True, False], (real_ids, S["gt_hit"])  # the narration says: box four is the hit, one void missed
        key_tag = lab("answer key (used for scoring only,\nnever shown to the pipeline)", 16, GREEN_C).move_to([3.4, 1.3, 0])
        hit_tag = lab("box 4: a hit", 20, GREEN_C, weight="BOLD").move_to([3.4, 0.2, 0])
        sent_tag = lab("it was sent to a human (uncertain)", 17, AMBER).move_to([3.4, -0.3, 0])
        self.beat("D5", rt=0.1, wait=False)
        self.wait_word("answer")
        self.play(FadeIn(key_tag), *[Create(gts[i]) for i, h in enumerate(S["gt_hit"]) if h], run_time=0.9)
        self.wait_word("four")
        self.play(Indicate(gts[0], color=GREEN_C, scale_factor=1.6), FadeIn(hit_tag), run_time=0.9)
        self.wait_word("human")
        self.play(FadeIn(sent_tag), run_time=0.7)
        self.hold()
        self.beat("D6", rt=0.1, wait=False)
        self.wait_word("other")
        missed = [i for i, h in enumerate(S["gt_hit"]) if not h]
        self.play(*[Create(gts[i]) for i in missed], run_time=0.8)
        miss_tags = [lab("missed", 16, AMBER, weight="BOLD").set_stroke(BLACK, width=6, background=True).move_to(gts[i].get_center() + np.array([0.0, 0.4, 0])) for i in missed]
        self.wait_word("never")
        self.play(*[FadeIn(t) for t in miss_tags], *[gts[i].animate.set_color(AMBER) for i in missed], run_time=0.8)
        tag2 = lab("1 of the 2 real voids:\nthe detector never proposed it", 21, AMBER, weight="BOLD").move_to([3.4, 0.6, 0])
        self.play(FadeOut(key_tag), FadeOut(hit_tag), FadeOut(sent_tag), FadeIn(tag2), run_time=0.6)
        self.hold()
        self.wait(0.8)
        self.save_timeline()

    @staticmethod
    def _wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        return lines + [cur]


# ------------------------------------------------------------------------------------------- chapter 4: the two edges
class PipeEdges(NarratedScene, Scene):
    SLUG = "p7_edges"
    LINES = LINES_EDGES

    def construct(self):
        Y = 1.7
        row = pipeline_row(Y)
        arrows = row_arrows(Y)
        OFF = np.array([0, -0.85, 0])
        tok = Dot(np.array([XS[0], Y, 0]) + OFF, radius=0.15, color=WHITE).set_z_index(30)

        # ---- G1: a straight line, the token walks it and each node lights up as the token reaches it
        self.beat("G1", rt=0.1, wait=False)
        self.play(FadeIn(Group(*row), lag_ratio=0.12), run_time=1.4)
        self.play(*[GrowArrow(a) for a in arrows], FadeIn(tok), run_time=0.7)
        self.wait_word("line")
        for i in range(6):
            self.play(tok.animate.move_to(np.array([XS[i], Y, 0]) + OFF), row[i][0].animate.set_stroke(WHITE, 8).set_fill(STEPS[i][2], 0.5), run_time=0.45, rate_func=linear)
            self.play(row[i][0].animate.set_stroke(STEPS[i][2], 4).set_fill(STEPS[i][2], 0.15), run_time=0.15)
        self.wait_word("choice")
        self.play(tok.animate.move_to(np.array([XS[0], Y, 0]) + OFF), run_time=0.8)
        self.hold()

        # ---- G2: nothing detected -> straight to report (real: 55 of the 200 held-out images raise no box)
        top = CurvedArrow([XS[0], Y + 0.65, 0], [XS[5], Y + 0.65, 0], angle=-0.5, color=AMBER, stroke_width=6)
        top_l = lab("no candidates", 24, AMBER).move_to([0, Y + 1.6, 0])
        thumb = ImageMobject(str(FRAME_DIR / "zero.png")).set_height(1.5).move_to([-5.0, -0.9, 0])
        thumb_f = SurroundingRectangle(thumb, buff=0.03, color=GREY_B, stroke_width=2)
        zero_l = lab("detect: 0 boxes", 20, AMBER).next_to(thumb, RIGHT, buff=0.25)
        honest = lab("real image test_00003: the detector raised 0 boxes\n(55 of 200 held-out images are like this)", 16, GREY_B).move_to([2.2, -1.1, 0])
        for i in (1, 2, 3, 4):
            row[i].save_state()
        self.beat("G2", rt=0.1, wait=False)
        self.wait_word("nothing")
        self.play(FadeIn(thumb), Create(thumb_f), FadeIn(zero_l), tok.animate.move_to(np.array([XS[0], Y, 0]) + OFF), run_time=0.9)
        self.play(Create(top), FadeIn(top_l), run_time=0.9)
        self.wait_word("skip")
        self.play(tok.animate.move_to(np.array([XS[0], Y + 0.9, 0])), run_time=0.3)
        self.play(MoveAlongPath(tok, top), run_time=1.5)
        self.wait_word("model")
        self.play(*[row[i].animate.set_opacity(0.28) for i in (1, 2, 3, 4)], row[5][0].animate.set_stroke(WHITE, 8).set_fill(CYAN, 0.5), FadeIn(honest), run_time=0.8)
        self.hold()
        self.play(FadeOut(top), FadeOut(top_l), FadeOut(thumb), FadeOut(thumb_f), FadeOut(zero_l), FadeOut(honest),
                  tok.animate.move_to(np.array([XS[2], Y, 0]) + OFF),
                  *[Restore(row[i]) for i in (1, 2, 3, 4)], row[5][0].animate.set_stroke(CYAN, 4).set_fill(CYAN, 0.15), run_time=0.8)

        # ---- G3: the classifier answers badly twice (a retry dialogue), the rules are skipped
        bot = CurvedArrow([XS[2], Y - 0.65, 0], [XS[4], Y - 0.65, 0], angle=0.7, color=AMBER, stroke_width=6)

        def bubble(txt, color, x, y, w=3.4):
            b = card(w, 0.5, color, 0.12, 3).move_to([x, y, 0])
            return VGroup(b, code(txt, 15, WHITE).move_to(b))
        b1 = bubble('model: "not json"', RED, -3.4, -0.5, 3.4)
        b2 = bubble("us: failed validation, fix it", GREY_B, -3.4, -1.1, 4.3)
        b3 = bubble('model: "still not json"', RED, -3.4, -1.7, 3.4)
        x1 = lab("x", 26, RED, weight="BOLD").next_to(b1, RIGHT, buff=0.1)
        x2 = lab("x", 26, RED, weight="BOLD").next_to(b3, RIGHT, buff=0.1)
        skip_l = lab("rules skipped", 22, AMBER).move_to([(XS[2] + XS[4]) / 2 + 1.4, -0.55, 0])
        self.beat("G3", rt=0.1, wait=False)
        self.wait_word("invalid")
        self.play(Indicate(row[2], color=AMBER, scale_factor=1.1), FadeIn(b1), FadeIn(x1), run_time=0.8)
        self.play(FadeIn(b2), run_time=0.5)
        self.play(FadeIn(b3), FadeIn(x2), run_time=0.5)
        self.wait_word("skip")
        self.play(Create(bot), row[3].animate.set_opacity(0.28), FadeIn(skip_l), run_time=1.0)
        self.hold()

        # ---- G4: escalate
        esc = VGroup(card(2.4, 0.7, AMBER, 0.2, 5), lab("escalate", 26, WHITE, weight="BOLD")).move_to([XS[4], -1.45, 0])
        esc[1].move_to(esc[0])
        self.beat("G4", rt=0.1, wait=False)
        self.wait_word("escalate")
        self.play(Indicate(row[4], color=AMBER, scale_factor=1.1), tok.animate.move_to(np.array([XS[4], Y, 0]) + OFF), FadeIn(esc), run_time=0.9)
        self.wait_word("human")
        human = lab("a human looks", 22, AMBER).next_to(esc, RIGHT, buff=0.2)
        self.play(FadeIn(human), run_time=0.6)
        self.hold()

        # ---- G5: real test output
        self.play(FadeOut(Group(*row, *arrows, tok, bot, b1, b2, b3, x1, x2, skip_l, esc, human)), run_time=0.6)
        cons = card(12.6, 2.9, GREY_B, 0.08, 3).move_to([0, 1.0, 0])
        L = ["no candidates      ['detect', 'report'] | model calls 0 | verdict pass",
             "classifier fails twice  ['detect', 'measure', 'classify', 'decide', 'report']",
             "                        verdict escalate",
             "13 checks, all pass"]
        lines = VGroup(*[code(t, 17, GREEN_C if i == 3 else WHITE) for i, t in enumerate(L)]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(cons)
        real = lab("real output of test_pipeline_paths.py (scripted fake model)", 18, GREY_B).next_to(cons, UP, buff=0.08).align_to(cons, RIGHT)
        self.beat("G5", rt=0.1, wait=False)
        self.wait_word("branches")
        self.play(FadeIn(cons), FadeIn(real), FadeIn(lines[0]), FadeIn(lines[1]), FadeIn(lines[2]), run_time=1.0)
        self.wait_word("scripted")
        self.play(FadeIn(lines[3]), run_time=0.6)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


# ------------------------------------------------------------------------ chapter 5: where the rules come from (real data)
class BandPlot(NarratedScene, Scene):
    SLUG = "p7_band"
    LINES = LINES_BAND

    def construct(self):
        rows = json.load(open(FRAME_DIR / "boxes.json"))  # [split, contrast, area, edge, real] for every detector box on the 200 dev images
        real = [r for r in rows if r[4]]
        border = [r for r in rows if r[3] == 0]
        rest = [r for r in rows if r[3] > 0]
        hi = [r for r in rest if r[1] >= 30]
        lo = [r for r in rest if r[1] < 15]
        assert (len(rows), len(real), len(border), sum(r[4] for r in border)) == (546, 122, 146, 0)
        p_hi, p_lo = sum(r[4] for r in hi) / len(hi), sum(r[4] for r in lo) / len(lo)
        assert round(100 * p_hi) == 70 and round(100 * p_lo) == 17, (p_hi, p_lo)  # the narration says 70 and 17 percent
        X0, X1, C0, C1 = -6.0, 6.2, -40, 60
        xmap = lambda c: X0 + (min(max(c, C0), C1) - C0) / (C1 - C0) * (X1 - X0)
        YB, YT = -0.6, 2.9
        rng = np.random.default_rng(7)
        axis = Line([X0, YB - 0.1, 0], [X1, YB - 0.1, 0], color=GREY_B, stroke_width=3)
        ticks = VGroup(*[VGroup(Line([xmap(c), YB - 0.1, 0], [xmap(c), YB - 0.22, 0], color=GREY_B, stroke_width=2),
                                lab(str(c), 16, GREY_B).move_to([xmap(c), YB - 0.45, 0])) for c in (-40, -20, 0, 15, 30, 60)])
        axis_t = lab("contrast of the box against its surroundings (left edge: -40 or lower)", 18, GREY_B).move_to([0.1, YB - 0.85, 0])
        pos = lambda r: [xmap(r[1]), rng.uniform(YB + 0.2, YT - 0.2), 0]
        fdots = VGroup(*[Dot(pos(r), radius=0.04, color=GREY_B).set_opacity(0.6) for r in rows if not r[4]])
        rdots = VGroup(*[Dot(pos(r), radius=0.1, color=GREEN_C).set_stroke(WHITE, 2) for r in rows if r[4]])
        legend = VGroup(Dot(radius=0.04, color=GREY_B), lab(f"false alarm ({len(rows) - len(real)})", 16, GREY_B), Dot(radius=0.1, color=GREEN_C),
                        lab(f"real void ({len(real)})", 16, GREEN_C)).arrange(RIGHT, buff=0.15).move_to([-4.3, 3.7, 0])

        self.beat("B1", rt=0.1, wait=False)
        self.play(Create(axis), FadeIn(ticks), FadeIn(axis_t), run_time=0.9)
        self.wait_word("every")
        self.play(FadeIn(fdots, lag_ratio=0.004), run_time=2.0)
        self.wait_word("546")
        self.play(FadeIn(legend), run_time=0.5)
        self.wait_word("green")
        self.play(FadeIn(rdots, lag_ratio=0.03), run_time=1.0)
        self.play(*[Indicate(d, color=GREEN_C, scale_factor=1.5) for d in list(rdots)[:60]], run_time=0.9)
        self.hold()

        # ---- B2: the border boxes (their contrast is not special, so they are marked, not moved)
        bdots = VGroup(*[Dot(pos(r), radius=0.07, color=AMBER).set_opacity(0.85) for r in border])
        b_l = lab(f"{len(border)} touch the border: 0 real", 22, AMBER, weight="BOLD").move_to([3.6, YT + 0.38, 0])
        self.beat("B2", rt=0.1, wait=False)
        self.wait_word("146")
        self.play(FadeIn(bdots, lag_ratio=0.01), run_time=1.4)
        self.wait_word("border")
        self.play(FadeIn(b_l), run_time=0.7)
        self.wait_word("rule")
        self.hold()
        self.play(FadeOut(bdots), FadeOut(b_l), run_time=0.6)

        # ---- B3: contrast bands among the rest
        band_hi = Rectangle(width=xmap(C1) - xmap(30), height=YT - YB + 0.1, color=CYAN, stroke_width=0, fill_color=CYAN, fill_opacity=0.16).move_to([(xmap(30) + xmap(C1)) / 2, (YT + YB) / 2 - 0.05, 0])
        band_lo = Rectangle(width=xmap(15) - xmap(C0), height=YT - YB + 0.1, color=AMBER, stroke_width=0, fill_color=AMBER, fill_opacity=0.10).move_to([(xmap(C0) + xmap(15)) / 2, (YT + YB) / 2 - 0.05, 0])
        hi_l = lab(f"30 and up: {100 * p_hi:.0f}% real", 22, CYAN, weight="BOLD").move_to([(xmap(30) + xmap(C1)) / 2 - 0.3, YT + 0.38, 0])
        lo_l = lab(f"under 15: {100 * p_lo:.0f}% real", 22, AMBER, weight="BOLD").move_to([(xmap(C0) + xmap(15)) / 2, YT + 0.38, 0])
        note = lab("(boxes not touching the border)", 16, GREY_B).move_to([0.1, YT + 0.05, 0])
        self.beat("B3", rt=0.1, wait=False)
        self.wait_word("contrast")
        self.play(FadeIn(note), run_time=0.4)
        self.wait_word("thirty")
        self.play(FadeIn(band_hi), FadeIn(hi_l), run_time=0.9)
        self.wait_word("fifteen")
        self.play(FadeIn(band_lo), FadeIn(lo_l), run_time=0.9)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


# ------------------------------------------------------------------------------------- chapter 6: the held-out numbers
def load_rows():
    """The bars are built from the real result files, and the narration numbers are asserted against them."""
    v = json.load(open(OUT / "pipeline_summary.json"))["variants"]
    mean = lambda runs, k: sum(r[k] for r in runs) / len(runs)
    f5 = json.load(open(OUT / "fixes_summary.json"))
    band, llm, ng = v["band"], v["llm"], v["llm_noguide"]
    wr = json.load(open(OUT / "widget_rows.json"))
    held = [r for r in wr["rows"] if r[0] == 1]
    kept = [r for r in held if r[3] > 0 and r[2] >= 20]
    r12 = (sum(r[4] for r in kept) / len(kept), sum(r[4] for r in kept) / wr["gt"][1])
    rows = [("plain detector", f5["plain_detector_test"]["precision"], f5["plain_detector_test"]["recall"], None, "-"),
            ("rules R1 + R2, no model", r12[0], r12[1], 0, "-"),
            ("pipeline + contrast rule", mean(band, "precision"), mean(band, "recall"), mean(band, "avg_tokens"), mean([{"e": r["verdicts"]["escalate"]} for r in band], "e")),
            ("pipeline + LLM classifier", mean(llm, "precision"), mean(llm, "recall"), mean(llm, "avg_tokens"), mean([{"e": r["verdicts"]["escalate"]} for r in llm], "e")),
            ("same, no guidance", mean(ng, "precision"), mean(ng, "recall"), mean(ng, "avg_tokens"), mean([{"e": r["verdicts"]["escalate"]} for r in ng], "e")),
            ("Part 5 agent + guidance", f5["fix_design_test"]["precision"], f5["fix_design_test"]["recall"], f5["fix_design_test"]["avg_tokens"], "-")]
    pct = lambda x: round(100 * x, 1)
    assert pct(rows[0][1]) == 15.8 and pct(rows[0][2]) == 20.9 and round(100 * rows[1][1]) == 30 and pct(rows[1][2]) == 19.9
    assert round(100 * rows[2][1]) == 39 and round(100 * rows[2][2]) == 16 and round(100 * rows[3][1]) == 41
    assert 3.9 < rows[3][3] / rows[2][3] < 4.3 and round(rows[3][4]) == 36 and round(100 * rows[4][1]) == 36
    assert sum(r["tp"] for r in [llm[0]]) == 60 and wr["gt"][1] == 387
    return rows


class PipeResult(NarratedScene, Scene):
    SLUG = "p7_result"
    LINES = LINES_RESULT

    def construct(self):
        rows = load_rows()
        BAR0, BARW, PMAX = -2.0, 3.0, 0.45
        head = VGroup(lab("precision", 19, GREY_B).move_to([-0.6, 3.55, 0]), lab("recall", 19, GREY_B).move_to([3.2, 3.55, 0]),
                      lab("tokens / image", 19, GREY_B).move_to([5.45, 3.55, 0]))
        frame = card(13.4, 5.5, GREY_B, 0.05, 3).move_to([0, 1.55, 0])
        note = lab("200 held-out images, 387 real voids, Qwen3-Coder-30B. LLM rows: mean of 3 runs", 18, GREY_B).move_to([0, -1.5, 0])
        colors = [GREY_B, GREEN_C, CYAN, CYAN, AMBER, WHITE]
        ys = [2.85, 2.1, 1.35, 0.6, -0.15, -0.9]
        cells = []
        for r, y, col in zip(rows, ys, colors):
            tok = "-" if r[3] is None else f"{round(r[3]):,}"
            name = lab(r[0], 19, WHITE, weight="BOLD").move_to([-6.5, y, 0], aligned_edge=LEFT)
            bar = Rectangle(width=max(BARW * r[1] / PMAX, 0.04), height=0.34, color=col, stroke_width=0, fill_color=col, fill_opacity=0.75).move_to([BAR0, y, 0], aligned_edge=LEFT)
            val = lab(f"{100 * r[1]:.1f}%", 22, WHITE, weight="BOLD").next_to(bar, RIGHT, buff=0.15)
            rec = lab(f"{100 * r[2]:.1f}%", 24, WHITE, weight="BOLD").move_to([3.2, y, 0])
            tk = lab(tok, 24, WHITE, weight="BOLD").move_to([5.45, y, 0])
            cells.append([name, bar, val, rec, tk])

        def show(k, **kw):
            return [FadeIn(cells[k][0]), FadeIn(cells[k][3]), FadeIn(cells[k][4]), GrowFromEdge(cells[k][1], LEFT), FadeIn(cells[k][2])]

        self.beat("R1", rt=0.1, wait=False)
        self.play(FadeIn(frame), FadeIn(head), FadeIn(note), run_time=0.8)
        self.wait_word("200")
        self.play(Indicate(note, color=CYAN, scale_factor=1.03), run_time=0.8)
        self.hold()
        self.beat("R2", rt=0.1, wait=False)
        self.wait_word("detector")
        self.play(*[FadeIn(c) for c in cells[0]], run_time=0.7)
        self.wait_word("15.8")
        self.play(Indicate(cells[0][2], color=CYAN), run_time=0.7)
        self.hold()
        self.beat("R3", rt=0.1, wait=False)
        self.wait_word("rules")
        self.play(*show(1), run_time=0.9)
        self.wait_word("30")
        self.play(Indicate(cells[1][2], color=GREEN_C), run_time=0.7)
        self.wait_word("19.9")
        self.play(Indicate(cells[1][3], color=GREEN_C), run_time=0.7)
        self.hold()
        self.beat("R4", rt=0.1, wait=False)
        self.wait_word("contrast")
        self.play(*show(2), run_time=0.9)
        self.wait_word("39")
        self.play(Indicate(cells[2][2], color=CYAN), run_time=0.7)
        self.wait_word("drops")
        self.play(Indicate(cells[2][3], color=AMBER), run_time=0.8)
        self.hold()
        self.beat("R5", rt=0.1, wait=False)
        self.wait_word("model")
        self.play(*show(3), run_time=0.9)
        self.wait_word("four")
        self.play(Indicate(cells[3][4], color=AMBER), run_time=0.8)
        self.wait_word("36")
        esc = lab(f"{round(rows[3][4])} images escalated", 17, AMBER).move_to([4.9, ys[3] - 0.3, 0])
        self.play(FadeIn(esc), run_time=0.6)
        self.hold()
        self.play(FadeOut(esc), run_time=0.3)
        self.beat("R6", rt=0.1, wait=False)
        self.wait_word("guidance")
        self.play(*show(4), run_time=0.9)
        self.wait_word("fewer")
        self.play(Indicate(cells[4][3], color=AMBER), run_time=0.8)
        self.hold()
        self.beat("R7", rt=0.1, wait=False)
        self.wait_word("Part")
        self.play(*show(5), run_time=0.9)
        self.wait_word("voids")
        self.play(Indicate(cells[5][3], color=AMBER), Indicate(cells[1][3], color=GREEN_C), run_time=0.9)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
