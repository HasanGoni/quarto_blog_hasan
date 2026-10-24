"""Part 8 video, chapter "the box call" (Manim: it plots real per-box probabilities for 5 judges, which is what Manim is good at).

Every dot is one of the 513 held-out detector boxes (81 real); its height is the probability a judge gave. Read from out/typed/results_*.json,
the same files the post's table is built from. Narration numbers are asserted against them when the scene is built.
Colour rule: yellow is only the subtitle highlighter. Green = real defect, grey = false alarm.
Build:  ./make_chapter.sh p8_box BoxCall p8_box clone warm"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
import numpy as np
from manim import *  # noqa: F403
from video_speech import to_markup
from vidbase import NarratedScene

config.background_color = "#181818"
FONT = "DejaVu Sans"
CYAN, AMBER = "#6EC6FF", "#F2A93B"
OUT = Path(__file__).resolve().parent.parent / "out" / "typed"

LINES = {
    "K1": "In the pipeline, one call is a pure judgment: is this box a **real** defect? We know the true answer for all **513** held-out boxes.",
    "K2": "Now the plain **rule**: contrast of at least eight. It ranks the boxes with a score of **0.64**.",
    "K3": "Next, the small typed model, with no guidance. Its score is **0.43**. That is worse than a coin flip.",
    "K4": "Now write the guidance into the question, and the score rises to **0.60**.",
    "K5": "But look at the spread. Every probability sits between **five** and **fifty** percent. It is honest about being unsure, and **not** very useful.",
    "K6": "Next, the language model with no guidance, when we ask it to write a probability. Its score is **0.36**. That is worse than a coin flip too.",
    "K7": "With guidance it reaches **0.70**, the best score here. But it needs over half a **second** and over a hundred tokens.",
    "K7B": "Read its token probabilities instead, and the score is **0.64**, the same as the rule, in about **two hundred** milliseconds.",
    "K8": "With only eighty-one real defects, these gaps are within **noise**. So for numbers, a simple **rule** is the sensible choice. It is exact, instant and free.",
}


def lab(s, size=24, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


def auc(p, y):
    pos, neg = [a for a, b in zip(p, y) if b], [a for a, b in zip(p, y) if not b]
    return sum((x > z) + 0.5 * (x == z) for x in pos for z in neg) / (len(pos) * len(neg))


def load():
    rows = [json.loads(l) for l in (OUT / "candidates.jsonl").read_text().splitlines()]
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    y = [rows[i]["y"] for i in test]
    cols = []
    for key, judge, tag in (("rule", "band", "guided"), ("laya0", "laya-english", "plain"), ("laya1", "laya-english", "guided"),
                            ("llm0", "llm-verbalized", "plain"), ("llm1", "llm-verbalized", "guided"), ("llm2", "llm-logprob", "guided")):
        res = json.load(open(OUT / f"results_{judge}.json"))["A"]
        p = [res["p"][tag][i] for i in test]
        lat = res["scores"][tag]["latency"]["p50_ms"]
        tok = res["scores"][tag]["tokens"]
        cols.append(dict(key=key, p=p, auc=auc(p, y), ms=lat, tok=tok))
    return y, cols


class BoxCall(NarratedScene, Scene):
    SLUG = "p8_box"
    LINES = LINES

    def construct(self):
        y, cols = load()
        A = {c["key"]: c for c in cols}
        # the narration says these numbers; refuse to build if the data disagrees
        want = {"rule": "0.64", "laya0": "0.43", "laya1": "0.60", "llm0": "0.36", "llm1": "0.70", "llm2": "0.64"}
        for k, v in want.items():
            assert f"{A[k]['auc']:.2f}" == v, (k, A[k]["auc"], v)
        lo, hi = min(A["laya1"]["p"]), max(A["laya1"]["p"])
        assert (round(100 * lo), round(100 * hi)) == (5, 50), (lo, hi)
        assert (len(y), sum(y)) == (513, 81) and 0.5 < A["llm1"]["ms"] / 1000 < 0.65 and A["llm1"]["tok"] > 100 and A["rule"]["tok"] == 0 and 180 < A["llm2"]["ms"] < 250

        X = {"rule": -5.3, "laya0": -3.2, "laya1": -1.05, "llm0": 1.1, "llm1": 3.25, "llm2": 5.4}
        YB, YT, W = -1.0, 2.35, 0.8
        yy = lambda p: YB + p * (YT - YB)
        rng = np.random.default_rng(3)
        names = {"rule": "rule", "laya0": "Laya\nno guidance", "laya1": "Laya\n+ guidance", "llm0": "LLM (number)\nno guidance", "llm1": "LLM (number)\n+ guidance", "llm2": "LLM (token prob.)\n+ guidance"}
        colour = {"rule": GREEN_C, "laya0": CYAN, "laya1": CYAN, "llm0": AMBER, "llm1": AMBER, "llm2": "#FFD27A"}
        axis = VGroup(Line([-6.2, YB, 0], [6.4, YB, 0], color=GREY_B, stroke_width=2), Line([-6.2, YB, 0], [-6.2, YT, 0], color=GREY_B, stroke_width=2))
        ticks = VGroup(*[VGroup(lab(f"{int(p * 100)}%", 16, GREY_B).move_to([-6.65, yy(p), 0]),
                                Line([-6.2, yy(p), 0], [6.4, yy(p), 0], color=GREY_B, stroke_width=1, stroke_opacity=0.25)) for p in (0, 0.5, 1)])
        ytitle = lab("height = the probability a judge gave that the box is real", 16, GREY_B).move_to([-3.0, 3.75, 0])
        legend = VGroup(Dot(radius=0.03, color=GREY_B), lab("false alarm (432)", 15, GREY_B), Dot(radius=0.09, color=GREEN_C), lab("real void (81)", 15, GREEN_C)).arrange(RIGHT, buff=0.12).move_to([4.6, 3.75, 0])

        def column(c):
            xs = X[c["key"]]
            dots, reals = VGroup(), []
            for p, truth in sorted(zip(c["p"], y), key=lambda t: t[1]):   # real dots last, so they sit on top
                pos = [xs + rng.uniform(-W, W), yy(p), 0]
                d = Dot(pos, radius=0.1 if truth else 0.032, color=GREEN_C if truth else GREY_B).set_opacity(1 if truth else 0.55).set_stroke(WHITE if truth else None, 2 if truth else 0)
                dots.add(d)
                if truth:
                    reals.append(d)
            head = lab(names[c["key"]], 13, colour[c["key"]], weight="BOLD").move_to([xs, YB - 0.55, 0])
            ms = "0 ms" if c["ms"] < 0.01 else (f"{c['ms']:.0f} ms")
            info = lab(f"score {c['auc']:.2f}\n{ms}\n{int(round(c['tok']))} tokens", 14, WHITE).move_to([xs, YT + 0.62, 0])
            frame = SurroundingRectangle(Group(dots), buff=0.12, color=colour[c["key"]], stroke_width=2, stroke_opacity=0.5)
            return dots, head, info, frame, reals

        cd = {c["key"]: column(c) for c in cols}

        # ---- K1
        self.beat("K1", rt=0.1, wait=False)
        self.play(Create(axis), FadeIn(ticks), FadeIn(ytitle), run_time=1.0)
        self.wait_word("real")
        self.play(FadeIn(legend), run_time=0.6)
        self.wait_word("513")
        self.play(Indicate(legend, color=WHITE, scale_factor=1.06), run_time=0.8)
        self.hold()

        def show(key, extra=()):
            dots, head, info, frame, _ = cd[key]
            return [FadeIn(dots, lag_ratio=0.004), FadeIn(head), Create(frame), *extra]

        def real_dots(key):
            return cd[key][4]

        # ---- K2 the rule
        self.beat("K2", rt=0.1, wait=False)
        self.wait_word("rule")
        self.play(*show("rule"), run_time=1.5)
        self.wait_word("0.64")
        self.play(FadeIn(cd["rule"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("rule"))[:20]], run_time=1.0)
        self.hold()

        # ---- K3 Laya, no guidance
        self.beat("K3", rt=0.1, wait=False)
        self.wait_word("typed")
        self.play(*show("laya0"), run_time=1.5)
        self.wait_word("0.43")
        self.play(FadeIn(cd["laya0"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("laya0"))[:20]], run_time=1.0)
        self.hold()

        # ---- K4 Laya, guidance
        self.beat("K4", rt=0.1, wait=False)
        self.wait_word("guidance")
        self.play(*show("laya1"), run_time=1.5)
        self.wait_word("0.60")
        self.play(FadeIn(cd["laya1"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("laya1"))[:20]], run_time=1.0)
        self.hold()

        # ---- K5 the spread
        xs = X["laya1"]
        band = Rectangle(width=2.3, height=yy(hi) - yy(lo) + 0.1, color=AMBER, stroke_width=3, fill_color=AMBER, fill_opacity=0.12).move_to([xs, (yy(lo) + yy(hi)) / 2, 0])
        brace = lab("5% to 50%", 17, AMBER, weight="BOLD").next_to(band, RIGHT, buff=0.12).shift(DOWN * 0.05)
        self.beat("K5", rt=0.1, wait=False)
        self.wait_word("spread")
        self.play(Create(band), run_time=0.8)
        self.wait_word("five")
        self.play(FadeIn(brace), run_time=0.7)
        self.wait_word("not")
        self.play(Indicate(band, color=AMBER, scale_factor=1.05), run_time=0.8)
        self.hold()
        self.play(FadeOut(band), FadeOut(brace), run_time=0.4)

        # ---- K6 LLM, no guidance
        self.beat("K6", rt=0.1, wait=False)
        self.wait_word("language")
        self.play(*show("llm0"), run_time=1.5)
        self.wait_word("0.36")
        self.play(FadeIn(cd["llm0"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("llm0"))[:20]], run_time=1.0)
        self.hold()

        # ---- K7 LLM, guidance
        self.beat("K7", rt=0.1, wait=False)
        self.wait_word("guidance")
        self.play(*show("llm1"), run_time=1.5)
        self.wait_word("0.70")
        self.play(FadeIn(cd["llm1"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("llm1"))[:20]], run_time=1.0)
        self.wait_word("second")
        self.play(Indicate(cd["llm1"][2], color=AMBER, scale_factor=1.12), run_time=0.9)
        self.hold()

        # ---- K7B the token probability
        self.beat("K7B", rt=0.1, wait=False)
        self.wait_word("token")
        self.play(*show("llm2"), run_time=1.5)
        self.wait_word("0.64")
        self.play(FadeIn(cd["llm2"][2]), *[Indicate(d, color=WHITE, scale_factor=1.8) for d in list(real_dots("llm2"))[:20]], run_time=1.0)
        self.hold()

        # ---- K8 the rule wins on numbers
        self.beat("K8", rt=0.1, wait=False)
        self.wait_word("noise")
        self.play(*[Indicate(cd[k][2], color=WHITE, scale_factor=1.1) for k in ("rule", "laya1", "llm1", "llm2")], run_time=1.0)
        self.wait_word("rule")
        self.play(Indicate(cd["rule"][3], color=GREEN_C, scale_factor=1.04), Indicate(cd["rule"][2], color=GREEN_C, scale_factor=1.15), run_time=1.0)
        self.wait_word("free")
        self.play(Indicate(cd["rule"][2], color=GREEN_C, scale_factor=1.15), run_time=0.8)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
