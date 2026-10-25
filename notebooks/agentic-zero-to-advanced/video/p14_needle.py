"""Part 14 video, chapter "the needle in the big file" (Manim: real token counts as bars against the 65,536-token window).

Every number is read from out/ctx_summary.json via p14_facts.py and asserted there. Bars show what each strategy tried to load or actually used:
the table is about 113,500 tokens (file size / 4), the window is 65,536.
Colour rule: yellow is only the subtitle highlighter. Green = worked, red = refused, amber = worse.
Build:  ./make_chapter.sh p14_needle Needle p14_needle clone warm"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from video_speech import to_markup
from vidbase import NarratedScene

config.background_color = "#181818"
FONT = "DejaVu Sans"
CYAN, AMBER, RED, GREEN = "#6EC6FF", "#F2A93B", "#FF6B5E", "#5BD18B"
F = json.loads((Path(__file__).resolve().parent / "mc" / "src" / "facts_p14.json").read_text())

LINES = {
    "N1": "One question: find a single row in a four-thousand-row **table**. That table is over a hundred thousand **tokens**.",
    "N2": "Paste it in, and the server **refuses**. Give the model a whole-file **reader**, and that fails too. Zero out of **six**.",
    "N3": "Give it **grep** and paged reads, and it answers **six** out of six, with under three thousand tokens in total.",
    "N4": "Then I added a sub-agent tool. It dropped to **three** out of six. I do not know **why**.",
}


def lab(s, size=24, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


class Needle(NarratedScene, Scene):
    SLUG = "p14_needle"
    LINES = LINES

    def construct(self):
        t1 = F["t1"]
        # the narration says these; refuse to build if the data disagrees
        assert 100_000 < F["csv_tokens"] < 120_000 and F["window"] == 65536
        assert t1["stuff"]["overflow"] == 6 and t1["payload"]["overflow"] == 6
        assert (t1["harness"]["correct"], t1["harness"]["n"]) == (6, 6) and t1["harness"]["tokens"] < 3000
        assert (t1["harness+delegate"]["correct"], t1["harness+delegate"]["n"]) == (3, 6)

        X0, SC = -3.2, 7.0 / 120_000          # bar start x, units per token (120k tokens = 7.0 units)
        xs = lambda tok: X0 + tok * SC
        rows = [2.5, 1.4, 0.3, -0.8]
        win_x = xs(F["window"])
        axis = Line([X0, 3.2, 0], [X0, -1.45, 0], color=GREY_B, stroke_width=2)
        win = DashedLine([win_x, 3.2, 0], [win_x, -1.45, 0], color=CYAN, stroke_width=3)
        win_lab = lab("context window  65,536", 17, CYAN).move_to([win_x, 3.45, 0])
        names = ["paste the table in", "whole-file reader", "grep + paged reads", "+ a sub-agent tool"]
        row_labels = [lab(n, 20, WHITE).move_to([X0 - 0.15, y, 0], aligned_edge=RIGHT) for n, y in zip(names, rows)]
        for l, y in zip(row_labels, rows):
            l.move_to([X0 - 0.2 - l.width / 2, y, 0])

        def bar(y, tok, color, op=0.9):
            r = Rectangle(width=max(0.06, tok * SC), height=0.62, color=color, fill_color=color, fill_opacity=op, stroke_width=0)
            r.move_to([X0 + r.width / 2, y, 0])
            return r

        b_stuff = bar(rows[0], F["csv_tokens"], RED, 0.75)
        b_pay = bar(rows[1], F["csv_tokens"], RED, 0.75)
        b_har = bar(rows[2], t1["harness"]["tokens"], GREEN)
        b_sub = bar(rows[3], t1["harness+delegate"]["tokens"], AMBER)
        r_stuff = lab("refused: 6 of 6", 20, WHITE, weight="BOLD").move_to(b_stuff).align_to(b_stuff, LEFT).shift(RIGHT * 0.25)
        r_pay = lab("refused: 6 of 6", 20, WHITE, weight="BOLD").move_to(b_pay).align_to(b_pay, LEFT).shift(RIGHT * 0.25)
        r_har = lab(f"6 of 6 right, {t1['harness']['tokens']:,} tokens", 20, GREEN, weight="BOLD").next_to(b_har, RIGHT, buff=0.2)
        r_sub = lab(f"3 of 6 right, {t1['harness+delegate']['tokens']:,} tokens", 20, AMBER, weight="BOLD").next_to(b_sub, RIGHT, buff=0.2)
        note = lab(f"the table is about {F['csv_tokens']:,} tokens (file size divided by four)", 16, GREY_B).move_to([0.3, -1.85, 0])
        title_q = lab("find one row in a 4,000-row table", 20, GREY_B).move_to([-3.6, 3.45, 0])

        # ---- N1
        self.beat("N1", rt=0.1, wait=False)
        self.play(Create(axis), FadeIn(title_q), run_time=0.8)
        self.wait_word("table")
        self.play(FadeIn(win), FadeIn(win_lab), run_time=0.7)
        self.wait_word("tokens")
        self.play(FadeIn(note), run_time=0.6)
        self.hold()

        # ---- N2
        self.beat("N2", rt=0.1, wait=False)
        self.play(FadeIn(row_labels[0]), GrowFromEdge(b_stuff, LEFT), run_time=1.4)
        self.wait_word("refuses")
        self.play(FadeIn(r_stuff), run_time=0.5)
        self.wait_word("reader")
        self.play(FadeIn(row_labels[1]), GrowFromEdge(b_pay, LEFT), run_time=1.2)
        self.wait_word("six")
        self.play(FadeIn(r_pay), run_time=0.5)
        self.hold()

        # ---- N3
        self.beat("N3", rt=0.1, wait=False)
        self.wait_word("grep")
        self.play(FadeIn(row_labels[2]), GrowFromEdge(b_har, LEFT), run_time=0.8)
        self.wait_word("six")
        self.play(FadeIn(r_har), run_time=0.6)
        self.wait_word("three")
        self.play(Indicate(b_har, color=GREEN, scale_factor=1.6), run_time=0.8)
        self.hold()

        # ---- N4
        self.beat("N4", rt=0.1, wait=False)
        self.wait_word("sub-agent")
        self.play(FadeIn(row_labels[3]), GrowFromEdge(b_sub, LEFT), run_time=0.9)
        self.wait_word("three")
        self.play(FadeIn(r_sub), run_time=0.6)
        self.wait_word("why")
        self.play(Indicate(r_sub, color=AMBER, scale_factor=1.12), run_time=0.8)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
