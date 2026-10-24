"""Part 8: narration for the three Motion Canvas chapters. A Manim scene that only speaks: make_chapter.sh turns LINES into your warm cloned
voice and writes assets/um/timeline_<slug>.warm.json (word times). The Motion Canvas scenes read that timeline and the extracted audio;
the visuals live in mc/src/scenes/*.tsx.   Build:  ./make_chapter.sh p8_voice ToolsVoice p8_tools clone warm"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from video_speech import to_markup
from vidbase import NarratedScene

config.background_color = "#181818"

TOOLS = {
    "T1": "In Part 7, every step was one of two kinds of tool: plain **code**, or a language **model**.",
    "T2": "There is a third kind in between. A small **typed** model reads text, and answers a typed question with a **probability**.",
    "T3": "Code is exact. One rule check took about **forty** nanoseconds, and costs nothing.",
    "T4": "The typed model answered in about **twenty** milliseconds on the GPU, and about **five hundred** on a plain CPU, with no tokens at all.",
    "T5": "The language model is the only one that can **write**. One note took about **two and a half** seconds, and over two hundred tokens.",
    "T6": "In the end, the question is not which tool is best. It is which tool this **step** needs.",
}
NOTES = {
    "N1": "In Part 7, the model wrote a note for every flagged image. Here is a real **note**.",
    "N2": "A typed model reads it, and answers two questions: was this **escalated**, and is anything **uncertain**?",
    "N3": "It said **ninety-seven** percent, and **ninety-three** percent. Both match the record, and it took twenty milliseconds.",
    "N4": "Across all **520** real notes, it matched the record on **448**.",
    "N5": "I audited the other **72**. In **70** of them, the typed model was the one that was wrong.",
    "N6": "**Sixty-five** times it called a fine note uncertain, even though the note says confident defect.",
    "N7": "The last **two** were arguable. And **none** of the seventy-two was a real flaw in a note.",
    "N8": "A guard that finds no real problem is not worth running. When the writer is almost always right, a checker that is **eighty-six** percent right mostly flags good notes.",
}
CHOOSE = {
    "C1": "Now, which tool for which step? Here is what the measurements say.",
    "C2": "If the step is **numbers**, use a rule. On the box call, no model beat it by a margin I could measure, and it cost nothing.",
    "C3": "If the step is reading **text** into a typed answer, a small typed model is fast, and good at yes-or-no questions. It was weaker at **counting**.",
    "C4": "If the step must **write**, use the language model, and keep the facts in code.",
    "C5": "And measure on your own labeled data. A model that never says yes can still look **well** calibrated.",
}


class _Voice(NarratedScene, Scene):
    def construct(self):
        for k in self.LINES:
            self.beat(k, wait=True)
        self.wait(0.8)
        self.save_timeline()


class ToolsVoice(_Voice):
    SLUG, LINES = "p8_tools", TOOLS


class NotesVoice(_Voice):
    SLUG, LINES = "p8_notes", NOTES


class ChooseVoice(_Voice):
    SLUG, LINES = "p8_choose", CHOOSE
