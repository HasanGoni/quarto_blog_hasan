"""Part 14: narration for the three Motion Canvas chapters (desk, fits, brief). A Manim scene that only speaks: make_chapter.sh turns LINES into the
warm cloned voice and writes assets/um/timeline_<slug>.warm.json (word times). The Motion Canvas scenes read that timeline and the extracted audio;
the visuals live in mc/src/scenes/p14_*.tsx.   Build:  VOICE_MOD=p14_voice ./make_mc_chapter.sh p14_desk DeskVoice"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".hermes" / "scripts"))
from manim import *  # noqa: F403
from video_speech import to_markup
from vidbase import NarratedScene

config.background_color = "#181818"

DESK = {
    "D1": "An agent has a small **desk**: its context window. Here, sixty-five thousand tokens.",
    "D2": "Your data is the **library**, and it is far bigger than the desk.",
    "D3": "Retrieval fails in three ways. It **searches** badly, it carries too **much**, or it carries the **wrong** pages.",
    "D4": "So the harness decides how much of the library crosses over. Search by **text**, read in **pages**, and pass **paths**, not whole files.",
}
FITS = {
    "F1": "Next, two hundred reports, about twenty-three thousand tokens. That **fits**.",
    "F2": "Pasting all of them in answered the counting question **three** out of three. Every tool setup got **zero**.",
    "F3": "My first run said the opposite. I had capped the answer at twelve hundred tokens, and the model was **cut** off mid-count.",
    "F4": "So first ask whether it **fits**. The checklist is the plan for when it **stops** fitting.",
}
BRIEF = {
    "B1": "A sub-agent that reads one report needs about three hundred and **sixty-four** tokens. With ninety-nine other reports it needs nearly fourteen **thousand**.",
    "B2": "All **eighty-four** runs were exactly right. I saw no context rot at this size. The saving was **thirty-eight** times fewer tokens.",
    "B3": "Visual search works too. Of the four nearest DINOv3 neighbours, **seventy-three** percent share the part class. Random neighbours: **ten** percent.",
    "B4": "But with twenty small records, the plain **payload** was cheapest. Small results do not need **offloading**.",
}
PRACTICE = {
    "P1": "Here is what I would do, in order, and where each one meets the earlier **parts**.",
    "P2": "First, **check** whether it fits. Pasting twenty-three thousand tokens beat every tool setup I built.",
    "P3": "If it does not fit, use the **cheapest** tool that works, like grep, then page the file. That is the rule from Part **eight**: plain code first, then a small model, then a language model.",
    "P4": "Pass **paths** only when results are big, and give a sub-agent a **narrow** brief. It is Part seven's split again: code owns the order, the model owns the judgment.",
    "P5": "And **look** at individual answers, because that is how I found my own bug. Part eight had one too.",
    "P6": "Three things I did not test: **skills**, **injection**, and stale documents. They belong to the later Deep Agents and security posts.",
}
FRAMEWORKS = {
    "W1": "Same model, same seventeen questions, three **frameworks**: my own loop, Deep Agents, and Google **ADK**.",
    "W2": "On the needle in the big table, my loop needed about twenty-eight hundred tokens per right answer.",
    "W3": "ADK with my tools, five **thousand**. ADK with its own tools, eight thousand.",
    "W4": "Deep Agents, **thirteen** thousand.",
    "W5": "Out of seventeen, Deep Agents scored six, or ten with a sub-agent. ADK scored nine, or eight.",
    "W6": "ADK with its own tools, **twelve**. My loop, twelve, with its best set-up for each task.",
    "W7": "Most failures were not wrong answers. They were runs that **never** answered. Deep Agents hit the forty-call cap on eighteen of thirty-four runs.",
    "W8": "In one run it had the answer after the first search, and then **re-read** the same line again and again.",
    "W9": "A guard that stops repeated calls saved tokens, but it did not fix the **loop**.",
    "W10": "And ADK's own shell runs with your environment variables. I ran it inside a Docker box with no **network**.",
    "W11": "So set a **cap**, read the loops, and box the shell. This was one local model, not **Gemini**.",
}


class _Voice(NarratedScene, Scene):
    def construct(self):
        for k in self.LINES:
            self.beat(k, wait=True)
        self.wait(0.8)
        self.save_timeline()


class DeskVoice(_Voice):
    SLUG, LINES = "p14_desk", DESK


class FitsVoice(_Voice):
    SLUG, LINES = "p14_fits", FITS


class BriefVoice(_Voice):
    SLUG, LINES = "p14_brief", BRIEF


class PracticeVoice(_Voice):
    SLUG, LINES = "p14_practice", PRACTICE


class FrameworksVoice(_Voice):
    SLUG, LINES = "p14_frameworks", FRAMEWORKS
