"""Part 6 video chapters: the Part 5 agent loop as a LangGraph graph. Manim, narrated, public (no X-ray images, only diagrams).

Chapter 1 (GraphState): state, nodes. Chapter 2 (GraphEdges): edges, a token walks the graph. Chapter 3 (GraphResult): pause + the real numbers.
Colour rule: yellow is only the subtitle highlighter. Text is white or cyan; rings and arrows use cyan/green/amber.
Build:  ./make_chapter.sh p6_graph GraphState p6_state     (about 2 min each, run in the foreground)"""
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
CYAN, AMBER = "#6EC6FF", "#F2A93B"

LINES_STATE = {
    "S1": "Last time we wrote the agent as a **loop**. This time we draw the same agent as a **graph**.",
    "S2": "A graph has three parts: **state**, **nodes**, and **edges**.",
    "S3": "The state is a **whiteboard**. Every step **reads** it, and every step **writes** on it.",
    "S4": "A node is just a **Python function**. It reads the board, and returns **only what changed**.",
    "S5": "Messages get **appended**. The step counter gets **replaced**.",
}
LINES_EDGES = {
    "E1": "Now the **edges**. An edge decides who goes **next**.",
    "E2": "After the agent runs, one edge asks: did the model request a **tool**?",
    "E3": "If **yes**, we run the tools node, the **result** lands on the board, and we ask the model **again**.",
    "E4": "If the model only **talked**, we nudge it once, and ask **again**.",
    "E5": "And when the report is **submitted**, or the step **limit** is hit, we **stop**.",
}
LINES_RESULT = {
    "R1": "Because the state is saved after every node, we can **pause** the graph right before a tool runs.",
    "R2": "We look at what the model wants to do, and then we **resume**.",
    "R3": "Then we run **400** images. Same model, same tools, same scoring as Part 5.",
    "R4": "On the held-out images the graph finds **64** real voids, and the loop found **63**. **218** false alarms, against **215**. Same quality, **by design**.",
    "R5": "So the win is not accuracy. The flow is now **drawable**, **pausable**, and **testable**.",
}
RES = json.load(open(Path(__file__).resolve().parent.parent / "out" / "graph_agent_summary.json")) if (Path(__file__).resolve().parent.parent / "out" / "graph_agent_summary.json").exists() else None


def lab(s, size=26, color=WHITE, **kw):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=FONT, font_size=size, color=color, **kw)


def code(s, size=22, color=WHITE):
    return MarkupText(s.replace("&", "&amp;").replace("<", "&lt;"), font=MONO, font_size=size, color=color)


def card(w, h, color, fill=0.1, sw=4):
    return RoundedRectangle(corner_radius=0.18, width=w, height=h, color=color, stroke_width=sw, fill_color=color, fill_opacity=fill)


def node(title, sub, color, w=3.0, h=1.2):
    c = card(w, h, color, 0.15)
    t = lab(title, 30, WHITE, weight="BOLD").move_to(c.get_center() + UP * 0.2)
    s = lab(sub, 18, color).move_to(c.get_center() + DOWN * 0.3)
    return VGroup(c, t, s)


class GraphState(NarratedScene, Scene):
    SLUG = "p6_state"
    LINES = LINES_STATE

    def construct(self):
        # ---- S1: the loop becomes a graph
        loop = VGroup(card(5.4, 2.6, GREY_B, 0.1), lab("Part 5", 24, GREY_B), code("for step in range(12):\n    ask the model\n    run the tool\n    if done: break", 21))
        loop[1].move_to(loop[0].get_center() + UP * 1.0)
        loop[2].move_to(loop[0].get_center() + DOWN * 0.2)
        loop.move_to([0, 1.5, 0])
        mini = VGroup()
        pts = {"agent": [2.4, 1.6], "tools": [5.0, 2.7], "nudge": [5.0, 0.5]}
        for k, p in pts.items():
            mini.add(VGroup(card(2.0, 0.8, CYAN, 0.15), lab(k, 24, WHITE, weight="BOLD")).move_to([*p, 0]))
        for k in ("tools", "nudge"):
            mini.add(Arrow([3.4, 1.6, 0], [4.0, pts[k][1], 0], buff=0.05, color=GREY_B, stroke_width=5, max_tip_length_to_length_ratio=0.25))
        mini.scale(1.2).move_to([0, 1.3, 0])
        graph_t = lab("Part 6", 26, CYAN).move_to([0, 3.55, 0])
        self.beat("S1", rt=0.1, wait=False)
        self.play(FadeIn(loop), run_time=0.8)
        self.wait_word("graph")
        self.play(FadeOut(loop), run_time=0.5)
        self.play(FadeIn(mini), FadeIn(graph_t), run_time=1.0)
        self.hold()

        # ---- S2: three parts
        self.play(FadeOut(mini), FadeOut(graph_t), run_time=0.5)
        parts = [("state", "what we remember", CYAN), ("nodes", "what we do", GREEN_C), ("edges", "who goes next", AMBER)]
        cards = []
        for (name, sub, col), x in zip(parts, (-4.6, 0.0, 4.6)):
            c = card(3.9, 2.1, col, 0.12, 5)
            cards.append(VGroup(c, lab(name, 40, WHITE, weight="BOLD").move_to(c.get_center() + UP * 0.35), lab(sub, 24, col).move_to(c.get_center() + DOWN * 0.45)).move_to([x, 1.3, 0]))
        self.beat("S2", rt=0.1, wait=False)
        for w, cd in zip(("state", "nodes", "edges"), cards):
            self.wait_word(w)
            self.play(FadeIn(cd), run_time=0.6)
        self.hold()

        # ---- S3: the whiteboard
        self.play(FadeOut(Group(*cards)), run_time=0.5)
        board = card(6.6, 4.5, CYAN, 0.06, 5).move_to([-3.4, 1.2, 0])
        b_title = lab("state", 28, CYAN, weight="BOLD").move_to(board.get_top() + DOWN * 0.4)
        rows_spec = [("messages", "[system, user]"), ("trace", "[]"), ("step", "0"), ("tokens", "0"), ("report", "None")]
        rows = []
        for i, (k, v) in enumerate(rows_spec):
            y = board.get_top()[1] - 1.0 - i * 0.62
            rk = lab(k, 24, WHITE, weight="BOLD").move_to([-6.2, y, 0], aligned_edge=LEFT)
            rv = code(v, 22, GREY_B).move_to([-4.0, y, 0], aligned_edge=LEFT)
            rows.append((rk, rv))
        nd = node("agent", "a node", GREEN_C, 3.0, 1.3).move_to([4.2, 1.9, 0])
        a_read = Arrow(board.get_right() + UP * 0.5, nd.get_left() + UP * 0.1, buff=0.1, color=GREY_B, stroke_width=6)
        a_write = Arrow(nd.get_left() + DOWN * 0.3, board.get_right() + DOWN * 0.5, buff=0.1, color=GREEN_C, stroke_width=6)
        l_read = lab("reads", 22, GREY_B).next_to(a_read, UP, buff=0.05)
        l_write = lab("writes", 22, GREEN_C).next_to(a_write, DOWN, buff=0.05)
        self.beat("S3", rt=0.1, wait=False)
        self.wait_word("state")
        self.play(FadeIn(board), FadeIn(b_title), run_time=0.7)
        self.wait_word("whiteboard")
        self.play(*[FadeIn(m) for r in rows for m in r], run_time=1.4)
        self.wait_word("reads")
        self.play(FadeIn(nd), GrowArrow(a_read), FadeIn(l_read), run_time=0.8)
        self.wait_word("writes")
        self.play(GrowArrow(a_write), FadeIn(l_write), run_time=0.8)
        self.hold()

        # ---- S4: a node is a function that returns only what changed
        fn = code("def agent_node(state):\n    ...ask the model...\n    return {", 18)
        fn2 = code('  "messages": [new_message],\n  "step": 1 }', 18, GREEN_C)
        fnbox = VGroup(card(5.4, 2.0, GREEN_C, 0.08), VGroup(fn, fn2).arrange(DOWN, aligned_edge=LEFT, buff=0.08)).move_to([4.3, -0.85, 0])
        fn2.align_to(fn, LEFT).shift(RIGHT * 0.45)
        fnbox[1].move_to(fnbox[0])
        self.beat("S4", rt=0.1, wait=False)
        self.wait_word("Python")
        self.play(FadeIn(fnbox[0]), FadeIn(fn), run_time=0.8)
        self.wait_word("only")
        self.play(FadeIn(fn2), run_time=0.8)
        self.hold()

        # ---- S5: append vs replace
        new_msg = code("[system, user, assistant]", 18, GREEN_C).move_to(rows[0][1], aligned_edge=LEFT)
        new_step = code("1", 22, GREEN_C).move_to(rows[2][1], aligned_edge=LEFT)
        tag_a = lab("appended", 24, GREEN_C).move_to([1.7, 3.15, 0])
        tag_arrow = Arrow([0.9, 3.0, 0], [0.0, 2.6, 0], buff=0, color=GREEN_C, stroke_width=5, max_tip_length_to_length_ratio=0.3)
        tag_r = lab("replaced", 24, AMBER).move_to([-2.2, rows[2][1].get_y(), 0])
        self.beat("S5", rt=0.1, wait=False)
        self.wait_word("appended")
        self.play(Transform(rows[0][1], new_msg), FadeIn(tag_a), GrowArrow(tag_arrow), run_time=0.9)
        self.wait_word("replaced")
        self.play(Transform(rows[2][1], new_step), FadeIn(tag_r), run_time=0.7)
        self.play(Indicate(rows[2][1], color=AMBER), run_time=0.7)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


class GraphEdges(NarratedScene, Scene):
    SLUG = "p6_edges"
    LINES = LINES_EDGES

    def construct(self):
        P = {"start": [-6.2, 1.0, 0], "agent": [-2.7, 1.0, 0], "tools": [2.6, 2.7, 0], "nudge": [2.6, -0.7, 0], "end": [6.2, 1.0, 0]}
        start = VGroup(card(1.5, 0.8, GREY_B, 0.1), lab("START", 20)).move_to(P["start"])
        end = VGroup(card(1.4, 0.8, GREY_B, 0.1), lab("END", 20)).move_to(P["end"])
        agent = node("agent", "ask the model", CYAN, 2.8, 1.3).move_to(P["agent"])
        tools = node("tools", "run what it asked", GREEN_C, 3.0, 1.2).move_to(P["tools"])
        nudge = node("nudge", "use a tool, or submit", AMBER, 3.0, 1.2).move_to(P["nudge"])

        def arr(a, b, color=GREY_B, dashed=False):
            ar = Arrow(a, b, buff=0, color=color, stroke_width=6, max_tip_length_to_length_ratio=0.18)
            return DashedVMobject(ar, num_dashes=14) if dashed else ar

        a_start = arr([-5.45, 1.0, 0], [-4.1, 1.0, 0])
        a_at = arr([-1.3, 1.45, 0], [1.1, 2.55, 0], CYAN)
        a_ta = arr([1.1, 2.15, 0], [-1.3, 1.15, 0], GREEN_C)
        a_an = arr([-1.3, 0.8, 0], [1.1, -0.45, 0], CYAN)
        a_na = arr([1.1, -0.95, 0], [-1.3, 0.42, 0], AMBER)
        a_te = arr([4.1, 2.7, 0], [5.55, 1.4, 0], GREEN_C)
        a_ne = arr([4.1, -0.7, 0], [5.55, 0.6, 0], AMBER, dashed=True)
        OFF = np.array([0, -0.83, 0])
        tok = Dot(np.array(P["start"]) + OFF, radius=0.15, color=WHITE).set_z_index(30)

        # ---- E1: the nodes
        self.beat("E1", rt=0.1, wait=False)
        self.play(FadeIn(Group(start, agent, tools, nudge, end), lag_ratio=0.15), run_time=1.6)
        self.play(GrowArrow(a_start), run_time=0.5)
        self.wait_word("next")
        self.hold()

        # ---- E2: the question after the agent
        q = lab("did the model request a tool?", 24, CYAN).move_to([-2.7, 2.35, 0])
        ring = card(3.1, 1.6, CYAN, 0.0, 7).move_to(P["agent"])
        self.beat("E2", rt=0.1, wait=False)
        self.wait_word("agent")
        self.play(Create(ring), run_time=0.6)
        self.wait_word("edge")
        self.play(GrowArrow(a_at), GrowArrow(a_an), run_time=0.8)
        self.wait_word("tool")
        self.play(FadeIn(q), run_time=0.6)
        self.hold()
        self.play(FadeOut(ring), FadeOut(q), run_time=0.4)

        # ---- E3: yes -> tools -> back to agent
        yes = lab("yes", 22, CYAN).move_to([-0.2, 2.45, 0])
        self.beat("E3", rt=0.1, wait=False)
        self.play(FadeIn(tok.move_to(np.array(P["agent"]) + OFF)), run_time=0.3)
        self.wait_word("yes")
        self.play(FadeIn(yes), tok.animate.move_to(np.array(P["tools"]) + OFF), run_time=1.0)
        self.wait_word("result")
        self.play(GrowArrow(a_ta), run_time=0.5)
        self.play(tok.animate.move_to(np.array(P["agent"]) + OFF), run_time=0.9)
        self.wait_word("again")
        self.play(Indicate(agent, color=CYAN, scale_factor=1.06), run_time=0.8)
        self.hold()
        self.play(FadeOut(yes), run_time=0.3)

        # ---- E4: talked -> nudge -> back to agent
        no = lab("no, it just talked", 22, AMBER).move_to([-2.9, -0.45, 0])
        self.beat("E4", rt=0.1, wait=False)
        self.wait_word("talked")
        self.play(FadeIn(no), tok.animate.move_to(np.array(P["nudge"]) + OFF), run_time=1.0)
        self.wait_word("again")
        self.play(GrowArrow(a_na), run_time=0.4)
        self.play(tok.animate.move_to(np.array(P["agent"]) + OFF), run_time=0.9)
        self.hold()
        self.play(FadeOut(no), run_time=0.3)

        # ---- E5: the two ways out
        self.beat("E5", rt=0.1, wait=False)
        self.wait_word("submitted")
        self.play(tok.animate.move_to(np.array(P["tools"]) + OFF), run_time=0.7)
        self.play(GrowArrow(a_te), tok.animate.move_to(np.array(P["end"]) + OFF), run_time=1.0)
        self.wait_word("limit")
        self.play(Create(a_ne), run_time=0.8)
        self.wait_word("stop")
        self.play(Indicate(end, color=WHITE, scale_factor=1.15), run_time=0.8)
        self.hold()
        self.wait(0.8)
        self.save_timeline()


class GraphResult(NarratedScene, Scene):
    SLUG = "p6_result"
    LINES = LINES_RESULT

    def construct(self):
        g, l = RES["test"]["graph"], RES["test"]["part5_loop"]
        # ---- R1/R2: pause before the tools node, look, resume (real output of graph_pause_demo.py)
        n_agent = node("agent", "ran once", CYAN, 2.6, 1.1).move_to([-3.6, 2.5, 0])
        n_tools = node("tools", "about to run", GREEN_C, 2.6, 1.1).move_to([3.6, 2.5, 0])
        link = Arrow(n_agent.get_right(), n_tools.get_left(), buff=0.1, color=GREY_B, stroke_width=6)
        bars = VGroup(Rectangle(width=0.2, height=0.8, fill_color=AMBER, fill_opacity=1, stroke_width=0), Rectangle(width=0.2, height=0.8, fill_color=AMBER, fill_opacity=1, stroke_width=0)).arrange(RIGHT, buff=0.18).move_to([0, 2.5, 0]).set_z_index(5)
        cons = card(12.4, 2.5, GREY_B, 0.08, 3).move_to([0, 0.15, 0])
        L = ["paused before node: ('tools',)", "model wants to call: find_candidates {\"image\": \"test_00020\"}", "messages so far: 3 | step: 1",
             "paused again before: ('tools',) | step: 2 | tool results so far: 1"]
        lines = VGroup(*[code(t, 19) for t in L]).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(cons).shift(LEFT * 0.0)
        real = lab("real output of graph_pause_demo.py", 18, GREY_B).next_to(cons, UP, buff=0.08).align_to(cons, RIGHT)
        self.beat("R1", rt=0.1, wait=False)
        self.play(FadeIn(n_agent), FadeIn(n_tools), GrowArrow(link), run_time=0.9)
        self.wait_word("pause")
        ring = n_tools[0].copy().set_stroke(AMBER, 6).set_fill(opacity=0)
        self.play(FadeIn(bars), FadeIn(ring), run_time=0.5)
        self.play(FadeIn(cons), FadeIn(real), FadeIn(lines[0]), FadeIn(lines[1]), FadeIn(lines[2]), run_time=1.0)
        self.hold()
        self.beat("R2", rt=0.1, wait=False)
        self.play(Indicate(lines[1], color=CYAN, scale_factor=1.04), run_time=1.0)
        self.wait_word("resume")
        self.play(FadeOut(bars), n_agent[2].animate.become(lab("ran twice", 18, CYAN).move_to(n_agent[2])), run_time=0.6)
        self.play(FadeIn(lines[3]), run_time=0.8)
        self.hold()

        # ---- R3/R4: the real numbers, loop vs graph
        self.play(FadeOut(Group(n_agent, n_tools, link, cons, real, ring, *lines)), run_time=0.6)
        hdr = ["", "real voids", "false alarms", "missed", "steps/image", "tokens/image"]
        rows = [("Part 5 loop", l["tp"], l["fp"], l["fn"], l["avg_steps"], l["avg_tokens"]), ("Part 6 graph", g["tp"], g["fp"], g["fn"], g["avg_steps"], g["avg_tokens"])]
        xs = [-4.9, -2.6, -0.6, 1.4, 3.3, 5.3]
        head = VGroup(*[lab(h, 17, GREY_B).move_to([x, 3.2, 0]) for h, x in zip(hdr, xs) if h])
        tab = VGroup()
        for r, y in zip(rows, (2.2, 1.2)):
            for j, (v, x) in enumerate(zip(r, xs)):
                m = lab(str(v), 28 if j else 24, WHITE if j else CYAN, weight="BOLD")
                tab.add(m.move_to([x, y, 0]) if j else m.move_to([-6.3, y, 0], aligned_edge=LEFT))
        frame = card(13.0, 2.9, GREY_B, 0.05, 3).move_to([0.0, 2.2, 0])
        note = lab("held-out images (200 shown of 400 run) · Qwen3-Coder-30B · temperature 0.2", 20, GREY_B).move_to([0, -0.25, 0])
        self.beat("R3", rt=0.1, wait=False)
        self.play(FadeIn(frame), FadeIn(head), FadeIn(note), run_time=0.8)
        self.wait_word("400")
        self.play(FadeIn(Group(*tab[0:6])), run_time=0.7)
        self.hold()
        self.beat("R4", rt=0.1, wait=False)
        self.wait_word("graph")
        self.play(FadeIn(Group(*tab[6:12])), run_time=0.8)
        self.wait_word("64")
        self.play(Indicate(tab[1], color=CYAN), Indicate(tab[7], color=CYAN), run_time=0.9)
        self.wait_word("218")
        self.play(Indicate(tab[2], color=AMBER), Indicate(tab[8], color=AMBER), run_time=0.9)
        self.hold()

        # ---- R5: why bother
        self.play(FadeOut(Group(frame, head, note, tab)), run_time=0.6)
        cards = []
        for (name, sub, col), x in zip((("drawable", "the flow is a picture", CYAN), ("pausable", "stop before any node", AMBER), ("testable", "force every edge", GREEN_C)), (-4.6, 0.0, 4.6)):
            c = card(3.9, 2.1, col, 0.12, 5)
            cards.append(VGroup(c, lab(name, 38, WHITE, weight="BOLD").move_to(c.get_center() + UP * 0.35), lab(sub, 22, col).move_to(c.get_center() + DOWN * 0.45)).move_to([x, 1.3, 0]))
        self.beat("R5", rt=0.1, wait=False)
        for w, cd in zip(("drawable", "pausable", "testable"), cards):
            self.wait_word(w)
            self.play(FadeIn(cd), run_time=0.6)
        self.hold()
        self.wait(0.8)
        self.save_timeline()
