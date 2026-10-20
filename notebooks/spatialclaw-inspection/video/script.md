# SpatialClaw for Inspection, Part 1 video: script (DRAFT, not rendered)

Voice: warm tone, cloned voice. Tool per scene: Manim for 1 to 3 (zoom tour, numbers), Motion Canvas
would also fit scene 1 and 4 (pure diagram and text). Pick when we build.
Yellow is the subtitle highlighter only. `**word**` marks the highlighted words.
Every number below comes from the paper text (Table 2) or from my own smoke run. Anything drawn
that is not a real measurement is labelled "stand-in" on screen.

## Scene 1: The notebook that remembers (about 50 s)
Visual: three columns (single-pass, menu, kernel) from the whiteboard diagram; the third column lights up.
1. Most agents with tools get a **menu**: pick a tool, get an answer, pick another.
2. SpatialClaw gives the model a **notebook** instead, one that remembers everything.
3. The model writes **one cell**, looks at what came out, then writes the next cell.
4. Nothing is forgotten between steps, and that is the **whole trick**.

## Scene 2: One cell at a time (about 75 s)
Visual: the five real cells from the smoke run, variables piling up on a side shelf. Label: "stand-in
agent, synthetic pad, threshold instead of SAM3".
1. Here is a real run, with a **stand-in** agent and a synthetic solder pad.
2. Cell one finds the dark voids. Cell two reuses them and counts **three voids**.
3. Together they cover **6.43 percent** of the pad.
4. Cell three makes a mistake, and the notebook answers with a **traceback**.
5. Cell four fixes it, because the earlier variables are still **alive**.
6. A failed step costs **one cell**, not the whole run.

## Scene 3: How much is the idea? (about 75 s)
Visual: bar chart 53.4, 56.7, 59.9 (paper Table 2, average of 20 benchmarks, Gemma4-31B), then the
BLINK row (75.7 without tools, 73.4 with SpatialClaw).
1. The paper reports **59.9 percent** on average, with the same model throughout.
2. With no tools the score is **53.4**, and a plain tool-call agent reaches **56.7**.
3. So the notebook idea is worth about **three points**, and the tools themselves about **three more**.
4. And on the BLINK benchmark, the no-tool model scored **75.7** against 73.4 with tools.
5. When the question is mostly recognition, tools can **hurt**.

## Scene 4: What we keep for inspection (about 70 s)
Visual: keep, swap, drop columns from the diagram; then the word REVIEW replacing a guess.
1. For inspection we **keep** the notebook, the loop, and the images shown back to the model.
2. We **drop** depth, video and the cluster, because an X-ray is not a scene.
3. We swap the free-form answer for a **typed verdict**.
4. And when the agent runs out of steps, it must say **review**, never a guess of pass.
5. Next time we write the smallest loop, and we **run it** before we claim anything.
