# Literate notebooks for Parts 9 and 14

## Part 9 (Human in the Loop): `09-human-in-the-loop.qmd`

Five notebooks. Cells marked "Your turn" are meant to be edited and rerun (09b: answer a review yourself and compare with the exact answer; 09d: change the policy and the reviewer).

| Notebook | Post section | Model server? | Runtime |
|---|---|---|---|
| `09a_where_a_human_belongs` | what is inside each verdict, what the reviewer sees | no | 2 s |
| `09b_pause_and_resume` | `interrupt()`, bad answers, the replay gotcha, a restart; **you review an image** | no | 3 s |
| `09c_who_to_ask` | review budget: deferral curves from the Part 8 judges | no (saved runs) | 1 s |
| `09d_does_the_human_help` | policies, reviewer quality, rubber stamp; **change the settings** | no (saved runs) | 3 s |
| `09e_queue_and_failures` | the queue, timeouts, refused answers, the test suite | one cell | 1 min |

```bash
cd notebooks/agentic-zero-to-advanced/literate
../.venv/bin/python run_all.py 09a_where_a_human_belongs 09b_pause_and_resume 09c_who_to_ask 09d_does_the_human_help 09e_queue_and_failures
```

To work in VS Code, pick the interpreter `notebooks/agentic-zero-to-advanced/.venv/bin/python` (Select Kernel, then Python Environments, then Enter interpreter path).

## Part 14 (Context Engineering Starts With Retrieval)

Six notebooks that follow `posts/series/agentic-zero-to-advanced/14-context-engineering-retrieval.qmd` section by section. Each explains the idea with a daily-life picture, shows the real source, and runs it. Outputs are saved in the files.

| Notebook | Post section | Model server? | Runtime |
|---|---|---|---|
| `14a_desk_and_archive` | the desk and the library, the archive | no | 4 s |
| `14b_harness_switches` | the five switches, offload rule, three live runs | yes | 20 s |
| `14c_results_t1_to_t5` | results table, bars, the 1,200-token bug | no (saved runs) | 5 s |
| `14d_subagent_brief` | T6 | yes (2 calls) | 15 s |
| `14e_two_frameworks` | Deep Agents and Google ADK | no (saved runs) | 1 s |
| `14f_loops_guard_sandbox` | loops, repeat guard, `Execute` sandbox | Docker only | 1 s |

The T1 to T5 runs, the framework runs and the guard test took hours, so 14c, 14e and part of 14f read the files in `../out/` instead of repeating them. Each notebook says which cells are live.

```bash
cd notebooks/agentic-zero-to-advanced
uv sync                                               # ipykernel, nbclient, nbformat are in pyproject.toml
cd literate
../.venv/bin/python run_all.py 14a_desk_and_archive 14b_harness_switches 14c_results_t1_to_t5 14d_subagent_brief 14e_two_frameworks 14f_loops_guard_sandbox
```

`nbformat` reads and writes the `.ipynb` file (a notebook is JSON). `nbclient` runs every cell in a headless Jupyter kernel and stores the outputs, like Run All without opening Jupyter. The notebooks need the same `.env` as the scripts (the local vLLM endpoint) and set `AGENTIC_ZERO_MODEL=qwen3.8-27b` if it is not set.

## What the Part 9 notebooks read

`09a`, `09b`, `09d` and `09e` need the synthetic void dataset (the same one Parts 2 to 8 use, at `notebooks/agentic-factory-segmentation/data/void_dataset_v1`) because they draw X-ray images and score against the exact answer. They also read the saved Part 7 run (`out/pipeline_results.json`) and, for `09c`, the saved Part 8 judge results (`out/typed/`). Those result files are committed (force-added, `out/` is otherwise ignored) so the notebooks run from a fresh clone once the dataset exists.
