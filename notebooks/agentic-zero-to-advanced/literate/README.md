# Literate notebooks for Part 14 (Context Engineering Starts With Retrieval)

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
uv pip install ipykernel nbclient nbformat            # once, into the project venv
cd literate
../.venv/bin/python run_all.py 14a_desk_and_archive 14b_harness_switches 14c_results_t1_to_t5 14d_subagent_brief 14e_two_frameworks 14f_loops_guard_sandbox
```

`nbformat` reads and writes the `.ipynb` file (a notebook is JSON). `nbclient` runs every cell in a headless Jupyter kernel and stores the outputs, like Run All without opening Jupyter. The notebooks need the same `.env` as the scripts (the local vLLM endpoint) and set `AGENTIC_ZERO_MODEL=qwen3.8-27b` if it is not set.
