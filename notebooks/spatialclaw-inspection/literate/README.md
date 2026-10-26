# Literate notebooks for SpatialClaw Part 5

Six notebooks that follow `posts/series/spatialclaw-inspection/05-sandbox.qmd` section by section. Each one explains the idea with a daily-life analogy, shows the real source, and runs it, so the outputs are real. They sit next to the code they explain (`../minimal_loop`, `../sandbox`) and import it directly.

| Notebook | Post section | Runtime |
|---|---|---|
| `05a_image_and_runner` | design, image and runner | 1 s |
| `05b_host_session_and_loop` | host side, loop change, 9 control-flow tests | 25 s |
| `05c_trusting_nothing` | validating every container reply (added after a security review) | 5 s |
| `05d_replay_equivalence` | replay of 634 recorded cells | 50 s |
| `05e_attacks` | 15 attacks, old kernel against container | 45 s |
| `05f_cost_live_run_and_limits` | cost, live run (read from saved results, not re-run), limits | 15 s |

Run from this folder, with Docker and the `sc-inspect-sandbox` image available:

```bash
cd notebooks/spatialclaw-inspection/literate
../.venv/bin/jupyter lab            # or open in VS Code with the project venv as kernel
# headless, in order, outputs written into the files:
../.venv/bin/python run_all.py 05a_image_and_runner 05b_host_session_and_loop 05c_trusting_nothing 05d_replay_equivalence 05e_attacks 05f_cost_live_run_and_limits
```

The model is a scripted stand-in wherever a notebook says so; `05f` reads the saved live run instead of calling the model server.
