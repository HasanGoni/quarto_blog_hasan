Four checks, all run for real.

| Check | Result |
|---|---|
| Control-flow tests in the container (scripted stand-in model) | 9 of 9 paths ok, including "alarm disabled", memory bomb, forged reply, process exit |
| Replay of **634 cells the model wrote in Part 4** through the container | 634 of 634 identical after one fix (the first replay found 1 difference in 634) |
| Attack battery, 15 cells | the old kernel let through file reads, file writes, a connection to a host service and a writable Docker socket; the container blocked every one of them |
| One live run of the 12 DEV images, model code in the container | 12 of 12 produced a measurement, 0 REVIEW, 0 sandbox errors |

And three things the sandbox did **not** do, so nobody leaves with the wrong idea:

1. **It did not make the numbers right.** A sandbox keeps wrong code away from the host. It says nothing about whether
   the measurement is correct.
2. **It is not a guarantee.** The 15 attacks are the ones I thought of. A container shares the host kernel.
3. **It did not change the cost.** Start-up is about 0.65 s in both set-ups; the container answers a cell faster
   (0.3 ms against 6.8 ms).

The live run's accuracy (11 of 12 verdicts, mean error 0.19 pp) looks better than Part 4's runs. **Do not read that
as "the sandbox helps".** It is one run of a noisy model; Part 4's two unsandboxed runs of the same set-up scored
mean errors of 2.63 and 0.26.
