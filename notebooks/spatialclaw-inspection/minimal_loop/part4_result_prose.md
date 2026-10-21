Same 12 images as Part 3, 14 turns allowed, **each set-up run twice**. Ground truth is the exact void ratio from the mask.

| 24 episodes per set-up (12 images x 2 runs) | A: text only | B: + `show()` | C: + `show()` + tools | "Always PASS" |
|---|---|---|---|---|
| Median absolute error of the void ratio | 0.21 pp | 0.25 pp | **0.16 pp** | n/a |
| 90th-percentile absolute error | 1.62 pp | 2.80 pp | **0.99 pp** | n/a |
| Mean absolute error | **0.71 pp** | 0.90 pp | 1.44 pp | n/a |
| Verdicts that match ground truth | 19 of 24 | 17 of 24 | 19 of 24 | 18 of 24 |
| False PASS (a real FAIL called PASS) | 2 | 2 | 2 | 6 |
| False FAIL | 2 | 4 | 3 | 0 |
| Ended in REVIEW | 1 | 1 | 0 | n/a |
| Mean turns used (of 14) | 11.2 | 8.0 | **7.2** | n/a |
| Mean tokens per image | 41,491 | 24,750 | **23,444** | n/a |

Three things I am willing to say, and one I am not.

1. **Run-to-run noise is as large as the differences between set-ups.** Run the very same set-up twice on the
   very same 12 images and the verdict changes on 3 to 5 of them. Condition C scored a mean error of 2.63 pp in its
   first run and **0.26 pp** in its second. If I had run each set-up once, I could have written either "tools make it
   much worse" or "tools make it much better", and both would have been wrong.
2. **Budget was part of Part 3's problem.** Part 3's agent used the last or second-to-last of 8 turns on all 12
   images. Given 14 turns, text-only A used about 11 on average and submitted early on some images; it also ran out
   once (REVIEW), which is the safe outcome, not a guessed PASS.
3. **Looking plus tools (C) was the *efficient* condition, and its typical error was the lowest**: about 40 %
   fewer tokens than text-only A, and the lowest median and 90th-percentile error. `show()` alone (B) bought nothing I
   can measure.
4. **What I will not say: that C is more accurate.** Its verdict accuracy (19 of 24) equals A's, and one run of C
   submitted **28.99 %** for an image whose true value is 0.65 % (details below). One catastrophic number in 24
   episodes is exactly the kind of thing a production system is judged on, and 24 episodes cannot tell me how often it
   happens.

On raw verdict agreement, none of the three is clearly better than the trivial rule (19, 17 and 19 of 24 against
18 for "always PASS"). What all three agent set-ups do better is **false PASS**: 2 of 24 against 6, at the price
of 2 to 4 false alarms.
