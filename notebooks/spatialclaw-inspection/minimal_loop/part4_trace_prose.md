- **The pictures were used, not ignored.** In condition B the model called `show()` in 60 of its 101 cells, in C in
  58 of 94 (first runs). It checked masks with `overlay`, zoomed with `crop`, and once caught its own mistake: in
  `val_00001` (B) it printed a patch, saw values around 30 to 50, and wrote in a code comment *"Wait - the part is DARK (values
  ~30-50) and background is BRIGHT (~220-255)! I had it backwards!"* That is the case for `show()`: a picture catches an assumption a number does not.
- **But seeing did not stop the blunders.** Right after that rescue, the same episode computed 4.33 % (true 0.61 %)
  because a 1,227 px region that the model itself called "the background hole, NOT a void" stayed in the sum
  (the exact mask of that image has an enclosed cut-out of about 1,080 px, the same thing as in the section above). The agent noticed, wrote the right sentence in a code comment, and the
  code did not do what the comment said.
- **The 28.99 % in C, first run, `val_00003`.** An 8,583 px region (29 % of the part) was labelled "VOID" by a
  threshold on local contrast. The image also has a 2,805 px enclosed cut-out that `part_mask` counts as part, and
  in turn 13 the model itself wondered whether the region was "a gradient artifact". In turn 12 the model overlaid it and wrote
  *"confirm they are the small round voids, not the whole part"*; in turn 13 it printed that 31.7 % of the part
  was above its threshold, which is a clear warning; in turn 14, its last turn, it submitted the number anyway.
  The rule layer did exactly what it should: 28.99 > 1.5, FAIL. Here a wrong number happened to land on the safe side
  of the limit. A wrong number can also land on the unsafe side. **A plausibility check in the rule layer** (a
  measurement far outside anything seen on good parts goes to REVIEW, not FAIL or PASS) is the obvious guard, and it
  belongs in Part 7. I did not add it here, and I did not retro-fit it to these runs.
- **Hard images stay hard.** `val_00008` (three real voids, true 5.02 %) was under-measured in all five runs without tools
  (0.69 to 2.22 %, including Part 3's), and measured 6.69 % and 4.25 % in the two C runs. It is the one image where
  tools clearly helped, and it is one image.
- **Fewer turns, same verdicts.** C used 7.2 turns on average over both runs, A used 11.2. My reading, not tested
  separately: with a ready `part_mask` and `label_blobs`, the first turns of Part 3 (find the part, label the blobs)
  collapse into one call.
