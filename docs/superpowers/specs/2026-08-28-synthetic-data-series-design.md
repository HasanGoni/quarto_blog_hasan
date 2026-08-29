# Synthetic Data series — design

Date: 2026-08-28
Status: approved, pending implementation plan

## Motivation

Hasan's day-job computer vision work (semiconductor defect inspection — voids, cracks, bent
leads) is chronically short on labeled defect examples, especially rare defect classes. The
current manual workflow is cut-and-paste: find a real defective region, crop it, paste it into a
good image. That's slow, limited in variety, and doesn't scale to rare classes.

This series is a public exploration of synthetic-defect-generation methods for X-ray-style
industrial inspection imagery — starting from first-principles physics and working through the
generative-modeling alternatives, ending in an honest, measured comparison. It's written for
the same audience and to the same bar as the existing "Paper of the Week" deep-dive posts (real
code, real captured output, real results) and shares the "every post ends with an honest
limitation" ethos of the `local-ai-defect-inspection` series.

## Non-goals

- Not using any of Hasan's actual (confidential) semiconductor work imagery or equipment specs.
  Everything here uses public data and is built as if equipment access were unavailable — which
  mirrors the real constraint the physics-approximation post (Part 1) is written to solve.
- Not attempting to acquire or license the CXR-AD dataset (arXiv 2505.03412) as part of this
  series — no confirmed public download exists at time of writing. Worth revisiting later as a
  follow-up if it becomes available; not a blocker for this series.
- Not building a production defect-detection pipeline. The measured comparison in Part 9 uses
  a downstream task as an *evaluation signal* for the synthetic-data methods, not as a
  deliverable in its own right.

## Series placement and structure

New standalone series: `posts/series/synthetic-data/`. Considered folding into
`local-ai-defect-inspection` (same domain, already using GDXray) but rejected — that series'
framing (VLM eval, LinkedIn-first) doesn't fit a generative/physics-methods series. Cross-link
both directions instead.

Nine posts, in this order (expanded 2026-08-28 after Part 2 shipped — the physics-vs-diffusion
comparison landed, but the diffusion defect *shape* still doesn't read as a real defect; SD1.5's
plain inpainting mask barely constrains internal shape, and there was appetite to try both a
shape-conditioning add-on and a stronger base model before the final comparison, rather than
declare "diffusion" done after one checkpoint; expanded again 2026-08-29 to add joint
image+mask generation, inspired by NVIDIA's MAISI/NV-Generate-CTMR line of work for synthesizing
3D medical images with paired segmentation masks — see Part 5 below):

1. **Physics-based X-ray simulation** — Beer-Lambert forward model, calibrated from real GDXray
   images since no equipment access exists.
2. **Mask-conditioned diffusion fine-tuning** — LoRA fine-tune (SD1.5 inpainting) on real GDXray
   defect crops+masks. Shipped; honest finding was defect *texture* beats physics, defect *noise
   statistics* and *shape fidelity* don't yet — see Parts 3-5 below.
3. **ControlNet-conditioned diffusion** — same SD1.5 base, add a ControlNet (edge/scribble
   conditioning) fed the *real* Otsu-refined defect silhouette from Part 2's own dataset builder
   (`build_defect_dataset`) instead of a synthetic crack shape or a bare rectangular mask — tests
   whether the "doesn't look like a real defect" complaint is a shape-conditioning problem before
   reaching for a bigger model.
4. **Larger base-model generation (SDXL inpainting)** — checked `black-forest-labs/FLUX.1-Fill-dev`
   at implementation time (2026-08-29): confirmed gated (`gated: "auto"` on the HF API), conflicting
   with this repo's non-gated-only preference. User chose the non-gated substitute over accepting
   Flux's license: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (confirmed non-gated) — a
   straight swap-in for the base generator, same LoRA fine-tune recipe as Parts 2-3, still a real
   base-model-scale jump from SD1.5 (larger UNet, higher native resolution).
5. **Joint image+mask diffusion generation** — instead of feeding a mask in as a conditioning
   input (Parts 2-4's approach), train the model to generate the image *and* its defect mask
   together, so defect shape is learned from the real distribution of defect shapes rather than
   drawn by hand or borrowed from another crop. Directly inspired by NVIDIA's MAISI architecture
   (latent diffusion / rectified flow trained to jointly output a 3D medical volume and its
   segmentation), scoped down to 2D single-channel masks on GDXray. See Part 5 below for the
   adaptation.
6. **Combining what works** — take whichever of ControlNet-shape-conditioning (Part 3), Flux's
   higher base fidelity (Part 4), and joint generation's learned shape distribution (Part 5)
   actually helped (empirically, from Parts 3-5, not assumed), combine them, and layer Part 1's
   calibrated Poisson noise model on top in pixel space (mask region only, after compositing real
   pixels back in outside it) to close the noise-statistics gap Part 2 found. Real open question
   going in: does combining actually compound the gains, or do the improvements not stack.
7. **GAN-based synthesis** — small GAN (DFMGAN-style) trained on the same crops. A distinct third
   generative family, independent of the diffusion track above.
8. **Cut-paste + Poisson blending baseline** — the classical manual-workflow approach, formalized.
9. **Post-training a real detector on synthetic data (the payoff)** — real baseline, then
   post-train the same model separately on real+synthetic splits from every method (physics,
   diffusion, ControlNet, Flux, joint generation, the combined hybrid, GAN, cut-paste), same
   recipe across conditions, ablate the real:synthetic ratio, report which method's data actually
   moves a real downstream model, broken out by the two original goals (rare-defect augmentation
   vs. minimizing labeling effort). Real training runs and real metrics, not a one-shot eval.

Each post gets a `series:` frontmatter block and opens with a callout linking back to
`index.qmd`, per the repo's established series pattern (`CLAUDE.md`). Wiring into
`_quarto.yml` navbar/sidebar and `README.md`'s series list happens as part of implementation,
not this design doc.

## Data

**GDXray** (castings + welds groups) is the real-data anchor for all nine posts — publicly
downloadable, free for research/education use, already integrated in this repo
(`local-ai-defect-inspection` series uses it), and has pixel-level defect masks for some series.
No semiconductor-specific category exists in GDXray, but casting voids and weld-porosity defects
share the same underlying X-ray physics (density gap → attenuation drop → intensity signature)
as the semiconductor voids/cracks this series is ultimately in service of. Domain-mismatch
tradeoff is accepted the same way `local-ai-defect-inspection` already accepted it.

## Tooling

New `uv`-managed subproject: `notebooks/synthetic-data-xray/`, following the
`paper-rau-sam2`/`paper-hpma-sam3` pattern — scoped `pyproject.toml`, `.venv` gitignored,
`uv sync` to set up. Shared across all nine posts (common data-loading/GDXray utilities), with
per-post scripts/notebooks inside.

## Part 1 — Physics-based X-ray simulation

**Equipment parameters that would ideally inform the forward model** (unavailable here, listed
for completeness and as the post's "what to ask your equipment vendor" section):

1. Tube voltage (kVp) → effective photon energy → material attenuation coefficients μ(E)
2. Source-to-object / source-to-detector distances → geometric magnification M = SDD/SOD
3. Focal spot size → geometric unsharpness Ug = spot_size × (M−1)
4. Detector pixel pitch, MTF, bit depth → resolution, detector-side blur, saturation behavior
5. Exposure (mAs) / detector gain → photon count → noise floor (photon-limited, variance scales
   with signal — not flat Gaussian)
6. Material composition + known thickness/CAD → ground-truth μ per material

**Approximation approach, since none of the above is available:** invert the Beer–Lambert law
(I = I0·exp(−μ·t)) on a real *clean* GDXray image to recover an implied per-pixel
thickness/density map, rather than simulating from a CAD model that doesn't exist for this data.
Concretely:

- Fit an effective attenuation coefficient from intensity ratios between GDXray's real defect
  masks and surrounding background (the pixel-level masks give the needed ratio directly).
- Fit a noise model via a photon-transfer-curve-style analysis (variance vs. mean intensity
  across flat background patches).
- Fit a blur kernel from the edge-spread function at real material/defect boundaries.
- Invert a real clean image to a thickness map t(x,y).
- Edit the thickness map inside a mask (void = local thickness reduction, crack = thin
  discontinuity), then forward-render through the fitted attenuation + noise + blur model.

Output: a synthetic defect image *and* its ground-truth mask (free, since the mask was the
edit target), physically grounded rather than hallucinated, using only images GDXray already
provides. This becomes the baseline the later parts are compared against in Part 9.

## Part 2 — Mask-conditioned diffusion fine-tuning (shipped)

- Same GDXray defect crops + masks used to calibrate Part 1 (338 real (image, mask) pairs via
  `synth_xray.diffusion_data.build_defect_dataset`).
- LoRA fine-tune of `stable-diffusion-v1-5/stable-diffusion-inpainting` (non-gated; `Boogu-Image`
  was checked and rejected — it's instruction-text editing only, `pipe(image=..., prompt=...)`,
  no mask parameter anywhere).
- Compared output against Part 1's physics-rendered images on the same synthetic-crack mask.
- **Real, honest finding**: diffusion learns defect *texture* physics has no way to learn
  (convincing porosity-pattern continuation), but stays close to background-smooth on local noise
  (1.03 vs. physics's 1.87, vs. a 0.87 clean baseline, vs. 4.45 for a real defect) — the VAE's 8x
  spatial compression structurally resists fine photon-noise texture. Separately, the generated
  defect *shape* still doesn't read as authentically defect-like — SD-inpainting's plain mask
  conditioning tells the model what region to fill, not what shape the content inside should
  take. Parts 3-6 below exist specifically to chase that second gap before Part 9 declares a
  winner.
- Known gap not yet fixed as of this doc's last update: the demo doesn't composite the real
  unmasked pixels back into the output, so the *whole* generated image (not just the masked
  region) differs slightly from the real input due to the VAE round-trip — fix before Parts 3-6
  reuse this pipeline, since noise-matching in Part 6 only makes sense once the non-mask region is
  pixel-identical to the real input.

## Part 3 — ControlNet-conditioned diffusion

- Same SD1.5 base and LoRA recipe as Part 2. Add a ControlNet (edge/scribble-style conditioning,
  non-gated `lllyasviel/control_v11p_sd15_*` checkpoints) fed the *real* Otsu-refined defect
  silhouette already produced by Part 2's `build_defect_dataset` — not a synthetic crack shape,
  not a bare rectangular box mask.
- Directly targets Part 2's shape-realism gap: does telling the model the real internal shape of
  a defect (not just its bounding region) produce more authentic-looking output, independent of
  base-model scale?
- Compare against Part 2's un-conditioned output on the same masks/prompts, same evaluation
  approach (visual + the local-std noise-texture measurement Part 2 established).

## Part 4 — Flux-based mask-conditioned generation (shipped)

- Model-choice detour, resolved 2026-08-29: `black-forest-labs/FLUX.1-Fill-dev` confirmed gated
  via the HF API (`gated: "auto"`). First substitute considered, SDXL inpainting, was rejected on
  review — still a UNet, so it would only test scale within the same architecture family, not
  architecture itself, which is the actual point of this part. PixArt-Sigma (non-gated, genuinely
  DiT-based) was checked too but has no native `diffusers` inpainting pipeline — would need
  hand-rolled masking, a real step down from Parts 2-3's library-native approach. Decision: use
  the real Flux-Fill-dev — checked the existing `HF_TOKEN` against the gated repo via the HF API
  (`GET .../tree/main` → 200) before committing, rather than assuming either way.
- LoRA (`r=8, alpha=16`) targets the real attention-projection module names found by inspecting
  `transformer.named_modules()` directly: `to_q/to_k/to_v/to_out.0` (image stream) +
  `add_q_proj/add_k_proj/add_v_proj/to_add_out` (text stream) — Flux's dual-stream MMDiT design.
- Training loop reuses `FluxFillPipeline`'s own real internal methods (`encode_prompt`,
  `prepare_mask_latents`, `_encode_vae_image`, `_pack_latents`, `_prepare_latent_image_ids`)
  rather than re-deriving the packed-latent-sequence format by hand — verified against the
  pipeline's real `__call__` source before writing the loop, not guessed. Real flow-matching
  (rectified flow) objective, not DDPM — `FlowMatchEulerDiscreteScheduler.scale_noise`'s own
  formula, target velocity `noise - real_latents`.
- **Real result, genuinely surprising**: on the single test case Parts 1-3 all used, Flux's output
  is *less* visible than Part 2's smaller SD1.5 model (46.1 gray-level contrast at the injection
  site for Part 2 vs. 1.8 for Part 4) — scale alone did not close the shape-realism gap here, if
  anything the opposite on this one example. Local noise-texture std also clusters with Parts 2-3
  (0.79-0.94), well below physics, regardless of architecture — the VAE-latent-space noise
  limitation Part 2 found looks shared across every latent diffusion model tested, not specific to
  SD1.5. A real negative result, reported as such rather than reframed as a win.

## Part 5 — Joint image+mask diffusion generation

**The idea, and why it's different from Parts 2-4:** every prior diffusion part treats the mask
as a *conditioning input* — something drawn (Part 2's synthetic crack), or lifted from a real
defect crop (Part 3's ControlNet silhouette) — that tells the model *where* to generate, not
*what shape* to generate there. NVIDIA's MAISI/NV-Generate-CTMR line of work (3D CT/MRI synthesis
with paired segmentation masks) takes the opposite approach: train the model to generate the
image *and* its mask together, as a joint output, so the mask itself is a learned sample from the
real distribution of defect shapes rather than something supplied externally. Applied to GDXray's
2D defects, this directly targets Part 2's honest finding that "SD-inpainting's plain mask
conditioning tells the model what region to fill, not what shape the content inside should take."

**Scoped down from MAISI for 2D:** NVIDIA's models generate full 3D volumes at up to 512×512×256
using latent diffusion (MAISI-v1) or latent rectified flow (MAISI-v2, ~33x faster inference) —
this part is the same core idea at 2D, single-channel-mask scale, which is far cheaper to train
and fits comfortably on local GB10 hardware. Architecture direction to resolve at implementation
time (real open question, not decided here):
- Stack image + binary mask as extra channels through a jointly-trained VAE/UNet (channel
  expansion on top of Part 2's SD1.5 inpainting checkpoint, if that fine-tunes cleanly), vs.
- A smaller custom joint latent diffusion model trained from scratch on the ~338 real
  (image, mask) pairs from Part 2's `build_defect_dataset`, given the dataset is far too small
  for a from-scratch 3D-scale model but may be workable at this scale and resolution.
- Worth a literature check before committing to either (search terms: "joint image and mask
  diffusion," "image-and-label synthesis diffusion," "MedSegDiff") in case an existing 2D
  joint-generation recipe is directly adaptable, rather than assuming a from-scratch design.
- Higher implementation risk than Parts 3-4 (which are straight swap-ins on Part 2's recipe) —
  say so plainly in the post if the joint architecture underperforms or doesn't converge cleanly
  on this little data; that's a legitimate, reportable finding for Part 6/9, not a result to hide.

**Especially visual by nature, lean into it:** unlike a single before/after pair, a joint
generative model naturally produces a *grid* of sampled (image, mask) pairs — show a sampling
grid, not just one result. Same deliverable bar as every post (whiteboard-style ELI5 diagram,
size-optimized GIF, real code, jargon table), but this post is a strong candidate to make the GIF
an actual sampling-diversity loop (cycling through several generated pairs) rather than a
training-progress curve, given how visual the joint-output format already is.

- Compare generated mask shapes against Part 2/3's approaches and against real GDXray defect
  shapes (the same Otsu-refined silhouettes Part 3 uses) — does a jointly-generated mask actually
  look more like a real defect silhouette than a hand-drawn crack or a copied one?
- Feeds into Part 6 as a third candidate technique alongside Part 3's ControlNet conditioning and
  Part 4's Flux base model.

## Part 6 — Combining what works

- Take whichever of Part 3 (real-shape ControlNet conditioning), Part 4 (Flux base model), and
  Part 5 (joint image+mask generation) empirically helped — not assumed, measured — and combine
  them.
- Layer Part 1's calibrated Poisson noise model on top in pixel space, masked-region only, after
  compositing real pixels back into the rest of the image (the Part 2 gap noted above) — closing
  the noise-statistics deficit Part 2 found without asking a VAE-based diffusion model to solve a
  problem it's structurally bad at.
- Real open question going in, stated as such in the post: do the individual improvements from
  Parts 2-5 actually compound when combined, or does combining them hit diminishing returns /
  conflicts (e.g., ControlNet's shape constraint fighting Flux's own generative prior, or joint
  generation's learned shapes fighting an externally-supplied ControlNet silhouette)? Report
  whatever's actually found.

## Part 7 — GAN-based synthesis

- Small GAN (DFMGAN-style: defect-aware generator conditioned on a mask + defect-free base)
  trained on the same GDXray defect crops.
- A distinct third generative family, independent of the diffusion track in Parts 2-6.
- Honest-limitation angle expected: GANs are notoriously unstable on small datasets — likely
  mode collapse or training instability with so few real examples. That's a legitimate, useful
  finding for Part 9, not a post to avoid because it might not "win."

## Part 8 — Cut-paste + Poisson blending baseline

- Formalizes the manual workflow Hasan already does at work: crop a real defective region,
  Poisson-blend it into a good image.
- Cheapest method computationally and conceptually. Sets the floor the other methods need to beat
  in Part 9.

## Part 9 — Post-training a real detector on synthetic data (the payoff)

Reframed 2026-08-28: this is not a quick eval tacked onto the comparison — **post-training a real
segmenter/detector is the actual payoff of the whole series**, and gets the same rigor this blog
already gives an SFT/RL post-training stage (see the RAU post's SFT + GRPO stages, or HPMA's
adapter training) rather than a one-shot "does it help, y/n" fine-tune.

- Real baseline: a pretrained segmenter/detector (reuse `local-ai-defect-inspection`'s model
  choice/eval machinery where it fits, e.g. Parts 3-4's fine-tuning setup) evaluated on real
  GDXray data only, before any synthetic augmentation — the number every other condition below
  has to beat.
- Post-train that same baseline separately on real+synthetic splits from each method (physics,
  diffusion, ControlNet, Flux, joint generation, the Part 6 combined hybrid, GAN, cut-paste) —
  same training recipe/hyperparameters across conditions, only the data differs, so differences
  in the result
  are attributable to the synthetic data's quality, not incidental training differences.
- Worth ablating, not just a single real:synthetic ratio: does more synthetic data monotonically
  help, plateau, or hurt (a real, reportable finding either way) for each method.
- Report which method's synthetic data actually improves the post-trained model, broken out by
  the two original goals: rare-defect augmentation vs. minimizing labeling effort
  (mask-conditioned generation methods get "free" ground-truth masks; cut-paste does not, since it
  needs a source mask to begin with).
- Empirical result, not a literature summary — real training runs, real before/after metrics on a
  real held-out eval set, same "real code, real captured output" bar as every other post in this
  series.

## Deliverable bar (all posts)

Matches the established "Paper of the Week" deep-dive pattern:
- Real code, no pseudocode.
- Real captured output and real before/after results.
- Size-optimized GIF where there's an iterative/training process to show (check size before
  publishing — quarto.pub rejects uploads much above ~1MB; shrink with `Image.quantize`/frame
  subsampling as needed).
- A whiteboard-style hand-drawn diagram per post for the ELI5 section — bold marker-font title
  with an arrow annotation, rounded hand-drawn-border boxes, connecting arrows/dashed lines,
  casual sans-serif labels (reference: LangChain "Build Your Own Agent Harness" video whiteboard
  style, screenshot provided during design).
- Same two-reading-level structure (ELI5 + data-scientist) and jargon-buster glossary table as
  the RAU/HPMA posts.

## Open risks / follow-ups (not blockers)

- CXR-AD dataset access — revisit if it becomes publicly downloadable; would be a closer domain
  match than GDXray.
- ~~Whether `Boogu-Image` exposes true mask-conditioned generation or only instruction-text
  editing~~ — checked: instruction-text editing only, no mask parameter. Resolved; Part 2 used
  SD1.5 inpainting instead.
- GDXray domain mismatch (castings/welds, not semiconductor) — accepted risk, consistent with
  `local-ai-defect-inspection`'s existing precedent.
- ~~Flux licensing (Part 4)~~ — resolved 2026-08-29: confirmed gated via the HF API, existing
  `HF_TOKEN` already had access, used the real Flux-Fill-dev checkpoint. See Part 4 entry above.
  If gated, decide explicitly (sign-off to use anyway vs. non-gated substitute) rather than
  discovering it mid-implementation.
- **NVIDIA Cosmos was considered and set aside** for this series — it's a world/video-simulation
  model family for physical-AI/robotics, not built around 2D masked image inpainting; no clear
  fit for this task's shape was found without a lot more research, so it isn't in the Part 3/4/6
  lineup above. Revisit only if a concrete Cosmos-based 2D-inpainting workflow surfaces later.
  (NVIDIA's *MAISI* line of work, by contrast, was a direct fit — see Part 5.)
- **Part 2's missing mask-compositing step** carries forward as a prerequisite for Part 6's
  noise-matching step specifically (see Part 2 entry above) — fix it once, reuse in both.
- **Part 5's joint image+mask architecture is genuinely unresolved** (channel-expansion fine-tune
  of Part 2's checkpoint vs. a from-scratch small joint model vs. adapting an existing published
  recipe) — higher implementation risk than the straight swap-ins in Parts 3-4, flagged in Part
  5's own section above. Decide the concrete architecture at implementation time, not here.
