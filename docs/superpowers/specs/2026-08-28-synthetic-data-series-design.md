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
- Not building a production defect-detection pipeline. The measured comparison in Part 5 uses
  a downstream task as an *evaluation signal* for the synthetic-data methods, not as a
  deliverable in its own right.

## Series placement and structure

New standalone series: `posts/series/synthetic-data/`. Considered folding into
`local-ai-defect-inspection` (same domain, already using GDXray) but rejected — that series'
framing (VLM eval, LinkedIn-first) doesn't fit a generative/physics-methods series. Cross-link
both directions instead.

Five posts, in this order:

1. **Physics-based X-ray simulation** — Beer-Lambert forward model, calibrated from real GDXray
   images since no equipment access exists.
2. **Mask-conditioned diffusion fine-tuning** — LoRA fine-tune on real GDXray defect crops+masks.
3. **GAN-based synthesis** — small GAN (DFMGAN-style) trained on the same crops.
4. **Cut-paste + Poisson blending baseline** — the classical manual-workflow approach, formalized.
5. **Compare everything** — run all four methods' synthetic output through the same downstream
   task, report which helps for which of the two goals (rare-defect augmentation vs. minimizing
   labeling effort).

Each post gets a `series:` frontmatter block and opens with a callout linking back to
`index.qmd`, per the repo's established series pattern (`CLAUDE.md`). Wiring into
`_quarto.yml` navbar/sidebar and `README.md`'s series list happens as part of implementation,
not this design doc.

## Data

**GDXray** (castings + welds groups) is the real-data anchor for all five posts — publicly
downloadable, free for research/education use, already integrated in this repo
(`local-ai-defect-inspection` series uses it), and has pixel-level defect masks for some series.
No semiconductor-specific category exists in GDXray, but casting voids and weld-porosity defects
share the same underlying X-ray physics (density gap → attenuation drop → intensity signature)
as the semiconductor voids/cracks this series is ultimately in service of. Domain-mismatch
tradeoff is accepted the same way `local-ai-defect-inspection` already accepted it.

## Tooling

New `uv`-managed subproject: `notebooks/synthetic-data-xray/`, following the
`paper-rau-sam2`/`paper-hpma-sam3` pattern — scoped `pyproject.toml`, `.venv` gitignored,
`uv sync` to set up. Shared across all five posts (common data-loading/GDXray utilities), with
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
provides. This becomes the baseline the later parts are compared against in Part 5.

## Part 2 — Mask-conditioned diffusion fine-tuning

- Same GDXray defect crops + masks used to calibrate Part 1.
- LoRA fine-tune a mask-conditioned inpainting/diffusion model (Stable Diffusion inpainting +
  ControlNet-seg, or the `Boogu-Image` checkpoint already explored in
  `notebooks/_drafts/boogu-image-semiconductor-defect-editing.ipynb` if it exposes mask
  conditioning — check first) so the model learns GDXray's actual X-ray texture instead of a
  natural-image prior.
- Compare output against Part 1's physics-rendered images on the same masks.
- Deliverable includes a GIF (training-progress or physics-vs-diffusion-vs-real comparison).

## Part 3 — GAN-based synthesis

- Small GAN (DFMGAN-style: defect-aware generator conditioned on a mask + defect-free base)
  trained on the same GDXray defect crops.
- Third distinct generative family alongside physics (Part 1) and diffusion (Part 2).
- Honest-limitation angle expected: GANs are notoriously unstable on small datasets — likely
  mode collapse or training instability with so few real examples. That's a legitimate, useful
  finding for Part 5, not a post to avoid because it might not "win."

## Part 4 — Cut-paste + Poisson blending baseline

- Formalizes the manual workflow Hasan already does at work: crop a real defective region,
  Poisson-blend it into a good image.
- Cheapest method computationally and conceptually. Sets the floor the other three methods need
  to beat in Part 5.

## Part 5 — Compare everything

- Run all four methods' synthetic images through the same downstream task (fine-tune a
  segmenter/detector on real+synthetic vs. real-only splits), reusing eval machinery from
  `local-ai-defect-inspection` Parts 3-4 where it fits.
- Report which method actually helps, broken out by the two original goals: rare-defect
  augmentation vs. minimizing labeling effort (mask-conditioned generation methods get "free"
  ground-truth masks; cut-paste does not, since it needs a source mask to begin with).
- Empirical result, not a literature summary — the payoff post.

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
- Whether `Boogu-Image` exposes true mask-conditioned generation or only instruction-text
  editing — needs checking before committing Part 2's model choice.
- GDXray domain mismatch (castings/welds, not semiconductor) — accepted risk, consistent with
  `local-ai-defect-inspection`'s existing precedent.
