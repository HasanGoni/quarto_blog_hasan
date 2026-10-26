# Robust segmentation evaluation and public datasets for an X-ray / industrial defect blog series

Research date: 2026-10-17. About 14 tool calls. Most facts come from web-search summaries plus three direct page fetches (GDXray page, Zenodo record). Anything not opened in full text is marked "snippet-level".

## Q1. Public X-ray / industrial datasets with pixel masks: licences, sizes, label problems

### Takeaway
No clean, openly licensed, radiographic defect dataset with pixel masks was verified. GDXray is the canonical set but is research/education-only, no redistribution, and its annotations are mostly boxes and labels, so showing its images in a public blog is a licence risk. The pragmatic route is synthetic data (own simulation) plus a few permissively licensed sets, with licences checked at source before publishing.

### Cited Findings
- GDXray / GDXray+ licence (custom, not an SPDX licence): images "can be used free of charge for research and educational purposes only"; "Redistribution and commercial use is prohibited"; users must cite the paper — [Mery GDXray page (fetched)](https://domingomery.ing.uc.cl/material/gdxray). Dataset Ninja also lists the licence as "custom" — [Dataset Ninja](https://datasetninja.com/gdxray).
- GDXray size: the official page says "more than 21.100" images; the 2015 paper describes 19,407; a search summary cites 20,966 for GDXray+, ~3.5 GB; five groups (castings, welds, baggage, natural objects, settings). The totals conflict by source — [Mery page](https://domingomery.ing.uc.cl/material/gdxray); [Dataset Ninja](https://datasetninja.com/gdxray); [paper PDF](https://diicc.uda.cl/profesores/vriffo/Publications\Journals\PDF\GDXray2015.pdf).
- GDXray annotations: mostly bounding-box coordinates or image labels in text files, not pixel masks (search summary of the dataset description) — [Dataset Ninja](https://datasetninja.com/gdxray). The official page fetched did not describe annotation types.
- GDXray weld series W0001 and W0003 were contributed by BAM Berlin, so extra terms may apply (not confirmed) — [GDXray summary](https://domingomery.ing.uc.cl/?p=161).
- GDXray-Welds is small: reportedly 68 raw images, needing manual cropping/annotation for patches — [Codesota summary, secondary](https://www.codesota.com/browse/industrial-inspection/weld-inspection/gdxray-welds).
- Series C0001 contains 72 images of an aluminium wheel casting — [Mery page](https://domingomery.ing.uc.cl/material/gdxray).
- Zenodo "X-ray weld seam image" (DOI 10.5281/zenodo.10618962): licence Apache License 2.0, 475.3 MB single zip, published 2024-02-05. The page does not state image count, annotation type or resolution — [Zenodo (fetched)](https://zenodo.org/records/10618962). Whether it has masks is unknown.
- WDXI weld dataset: reported as 13,766 16-bit TIF images, 7 defect types, severe class imbalance. This is from a search snippet only; no download page or licence was found, and it appears to be a detection (Faster R-CNN) dataset, not masks — [MDPI Appl. Sci. 15(22):12033](https://www.mdpi.com/2076-3417/15/22/12033/html); [BUPT paper](https://html.rhhz.net/buptjournal/html/20190603.htm).
- RIWD (radiographic weld semantic segmentation dataset) is described in a paper; public release and licence not confirmed — [search result, IEEE Access via DOAJ](https://doaj.org/article/ce43a9b33c8c4e8b9c822be1eb389d1c).
- Steel pipe weld X-ray set: 3,408 images, 8 defect types, Labelme annotations (YOLO/VOC formats) — [arXiv 2104.14907](https://arxiv.org/pdf/2104.14907). Licence not checked.
- Solder voids / BGA / PCB X-ray: no clearly public dataset with masks found. Papers exist (PCB-DeepLabV3 void segmentation, Sci. Rep. 2024; U-Net vs Mask R-CNN on voids, IEEE 2021) but data availability not verified — [PMC11126598](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11126598/); [IEEE 9551671](https://ieeexplore.ieee.org/document/9551671).
- MVTec AD (optical, with pixel masks): licence CC BY-NC-SA 4.0, non-commercial — [MVTec official page](https://www.mvtec.com/company/research/datasets/mvtec-ad). NC plus share-alike makes blog use arguable; read terms and credit properly.
- KolektorSDD2 (optical surface defects with masks): CC BY-SA 4.0 according to Dataset Ninja only (third party); unverified at source — [Dataset Ninja](https://cdn.datasetninja.com/kolektor-surface-defect-dataset-2).
- Severstal steel defects (Kaggle, 12,568 images, 4 classes, masks via RLE): no licence found; check the Kaggle competition rules — [review of benchmarks](https://arxiv.org/pdf/2305.13261) (snippet-level).
- Roboflow Universe "Defect segmentation" (weld): CC BY 4.0, but not shown to be radiographic — [Roboflow](https://universe.roboflow.com/weld-moiza/defect-segmentation-a92vg).
- Medical X-ray fallback: Montgomery (138 CXRs, 80 normal/58 TB, manual lung masks), Shenzhen (662 CXRs, 326 normal/336 TB; lung masks available separately), JSRT/SCR (247 images, 1024x1024, lungs/heart/clavicle contours by van Ginneken et al.) — [Res-CR-Net](https://arxiv.org/pdf/2011.08655); [QIMS paper](https://qims.amegroups.org/article/view/5132/html); [Dataset page](https://mia4lung.uib.es/datasets_segmentation.html). The CSIRO combined JSRT+Montgomery kit is CC BY 4.0, but that does not cover the originals — [CSIRO kit](https://researchdata.edu.au/lung-segmentation-data-kit/1376776). Official NLM and SCR licence texts were not found.

### Inferences
- A defensible plan: synthetic voids on real or simulated background, with the generator code released, so the series carries no third-party image licence risk. Use GDXray only as a "locally downloaded, not redistributed" demo; do not embed its images without asking the authors.
- The Zenodo Apache-2.0 weld set is the only radiographic candidate with a permissive licence, but its labels are undocumented. Inspect it before committing.
- Thin/tiny defect behaviour (voids) is better shown with synthetic ground truth, where masks are exact.

### Gaps
- No verified radiographic dataset with pixel masks AND a permissive licence. Not found: public void/lifted-chip X-ray masks.
- Licence of WDXI, RIWD, Severstal, and original Montgomery/Shenzhen/JSRT unverified. Official class imbalance numbers, resolutions and splits for most sets were not verified.
- Known label problems specific to these datasets were not found (only the generic ones in Q4).

## Q2. Published practice for robustness evaluation

### Takeaway
There is an established recipe: ImageNet-C style corruptions at 5 severities (Cityscapes-C / ADE20K-C / VOC-C), reporting mIoU per corruption and retention ratio, and group-DRO-style worst-group reporting. All are cheap to implement (noise, blur, contrast, brightness) and map directly to X-ray (Poisson noise, detector blur, contrast/gain drift).

### Cited Findings
- Kamann and Rother (CVPR 2020; IJCV 2021): about 400,000 corrupted images from Cityscapes, PASCAL VOC 2012, ADE20K, using DeepLabv3+ variants. Robustness generally rises with clean performance; atrous convolutions and long-range links help; models handle noise/blur better than digital and weather corruptions; the Dense Prediction Cell hurt corrupted performance — [arXiv 1908.05005](https://arxiv.org/pdf/1908.05005); [CVPR page](https://openaccess.thecvf.com/content_CVPR_2020/html/Kamann_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models_CVPR_2020_paper.html).
- Cityscapes-C / ADE20K-C protocol: 16 corruptions, 5 severities, grouped blur/noise/weather/digital; mIoU averaged over severities (noise: first 3 levels only); a retention rate = corrupted mIoU / clean mIoU is reported — [arXiv 2412.01941](https://arxiv.org/pdf/2412.01941) (snippet-level).
- Worst-group reporting: Group DRO (Sagawa et al., ICLR 2020) shows models with good average accuracy fail on atypical groups; stronger regularisation gave 10-40 point worst-group gains on NLI and two image tasks — [arXiv 1911.08731](https://arxiv.org/abs/1911.08731v2). That is classification; I found no segmentation-specific "worst-slice Dice" paper in this search.
- Calibration: standard ECE is dominated by the first/last bins and background pixels, so it hides rare, important errors; Average Calibration Error (ACE) weights bins equally — [arXiv 2506.03942](https://arxiv.org/pdf/2506.03942). Pixel-wise ECE and Signed Distance Calibration — [arXiv 2503.05107](https://arxiv.org/abs/2503.05107v2). Boundary-aware ECE (BA-ECE) bins by distance to the GT boundary — [arXiv 2506.16589](https://web3.arxiv.org/pdf/2506.16589). Temperature scaling reportedly cut ECE up to 56% without changing decisions (a thesis, secondary) — [Politecnico di Milano](https://www.politesi.polimi.it/handle/10589/253581). All snippet-level.
- Metrics Reloaded: choice of metric should follow a "problem fingerprint"; DSC and similar overlap metrics are dominated by large structures; distance metrics (HD95, NSD, boundary IoU) depend on boundary extraction and tolerance; implementations differ across tools — [arXiv 2206.01653](https://arxiv.org/pdf/2206.01653); [pitfalls arXiv 2302.01790](https://arxiv.org/pdf/2302.01790); [arXiv 2410.02630](https://arxiv.org/html/2410.02630v2).

### Inferences
- Blog-sized implementation: (1) corruption suite of ~6 X-ray-plausible perturbations x 5 severities, report mean and worst corruption plus retention; (2) worst-group IoU over metadata slices (thickness, noise level, defect size bin); (3) ECE and a boundary-band ECE on defect-vs-background; (4) multiple seeds.
- Plain ECE on defect-sparse images will look excellent because of background pixels; this is a good teaching point.

### Gaps
- Not found: confidence-interval practice for segmentation (bootstrap over images vs pixels), seed-variance literature. I recommend (inference, not cited) bootstrapping at image level because pixels are correlated; verify before stating as published practice.
- Cityscapes-C/ADE20K-C details are from a secondary paper, not the original Hendrycks-style code. Corruption suites for X-ray specifically: not found.

## Q3. Tiny / thin defects where mean IoU hides failure

### Takeaway
Literature supports size-stratified and object-level reporting alongside pixel metrics; thin structures make IoU brutally sensitive to 1-pixel boundary shifts.

### Cited Findings
- Instance-aware mIoU (IA-mIoU) evaluated per size bin: mIoU barely moved as small instances were removed while IA-mIoU fell steadily (PASCAL VOC) — [arXiv 2309.14117](https://arxiv.org/pdf/2309.14117).
- SegmentMeIfYouCan: a ground-truth component counts as detected if at least one pixel is hit; false-negative ratios reported per size interval; small components overlooked far more often — [arXiv 2104.14812](https://arxiv.org/pdf/2104.14812).
- Thin defects: on one crack dataset pixel recall 0.9911 but IoU 0.1533; fragmented crack counts as one missed instance plus fragments as false positives, so instance metrics need connectivity handling — [arXiv 2609.00469](https://arxiv.org/pdf/2609.00469) (single recent paper, snippet-level).
- Boundary IoU (CVPR 2021): IoU restricted to a band at the mask edge; more sensitive to boundary errors on large objects, not over-penalising small ones; also Boundary AP/PQ — [arXiv 2103.16562](https://arxiv.org/abs/2103.16562).
- Metrics Reloaded lists small-structure neglect as a named pitfall — [arXiv 2302.01790](https://arxiv.org/pdf/2302.01790).

### Inferences
- Report: defect-level recall and precision (connected components, hit if overlap >= some IoU or >= 1 pixel, state which), recall by size bin, plus Dice/IoU on the defect class only, not mean over background.
- Always state the pixel tolerance for boundary metrics (NSD threshold) because results depend on it.

### Gaps
- No void-specific (semiconductor) metric paper found. clDice for thin structures is mentioned in Metrics Reloaded but I did not open its primary paper.

## Q4. Label errors in segmentation and tools (grounding a "label-error agent + human review")

### Takeaway
Cleanlab-style confident learning has been extended to segmentation (Lad et al., arXiv 2307.05080) and ships in the cleanlab library; the output is a ranked list per image for human review, which fits the agent-plus-human design.

### Cited Findings
- "Estimating label quality and errors in semantic segmentation data via any model" extends confident learning to pixels; handles missing masks, wrong class, imprecise boundaries; any model's probabilities can be used — [arXiv 2307.05080](https://arxiv.org/pdf/2307.05080).
- Library usage: needs (N,H,W) labels plus out-of-sample pixel probabilities (cross-validation); `find_label_issues()` returns per-pixel boolean masks; images ranked by a self-confidence score; `downsample` trades speed for accuracy; import path differs between blog and docs (`cleanlab.semantic_segmentation` vs `cleanlab.segmentation`) so check the installed version — [cleanlab docs v2.7.1](https://docs.cleanlab.ai/v2.7.1/tutorials/segmentation.html); [cleanlab blog](https://cleanlab.ai/blog/segmentation-errors).

### Inferences
- Cross-validated predictions are needed, so on a small X-ray set this means k-fold training. In-sample probabilities will hide errors. Human review remains the decision step; thin defects have ambiguous boundaries, so flagged boundary disagreement is not necessarily an error.
- Synthetic data allows injecting known label noise (shifted, dropped, eroded masks) to measure the agent's precision/recall — a strong blog experiment (my suggestion).

### Gaps
- Not found: quantitative label-error rates for the datasets in Q1, or an inter-annotator study on X-ray defect masks. Cleanlab's reported detection performance numbers were not retrieved. Not verified: whether the library is maintained in 2026 beyond v2.7.1.

## Q5. Synthetic X-ray data and domain gap

### Takeaway
Simulation is used in published defect work, but sim-only training generally fails on real images; noise/artefact realism, mixing, or a small real set is needed.

### Cited Findings
- 2024 arXiv study on aluminium wheel X-ray: sim-only training is frequently inadequate; sim-to-real domain adaptation with a small labelled real set was more cost-efficient than a fully supervised oracle — [arXiv 2406.19175](https://arxiv.org/html/2406.19175v1).
- Fraunhofer EZRT (light-metal rims): an AI trained only on simulated data performs well on synthetic tests but fails on real data; a mix is a reasonable option — [Fraunhofer SCS](https://www.scs.fraunhofer.de/en/focus-projects/aiSAR.html).
- Sim-ADR: simulation plus noise extracted from experiments, reported over 87% detection accuracy on a 416-image test set — [search result, Springer](https://repository.ias.ac.in/140944) (snippet-level).
- CT of cast aluminium: claimed first defect detection in 3D CT trained solely on simulation, DOI 10.1016/j.neucom.2021.04.094 — [Heidelberg record](https://heibib.ub.uni-heidelberg.de/search/Record/1762909502).
- A GDXray-related result: U-Net pretrained on synthetic Radon projections reaching mIoU 0.85 on real images — abstract only; treat as unverified, I did not locate the source URL.

### Inferences
- For the blog: use synthetic data for controlled, fully-labelled experiments (known size, contrast, noise), and say explicitly that conclusions on absolute performance do not transfer to real tools. Add a sim-vs-real gap probe (evaluate on a small real public image set) as honesty check.

### Gaps
- Physics-simulation tool choices (e.g., open-source projectors) not researched. No semiconductor-package-specific synthetic void dataset found.
