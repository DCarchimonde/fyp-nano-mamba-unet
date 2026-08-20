# P2 Final Audit — Nano-Mamba U-Net

**Audit date:** 2026-08-20

**Independent adversarial re-review:** 2026-08-21

**Submission deadline:** 2026-08-23

**Audit branch:** `p2-final-audit-2026-08-20`

**Baseline commit:** `8a8189718ffd8e4a9920c3b6e4ed1f9cf54c70b1`

**Fixed title:** *Nano-Mamba U-Net: A Spatio-Temporal Framework for Cardiac Motion Analysis*

## 1. Submission decision at audit baseline

**Status: BLOCKED pending P0 closure.**

The central quantitative table is internally consistent and the patient split is reproducible, disjoint, and patient-level. However, the current repository and submitted evidence do not yet provide a defensible end-to-end provenance chain from code and checkpoint to every thesis claim and figure. In particular, the qualitative figure is not valid validation evidence, two training-curve claims cannot be checked from the submitted bundle, one cited record is false, portable thesis compilation is broken, multiple document versions conflict, and raw administrative files expose personal data in the public repository.

The deadline strategy is therefore:

1. close or neutralize every P0 item without changing the reported main results;
2. fix P1 items that do not require full retraining;
3. record residual limitations explicitly instead of over-claiming;
4. build a clean thesis PDF from the audited branch and archive a machine-checkable evidence manifest.

### Final disposition after remediation

**Status: CONDITIONALLY READY FOR PRIVATE SUBMISSION.**

The audited branch now has no open code, thesis, result-table, figure, or
presentation P0 defect. Unsupported qualitative evidence remains removed; the
learning curve has been restored only after recovery of six byte-hashed epoch
logs. The bibliography, scientific terminology, canonical PDF, quantitative
figures, privacy posture, evidence validator, reproducibility guide, and defence
materials were corrected and verified. The aggregate-consistency audit plus
recovered case/log/checkpoint checks pass; this establishes a strong cross-file
chain, not complete historical run attestation.

The qualification is deliberate but narrower after recovery of
`p2_closure_20260821_020939.zip`. All six original-format per-case tables, all
six 150-epoch logs, and metadata plus unique hashes for all six checkpoints now
pass cross-file validation. Strict evidence closure remains incomplete only
because the historical checkpoint-set identity, environment identity, exact
command, and full 200-case dataset snapshot are unconfirmed. Before portal
upload, the candidate must also create a private, ignored metadata file, fill
the institution-required declaration fields, sign through the accepted
workflow, rebuild, and visually inspect that private PDF.

## 2. Non-negotiable scientific boundaries

- The implemented task is **four-class 3D cardiac MRI segmentation** (background, RV, MYO, LV) on labelled ED/ES volumes.
- The tensor depth axis is anatomical slice depth, not time. The current experiment is not 4D cine tracking, temporal registration, motion estimation, or ejection-fraction estimation.
- The implemented bottleneck is a **Mamba-inspired gated sequence module**. It is not the full Mamba selective state-space algorithm and does not implement selective scan.
- The main comparison evidence must come only from `src/21_rigorous_experiment_pipeline.py` and the associated `summary_metrics.csv`.
- SegResNet16 has the highest validation mean Dice. The No-Mamba ablation also exceeds Nano-Mamba U-Net. The defensible contribution is the compact accuracy–efficiency trade-off, not best-in-class Dice or proof that the gated module causes an accuracy gain.
- The held-out set is a validation set used for checkpoint selection. It is not the official ACDC test set or an independent final test set.

## 3. Evidence inventory and verified facts

### 3.1 Audited inputs

| Evidence | Audit observation |
|---|---|
| Git baseline | Commit `8a8189718ffd8e4a9920c3b6e4ed1f9cf54c70b1` on `main`; audited on a separate branch |
| Uploaded final-audit bundle | 12 files; includes four figures, thesis build artefacts, split JSON, discovery JSON, and summary CSV/JSON; omits per-case CSVs, training logs, checkpoints, environment capture, and full data manifest |
| Uploaded thesis PDF | 69 pages; SHA-256 `50e731d4bcb59014fe0bc5699026224e5dd15c22e67976748d864fe04fe1a9d0`; byte-identical to the bundle PDF |
| Repository thesis PDF | 75 pages; SHA-256 `0a868b52745db3b70a173bdcd7b5489c66054452b57909e3c0ee8b13c39df681`; differs from the uploaded/bundle PDF |
| Main result script | `src/21_rigorous_experiment_pipeline.py` |
| Result table | Uploaded `summary_metrics.csv`; all row means equal the arithmetic mean of RV, MYO, and LV Dice; FPS and latency are reciprocal within rounding |
| Patient split | Seed 42; 80 train patients and 20 validation patients; no overlap; all patient IDs 001–100 assigned; reproduced exactly with `random.Random(42)` |
| Discovery report | Reports 100 patient directories, 200 cases, and zero skipped containers, but records only five example cases rather than a complete manifest |
| Reconstructed source lineage | Git contains the rigorous pipeline and bottleneck source before the ZIP summary timestamp; hashes and commits are recorded in `historical_source_lineage.json`. This supports a plausible timeline but does not prove which command/checkpoints generated the aggregate files. |
| Recovered closure ZIP | `p2_closure_20260821_020939.zip`, SHA-256 `16fbbe31db97e9695fc78aadbb46e9cf04f99fa1e1fd2d9e735f03d75d14a3b3`; all 19 ZIP checksum entries pass. It supplies six per-case tables, six epoch logs, six checkpoint records, a current environment capture, and an explicitly unconfirmed transcript. |
| Per-case validation | Six files × 40 cases; identical 20-patient/two-case coverage; every class mean and aggregate exactly reproduces the main table. |
| Epoch logs | Six files × 150 contiguous epochs; every best row matches the summary and checkpoint metadata. |
| Checkpoint records | Six readable, internally consistent records with unique full-file SHA-256 hashes; historical directory identity is not yet user-confirmed. |

### 3.2 Verified main results

| Model | Best epoch | RV Dice | MYO Dice | LV Dice | Mean Dice | Parameters | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNet3D | 105 | 78.35% | 74.57% | 89.59% | 80.83% | 4.809 M | 88.17 |
| Nano-Mamba U-Net | 121 | 82.11% | 80.35% | 91.88% | 84.78% | 1.456 M | 29.09 |
| No-Mamba U-Net | 134 | 83.26% | 81.30% | 92.37% | 85.64% | 2.288 M | 28.80 |
| Half-Mamba U-Net | 149 | 82.41% | 80.46% | 91.99% | 84.95% | 1.638 M | 29.04 |
| Attention U-Net | 62 | 69.43% | 72.38% | 82.53% | 74.78% | 5.909 M | 22.41 |
| SegResNet16 | 112 | 84.30% | 82.76% | 93.03% | 86.70% | 4.701 M | 25.50 |

Exact comparisons relevant to the thesis:

- Nano-Mamba U-Net exceeds UNet3D by **3.945 percentage points** in validation mean Dice and uses **69.714% fewer** reported parameters.
- SegResNet16 exceeds Nano-Mamba U-Net by **1.918 percentage points**; Nano-Mamba uses **69.021% fewer** reported parameters.
- No-Mamba exceeds Nano-Mamba by **0.862 percentage points** while using **57.076% more** reported parameters.
- Half-Mamba exceeds Nano-Mamba by **0.174 percentage points**. The paired post-hoc interval for Nano minus Half-Mamba is **[-0.909, +0.551] percentage points**, so this small ordering is unresolved.

Recovered post-hoc patient-level paired intervals (10,000 percentile-bootstrap
replicates, seed 20260820) are **[+2.829, +5.009]** percentage points for Nano
minus UNet3D, **[+3.508, +18.294]** for Nano minus Attention U-Net,
**[-3.010, -0.884]** for Nano minus SegResNet16, and **[-1.482, -0.230]** for
Nano minus No-Mamba. They are descriptive intervals on the validation patients
used for checkpoint selection, not pre-registered significance tests.

All values above are rounded only for presentation; the audited CSV remains the sole numerical source.

## 4. P0 — submission-blocking issues

The detailed rows below preserve the **baseline** status at issue discovery.
Final status after remediation is consolidated in Section 10. “Retraining”
distinguishes changes to already-trained weights from post-hoc re-evaluation or
local evidence collection.

| ID | Location | Problem, reason, and likely impact | Evidence | Required correction | Retraining? | Estimate | Status |
|---|---|---|---|---|---|---:|---|
| P0-01 | `src/16_thesis_visualization.py:42-48, 320`; `sample-chap-results-p2.tex:140-144`; `figures/qualitative_result.png` | The script requests the wrong rigorous checkpoint name, can fall back to a legacy checkpoint, and defaults to `patient005`, which is in the training split. The thesis caption calls the output a validation slice. This is false provenance and invalid qualitative validation evidence. | The rigorous pipeline saves `best_NanoMambaUNet.pth`; the visualizer requests `NanoMambaUNet_best.pth`. The audited split assigns `patient005` to training. | Make the visualizer fail closed unless it receives the rigorous checkpoint and split; reject non-validation cases; emit a provenance manifest. Remove the current figure/caption until regenerated from a validation case. | No training; local inference required to restore the figure | 1.5–2.5 h plus inference | Open |
| P0-02 | `sample-chap-results-p2.tex:103-107` | The thesis gives exact learning-curve values (3.97% at epoch 1 and 78.89% at epoch 30), but the uploaded evidence omits `training_log_NanoMambaUNet.csv`. Exact claims without primary evidence are not auditable. | Bundle inventory contains no `training_log_*.csv`; summary data only verify best epoch and final Dice. | Remove exact intermediate-epoch claims for the submission build unless the original CSV is supplied and hashed. If supplied, generate the curve directly from the CSV. | No | 20–60 min | Open |
| P0-03 | `myrefs.bib:60`; citations in `sample-chap-intro.tex:27` and `sample-chap-litreview.tex:37` | `peng2020review` is a false bibliographic record. Article number 100048 in *Patterns* belongs to a different article. A fabricated/misattributed source is an academic-integrity risk. | DOI record `10.1016/j.patter.2020.100048` identifies Sarah Callaghan's article, not the claimed medical-segmentation review. | Replace the two citations with a genuine, claim-matched primary review (for example, Chen et al., *Deep Learning for Cardiac Image Segmentation: A Review*, DOI `10.3389/fcvm.2020.00025`) and remove the false entry. Verify every cited entry against a primary publisher, DOI, PubMed, or official proceedings record. | No | 2–4 h for all cited entries | Open |
| P0-04 | Repository root and `paper_write/Universiti_Malaya_Thesis_Template/` | There are conflicting thesis states: the repository PDF is 75 pages, while the uploaded and bundle PDF is 69 pages and has a different hash. The bundle is also not tied to a Git commit. An examiner cannot establish which source produced the submitted PDF. | SHA-256 values in Section 3.1; bundle timestamps post-date the repository PDF. | Select the audited LaTeX source as canonical, build in a clean directory, publish one final PDF, and create a manifest containing commit SHA, source/result hashes, build command, tool versions, page count, and PDF hash. | No | 1–2 h after source fixes | Open |
| P0-05 | `sample-chap-results-p2.tex:60,71,98,142`; repository `figures/` | Four figures use absolute `D:/AI_FYP/...` paths, and three quantitative figures referenced by LaTeX are absent from the repository. A clean build from the audited commit cannot succeed. | Source inspection and repository inventory. | Add only verified, reproducibly generated quantitative figures; use paths relative to the LaTeX project; remove the invalid qualitative figure until it is regenerated. Add a clean-build check. | No | 1–2 h | Open |
| P0-06 | `school_requirement/` and Git history | Raw administrative PDFs in this public repository contain personal and academic data unnecessary for software or thesis reproducibility. This is a privacy and repository-hygiene failure. | Repository visibility and document inspection. Exact personal values are intentionally not repeated in this audit. | Remove raw sensitive files from the current branch; retain a redacted requirement summary only. Warn that a normal deletion does not purge previous commits; history rewriting and cache/search-engine cleanup require a separate coordinated operation. | No | 30–60 min for branch; separate history-remediation task | Open |
| P0-07 | `src/21_rigorous_experiment_pipeline.py:383-400`; missing `per_case_*.csv` | If a class is absent from both prediction and ground truth, the implementation assigns Dice = 1.0. That convention can inflate a case/class mean. The omitted per-case outputs prevent checking whether this occurred. | Metric implementation and bundle inventory. | Supply original per-case CSVs and run an absent-class/label audit. Report the convention explicitly. If any affected cases exist, recompute alternative aggregates from saved predictions/checkpoints and disclose both; do not silently replace the historical table. | No training; checkpoint inference only if original rows/predictions are unavailable | 1–3 h | Open |
| P0-08 | Entire evidence bundle; `src/21_rigorous_experiment_pipeline.py:478-545` | The script records per-case CSVs, training logs, and checkpoint paths, but the submitted bundle omits those files and all software/hardware provenance. The summary table alone cannot verify confidence intervals, checkpoint identity, curve claims, or environment-dependent speed. | Bundle contains 12 files but no per-case CSV, training log, checkpoint metadata extract, or environment report. | Provide one closure bundle containing original per-case CSVs, training logs, checkpoint metadata/hashes (weights may remain off-Git), software versions, GPU/CPU details, and command transcript. Add a validator that fails on missing evidence. | No; evidence collection only | 1–2 h user-side plus audit | Open |

## 5. P1 — high-priority quality and reproducibility issues

| ID | Location | Problem, reason, and likely impact | Evidence | Required correction | Retraining? | Estimate | Status |
|---|---|---|---|---|---|---:|---|
| P1-01 | `src/21_rigorous_experiment_pipeline.py:486-525`; thesis Methods/Limitations | The same 20-patient validation set selects checkpoints and supplies the final reported table. It is held out from gradient updates, but it is not an independent test estimate and is subject to model-selection optimism. | Training loop selects maximum validation Dice and then reloads that checkpoint for the reported validation metrics. | Use “held-out validation,” never “test.” Add this limitation and reserve official test/external validation for future work. | No | 20 min | Open |
| P1-02 | Main experiment design and Results | Only one deterministic split and one seed are reported. Without per-case rows, there are no confidence intervals, paired comparisons, or variability analysis. Small differences, especially Half-Mamba vs Nano-Mamba, cannot support superiority claims. | Split JSON and summary CSV; per-case rows absent. | When rows are supplied, report patient-level bootstrap 95% CIs and paired differences. Treat these as post-hoc descriptive analyses; avoid p-value fishing. Multiple-seed retraining is recommended future work, not a deadline blocker. | No for CIs; yes for multi-seed robustness | 2–4 h post-hoc; days for retraining | Open |
| P1-03 | `src/21_rigorous_experiment_pipeline.py:73-81, 104-114, 471`; `sample-chap-methodology.tex:197` | Models use batch sizes 1 or 2 while custom models use BatchNorm. Different batch statistics and optimization-step counts per epoch confound architecture fairness. The phrase “support fair comparison” is too strong. | Static code inspection. | State the exact batch sizes and fairness limitation. Do not claim strictly identical optimization. A future matched-batch or normalization-controlled study would require retraining. | No for wording; yes for controlled comparison | 20 min | Open |
| P1-04 | Nano/No-Mamba architectures and ablation discussion | No-Mamba has substantially more parameters and a different two-`DoubleConv` bottleneck. Because it outperforms Nano-Mamba, the ablation does not isolate the effect of sequence gating. | 2.288 M vs 1.456 M parameters and code definitions. | Frame it as a non-parameter-matched architectural ablation. State that it does not prove benefit or harm from Mamba-inspired gating alone. Recommend a matched-capacity ablation. | No for thesis; yes for new causal evidence | 30 min | Open |
| P1-05 | `src/21_rigorous_experiment_pipeline.py:369-376`; Methods and Limitations | Preprocessing resizes voxel arrays directly to `256×256×16` without explicit orientation canonicalization or spacing resampling. Physical geometry can be distorted and implementation assumptions are under-specified. | Transform pipeline contains `Load`, channel-first, `Resize`, intensity scaling, tensor conversion only. | Document the exact historical pipeline and limitation. Do not retroactively change preprocessing for reported weights. Add geometry checks and spacing/orientation-aware preprocessing for future retraining. | No for disclosure; yes for changed pipeline | 30–60 min | Open |
| P1-06 | `discover_acdc_cases`, `create_or_load_split`, `split_cases` | Discovery records only five example cases; an existing split is loaded without checking seed, fraction, current patient set, duplicates, or full coverage. Split-time checks only test overlap and non-empty subsets. | `src/21_rigorous_experiment_pipeline.py:311-366`. | Add a complete de-identified manifest with case counts, duplicate checks, label-set/shape/spacing summaries, and deterministic hashes. Validate cached split metadata and exact coverage before any run. | No | 2–4 h | Open |
| P1-07 | `src/21_rigorous_experiment_pipeline.py:42-50`; visualization script | Windows-specific `D:\AI_FYP` paths prevent reproducible use on other machines and encourage manual source editing. | Static code inspection. | Add CLI arguments for data, output, result CSV, split, and checkpoint paths; keep historical defaults only as optional compatibility defaults. | No | 1–2 h | Open |
| P1-08 | `src/16_thesis_visualization.py:100-218` | Quantitative plots embed copied numbers rather than reading the audited result CSV, allowing silent drift between tables and figures. `TkAgg` also blocks headless reproduction. | Static code inspection. | Generate all quantitative figures from `summary_metrics.csv` using a headless backend; record input hash and plotting-script commit. | No | 1–2 h | Open |
| P1-09 | `src/nano_mamba_core.py`; Methods equations and terminology | The module uses a symmetric depthwise 1D convolution, a scalar sigmoid gate, output projection, and residual connection. It has no selective scan. Moreover, `x_proj` creates `2*d_state+1` values but only channel 0 is consumed, leaving most parameters without an output path. Existing class names/comments overstate temporal/Mamba semantics. | `src/nano_mamba_core.py:29,47` and forward graph. | Rename/document it as Mamba-inspired while preserving state-dict compatibility and numerical behaviour. Explain the legacy unused projection channels; count both total and effective trainable parameters if feasible. Correct the thesis equation to match the executed graph. | No for compatible refactor/documentation; yes to redesign and retrain | 1–2 h | Open |
| P1-10 | Checkpoint payload and summary outputs | Checkpoints record only a small config subset; they omit Git SHA, data/split hash, library versions, batch size, model source hash, optimizer state, and hardware. Result JSON may contain machine-specific paths. | `src/21_rigorous_experiment_pipeline.py:509-545`. | Extend future provenance capture and add a sidecar manifest for historical outputs using hashes that can be established now. Do not claim unavailable historical metadata. | No | 1–2 h | Open |
| P1-11 | `benchmark()` and speed claims | FPS is measured on random input at batch 1 with five warm-ups and 30 timed runs, but hardware/software details and uncertainty are absent. Results are implementation-specific; UNet3D's much higher FPS despite more parameters shows parameter count is not a speed proxy. | `src/21_rigorous_experiment_pipeline.py:438-458` and summary CSV. | Report device, precision, versions, batch size, input size, warm-up/timed runs, mean/dispersion, and scope. Preserve FPS as an engineering measurement, not a universal architecture property. | No; rerun only if provenance cannot be recovered | 30–90 min | Open |
| P1-12 | Repository-wide | There are no focused automated checks for split integrity, metric edge cases, model output shape, checkpoint compatibility, result-table consistency, or figure provenance. | File inventory and static compile only. | Add fast tests/validators that do not require ACDC data; document GPU/data-dependent checks separately. | No | 2–4 h | Open |
| P1-13 | Methods architecture caption and block equation | The caption says the deepest bottleneck is “replaced,” while code first applies `DoubleConv(64,128)` and then appends the gated sequence block. The current abstraction omits the split gate branch and may imply a full state-space recurrence. | `sample-chap-methodology.tex:134`; `NanoMambaUNet.bottleneck`; `nano_mamba_core.py`. | Change “replacing” to “augmenting” and provide an implementation-faithful gated-convolution equation with explicit raster flattening and residual connection. | No | 45–90 min | Open |

## 6. P2 — presentation, maintainability, and polish issues

| ID | Location | Problem, reason, and likely impact | Evidence | Required correction | Retraining? | Estimate | Status |
|---|---|---|---|---|---|---:|---|
| P2-01 | `thesis.tex:46-84` | Candidate name, degree, title, and field are filled, but the known matric number is blank. Signature, declaration dates, passport/identity number, and witness fields should remain blank until formal signing. | Declaration source and project administrative record. | Fill only the matric/registration number in the source. Keep identity, signature, witness, and date fields blank. | No | 10 min | Open |
| P2-02 | `thesis.tex:86-87`; `myacronyms.tex` | A placeholder NLP acronym file is loaded even though it is unrelated and the acronym list is not printed. This causes a glossary warning and risks accidental irrelevant content. | LaTeX source and build log. | Remove the unused glossary load or replace it with an actual, printed project acronym list. | No | 10–30 min | Open |
| P2-03 | `myrefs.bib`, `myrefs_extra.bib`, generated bibliography | The database contains unused, duplicate, malformed, and likely incorrect entries. Literal `and others` produces visibly poor “. . . others” references. The V-Net title is duplicated under two keys. | Citation-key inventory and `.blg` warnings. | Keep only verified cited entries, use complete/consortium author data, add missing Otsu/SegResNet/software citations where claims require them, and rebuild twice to clear warnings. | No | 2–4 h (overlaps P0-03) | Open |
| P2-04 | `sample-chap-architecture.tex`, `sample-chap-results.tex`, `thesis-p1.tex` | Legacy/P1 files remain beside final sources and contain temporal-state-space claims, obsolete numerical results, and future-tense text. Although not currently included, they can be compiled or copied by mistake and confuse repository reviewers. | `thesis.tex:111-115` confirms they are inactive; stale content remains tracked. | Move to a clearly labelled archive or prepend strong legacy/non-evidence notices. Ensure the build script lists the only canonical entry point. | No | 20–40 min | Open |
| P2-05 | Uploaded PDF around the qualitative-results page and limitations page | Forced float placement creates a large blank region, and the limitations list is visually dense. Figure labels are readable but small. | Full 69-page render inspection. | Reflow after removing/replacing the invalid qualitative figure; use normal float placement and concise limitation subsections. Re-render every page for clipping/orphans. | No | 1–2 h | Open |
| P2-06 | Thesis source and submission checklist | No reproducible word-count record is included. Approximate source counts can include LaTeX commands, references, and inactive material. | `texcount` is not currently part of the repository workflow. | Add a documented `texcount` command/config and store the final count in the checklist; distinguish body text from captions, references, and appendices. | No | 20–40 min | Open |
| P2-07 | Chapter 1 scope boundary and viva preparation | The fixed title says “Spatio-Temporal,” while the completed study is 3D ED/ES volume segmentation and not temporal tracking. The thesis is mostly cautious, but the title-to-implementation evolution needs a compact explicit defence. | Fixed title, tensor pipeline, and source text. | Add a scope-evolution paragraph and a viva answer: “temporal” is project motivation/future direction; the evaluated module models a flattened spatial token sequence only. Never relabel depth as time. | No | 20–30 min | Open |
| P2-08 | Python sources | Core scripts contain informal comments, emoji, mixed-language explanations, long lines, and weak type/documentation coverage. This reduces examiner confidence and maintainability. | Static source review. | Professionalize comments/docstrings and formatting without changing numerical behaviour; run syntax/style checks. | No | 1–2 h | Open |
| P2-09 | Submission artefacts | The current repository does not contain a final P2 checklist, one-command reproduction guide, 15-minute presentation, Q&A bank, or consolidated scientific-boundary list. | File inventory and school briefing. | Produce these after P0 evidence is stabilized so they inherit the same numbers, limitations, and provenance. | No | 4–7 h | Open |

## 7. School-requirement cross-check

| Requirement from supplied school material | Current state | Action |
|---|---|---|
| P2 report: Chapters 1–5 plus Results, Discussion/Limitations, Conclusion/Future Work, and references | Structurally present as Chapters 1–8 | Retain structure; correct evidence and references |
| Report uses the supplied/project template | Uses `umalayathesis` coursework layout | Confirm with supervisor/faculty if a newer official cover/declaration template is required; do not silently replace at deadline |
| P2 assessment weighting: report 40%, presentation 20% | No final P2 presentation in repository | Build an evidence-aligned 15-minute deck |
| Presentation duration: 15 minutes plus 15 minutes Q&A | No final timing or Q&A bank | Prepare approximately 12–14 slides, speaker notes, timing, and question bank |
| Apply technical skills and explain completed work | Code/results exist, but provenance and limitations need consolidation | Add reproducibility guide, tests, evidence manifest, and defence boundaries |
| Exact registered title | Thesis title matches supplied Section B title | Keep unchanged everywhere |

## 8. Reference-audit protocol

Every cited entry must pass all of the following before submission:

1. The DOI/title/authors/year/venue match a primary publisher, Crossref DOI landing page, PubMed record, arXiv record, or official conference proceedings.
2. The cited source actually supports the sentence, not merely the general topic.
3. Author order and full author/consortium representation are correct; no literal `and others` shortcuts remain.
4. Software and dataset claims cite the official paper or official documentation.
5. Duplicate and uncited records are removed from the submission bibliography database.
6. The PDF reference list is visually inspected after a clean BibTeX rebuild.

## 9. Remaining confirmations from the original experiment machine

The numerical artefacts, epoch logs, and checkpoint metadata/hashes have now
been recovered. Complete historical attestation still requires evidence or a
truthful candidate confirmation for four links:

- the six current checkpoint files are unchanged from the 9 June run;
- the current `nanomamba` environment is the historical training/benchmark
  environment;
- the exact command and working directory are recovered from terminal, IDE, or
  contemporaneous notes; and
- the current ACDC tree is the historical dataset snapshot, after the new
  200-case NIfTI content-manifest tool passes.

If any link cannot be established, leave its confirmation flag false. The
thesis already reports the boundary, so no reconstruction or guessed command is
permitted. A qualitative figure remains optional and excluded from the current
submission evidence.

## 10. Final status matrix

### P0

| ID | Final status | Resolution or retained boundary |
|---|---|---|
| P0-01 | Closed | Visualisation now fails closed on checkpoint/split/case provenance; the invalid qualitative figure and caption were removed. |
| P0-02 | Closed | Six recovered 150-epoch logs are byte-hashed and cross-validated; the new curve is generated directly from them. |
| P0-03 | Closed | The false record was removed; 27 active citations map to 27 cleaned bibliography entries. |
| P0-04 | Closed | One canonical 73-page PDF is built from the audited source; `FINAL_ARTIFACT_MANIFEST.json` binds source commit and hashes. |
| P0-05 | Closed | Figure paths are relative; three summary-derived figures and one six-log training figure are present with hash provenance. |
| P0-06 | Accepted limitation | Sensitive current-branch copies were removed. Historical Git objects remain recoverable until a separately coordinated history rewrite. |
| P0-07 | Closed | The convention is disclosed; zero of 720 recovered class scores equals 1.0, so the empty-empty branch did not affect the reported table. |
| P0-08 | Partially closed | Case rows, logs, and checkpoint records are recovered and validated. Four historical identity/command/data links remain accepted limitations. |

### P1

| ID | Final status | Resolution or retained boundary |
|---|---|---|
| P1-01 | Closed | All final materials use “held-out validation” and explicitly deny an independent test estimate. |
| P1-02 | Closed with boundary | Deterministic patient-level paired bootstrap intervals are reported as post-hoc descriptive analyses; no significance or independent-test claim is made. |
| P1-03 | Closed | Exact batch sizes and BatchNorm/fairness limitations are disclosed. |
| P1-04 | Closed | No-Mamba is presented as a non-parameter-matched architectural ablation, not a causal gate test. |
| P1-05 | Closed | Historical resize preprocessing and geometry limitations are documented without altering reported weights. |
| P1-06 | Accepted limitation | A new post-hoc 200-case NIfTI content-manifest tool is provided; historical dataset identity remains open until the candidate runs it and truthfully confirms the snapshot. |
| P1-07 | Closed | Data/output/project/model locations are CLI arguments; historical paths remain compatibility defaults only. |
| P1-08 | Closed | Quantitative figures read the audited CSV/logs using a headless backend and write hash provenance. |
| P1-09 | Closed | Code and thesis now describe an implementation-faithful Mamba-inspired gate, preserve checkpoint aliases, and disclose unused projection outputs. |
| P1-10 | Partially closed | Six checkpoint hashes/metadata are recovered and internally consistent; historical set identity remains unconfirmed. Future manifests also capture shape fingerprints and mtimes. |
| P1-11 | Accepted limitation | A detailed current RTX 4060/PyTorch environment capture exists, but its identity with the historical benchmark environment is unconfirmed. |
| P1-12 | Closed | Seventeen data-free evidence/collection/visualisation regression tests and three optional PyTorch shape/compatibility tests were added. |
| P1-13 | Closed | Architecture caption and equations now match the executed augmenting gate and residual graph. |

### P2

| ID | Final status | Resolution or retained boundary |
|---|---|---|
| P2-01 | External action required | Private declaration identifiers/signatures intentionally remain outside Git and must be completed before portal upload. |
| P2-02 | Closed | The canonical thesis selects the new `noglossaries` class option and no longer creates an unused glossary. |
| P2-03 | Closed | Duplicate, unused, malformed, and false bibliography entries were removed; the final PDF was rebuilt and inspected. |
| P2-04 | Closed | Obsolete P1 entry point/chapters and conflicting P1 binaries were removed; historical exploratory scripts are quarantined by a non-evidence notice. |
| P2-05 | Closed | Invalid float content was removed; all 73 final pages were rendered and reviewed. |
| P2-06 | Closed | `WORD_COUNT.md` records the command, chapter counts, total, and counting boundary. |
| P2-07 | Closed | Thesis, boundary sheet, slides, and Q&A all give the same fixed-title scope defence. |
| P2-08 | Accepted limitation | Active final scripts were professionalised; informal exploratory code remains only under the clearly labelled `src/legacy/` archive. |
| P2-09 | Closed | Final checklist, reproducibility guide, 13-slide timed deck, 25-question viva bank, and scientific-boundary sheet are present. |

## 11. Independent re-review additions

The 2026-08-21 adversarial pass challenged the earlier closure language and
added the following corrections without changing the reported result values:

- renamed the default validator outcome from a broad “core pass” to the
  narrower **aggregate-consistency pass**;
- added range, cardinality, CSV/JSON/path, full-manifest, 40-row per-case,
  150-epoch log, best-row, checkpoint, environment, and command checks;
- reconstructed and hash-bound the limited Git/ZIP source timeline while
  explicitly refusing to treat it as proof of the historical run;
- changed “valid image-label pairs” to “non-empty paired paths” and retained the
  absent 200-case historical manifest as a strict-closure gap;
- bound optional qualitative evidence to the exact split, summary row,
  checkpoint epoch/Dice, architecture parameter count, and source hashes;
- documented that the training Dice term includes background under the MONAI
  default whereas the reported evaluation averages the three foreground classes;
- replaced the SegMamba preprint record with its MICCAI 2024 proceedings record;
- made current-source lineage hashing invariant to Git's Windows CRLF checkout
  conversion while retaining rejection of every substantive source change, and
  added repository line-ending policy plus a dedicated regression test; and
- isolated TeX font caches in the temporary build directory and rechecked all
  73 pages, including the corrected page break at the end of the limitations.

## 12. Recovered closure re-review (2026-08-21)

The uploaded closure ZIP was unpacked without trusting filenames, verified
against its internal checksum list, and audited independently of the collector
report. The following additional checks now pass:

- all 240 case rows have identical validation-case coverage across models and
  exactly reproduce the four summary Dice values;
- all 900 epoch rows are finite and contiguous, and each selected best row
  matches both the summary and recovered checkpoint record;
- all six checkpoint records have unique hashes, readable state dictionaries,
  expected model/config metadata, and matching selected epoch/Dice values;
- a 100,000-replicate independent NumPy cross-check reproduced the direction
  and approximate intervals of the validator's deterministic 10,000-replicate
  patient bootstrap;
- zero exact-unity scores among 720 class entries close the empty-empty metric
  concern for this result table; and
- ZIP member times form a plausible sequential run from discovery through the
  six models and final summary, while explicitly remaining non-attestational
  because ZIP timestamps encode no timezone or signer.

The collector was also corrected so future `SHA256SUMS.txt` files include the
generated audit report, checkpoint manifests include state-dictionary shape
fingerprints and mtimes, environment captures include Conda history and
`nvidia-smi` with credentials/user-home paths redacted, and a new dataset tool
audits all 200 three-dimensional NIfTI pairs without copying patient data. The
collector now returns exit 2 after writing an incomplete bundle and rejects an
unexpected audit exit as invalid evidence, so automation cannot report false
strict closure.

## 13. Resolution log

| Commit | Date | Issues changed | Verification |
|---|---|---|---|
| `21e9dd6293635eee4527c0f030a0cf7ac1d56768` | 2026-08-20 | Initial P0/P1/P2 baseline and audited evidence copy | Static review; PDF render review; result/split arithmetic and hash checks |
| `07493af8f56c29f85f52fc5b77c38062662a77ce` | 2026-08-20 | Current-branch school-document privacy cleanup | Tracked-file inventory and replacement summary review |
| `f6a40b5b1de2b1fd7b4100a2fba3b2b7bdc071ee` | 2026-08-20 | Result/figure provenance and implementation-faithful bottleneck naming | CSV-derived figure regeneration; fail-closed provenance checks |
| `c8698bc9e08731ee1bd6ace27900625183b571b3` | 2026-08-20 | Canonical evidence-bounded thesis, references, privacy fields, legacy cleanup | 71-page A4 build; full render review; no undefined citations/references or overfull boxes |
| `ebead16476573e7883dfe7831c7fea119ea91caf` | 2026-08-20 | Evidence validator, closure collector, future-run provenance, tests, reproduction guide | Aggregate consistency pass; strict closure intentionally fails; 3 data-free tests pass; 3 PyTorch tests skip in this environment |
| `81e41f0bb185db616851a3bac9ccde83cb49693c` | 2026-08-20 | Final defence deck, viva bank, and submission checklist | 13 slides rendered; no overflow; full montage and individual slide review |
| `73507db054c09efabf1a746a1111fe79ac1d9b64` | 2026-08-21 | Independent re-review: scoped audit status, strict artefact validation, source lineage, data-discovery wording, loss/evaluation boundary, qualitative guards, proceedings citation, build isolation, and final PDF | Aggregate consistency pass; strict closure exits 2 on documented gaps; 8 data-free tests pass; 3 PyTorch tests skip; clean 71-page A4 build and full visual review |
| `be8666bffd0068de27a8b6c0ff4b3d5164853043` | 2026-08-21 | Windows checkout portability: CRLF-neutral current-source hashing, explicit Git text/binary policy, regenerated evidence report, and regression coverage | 9 data-free tests pass, including CRLF acceptance and substantive-change rejection; 3 architecture tests passed in the user-reported PyTorch run; strict closure still exits 2 only on documented historical gaps |
