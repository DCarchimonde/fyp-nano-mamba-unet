# P2 Final Submission Checklist

## Submission gate

| Item | Status | Evidence / final action |
|---|---|---|
| Registered title exact and unchanged | Pass | Canonical LaTeX title and PDF cover |
| Chapters 1--8 and references present | Pass | Canonical 86-page PDF |
| Main table tied to rigorous CSV | Pass | Aggregate-consistency audit and committed evidence copies |
| Patient split is deterministic, disjoint, and complete | Pass | Seed 42; 80/20 patients; 160/40 cases |
| Quantitative figures use experiment evidence | Pass | Spatial summary/training plots plus audited full-cine curve, EF, and endpoint-overlay figures |
| Training-curve interpretation | Pass | Six 150-epoch logs; best epochs match the table; late-epoch drops are discussed |
| Invalid qualitative validation figure removed | Pass | Restoration requires explicit checkpoint and validation case |
| False/malformed reference removed | Pass | 27 active citations; 27 verified bibliography keys |
| Clean thesis build | Pass | Exit 0; A4; no undefined references/citations; no overfull boxes |
| Full PDF visual QA | Pass | All 86 pages rendered and reviewed; full-size checks include the new method/result/overlay pages |
| Final artifact manifest | Pass | Scientific scope, experiment design, hashes, build command, versions, and page/slide counts |
| Exact image-processing explanation | Pass | Thesis method, dedicated slide, bilingual cheat sheet, and viva Q5--12 |
| Exact tensor/sequence explanation | Pass | Input, bottleneck, token, and output shapes tied to code |
| Full-cine coverage | Pass | 20/20 validation patients; 550/550 frames; 1,100 frame rows |
| Full-cine independent audit | Pass | Artifact/hash/source/checkpoint/cohort/arithmetic/bootstrap checks |
| Image and cine processing explanation | Pass | Frame extraction, resize, scaling, fusion, argmax, native restoration, physical units |
| Spatio-temporal claim boundary | Pass | Complete global trajectories; spatial backbone + fixed fusion; no dense motion/strain claim |
| 15-minute defence package | Pass | 14-slide deck; all slides rendered/reviewed; notes on every slide; 60-question viva bank and bilingual cheat sheet |
| Public-repository privacy cleanup | Partial | Current copies removed; Git-history purge remains separate |
| Original per-case rows and training logs | Pass | Six 40-case tables and six 150-epoch logs are present and cross-validated |
| Patient-level CIs/paired differences | Pass | 10,000-replicate patient bootstrap, seed 20260820; explicitly post-hoc/descriptive |
| Empty-empty Dice impact | Pass | Zero exact-unity values among 720 audited class scores; branch did not affect table |
| Independent/official test evaluation | Not performed | Explicit limitation/future work |
| Multi-seed / matched-capacity evaluation | Not performed | Explicit limitation; no causal gate claim |
| Native-space preprocessing / augmentation | Partial | Predictions restored to native array shape and physical units use header spacings; training still uses direct resize and no augmentation |
| Phase and pathology analysis | Pass with boundary | Phase timing evaluated; pathology groups descriptive only (n=1--7) |
| Surface/dense-motion metrics | Not performed | Global centroid/radial surrogates are complete; no Hausdorff/ASD, optical flow, deformation, tissue correspondence, or regional strain claim |

## Private fields before upload

- Copy `submission-private.example.tex` to the ignored
  `submission-private.tex` on the private submission machine.
- Fill the correct matric/registration number.
- Fill identity/passport information only if the official form requires it.
- Add candidate signature/date and witness details through the institution's
  accepted signing workflow; do not commit these values publicly.
- Rebuild and visually recheck the declaration page.

## Final commands

```bash
python src/22_p2_evidence_audit.py
python src/25_spatiotemporal_result_audit.py --result-dir evidence/spatiotemporal_cine/raw --repo-root .
python -m unittest discover -s tests -v
python src/16_thesis_visualization.py
bash scripts/build_thesis.sh
```

`--strict-closure` is an optional archival-forensics mode. It is not a thesis,
presentation, or portal-submission requirement.

## Upload package

- final private thesis PDF;
- final presentation deck;
- any faculty forms required outside the public repository;
- branch/commit identifier and reproducibility guide if requested;
- repository link or commit identifier only if requested by the supervisor.

## Final manual review

- Check portal filename, file-size limit, deadline, and successful upload.
- Open the uploaded PDF from the portal and inspect cover, declaration,
  abstracts, contents, result table/figures, limitations, conclusion, and final
  reference page.
- Confirm that presentation numbers and wording match the thesis exactly.
- Rehearse `presentation/P2_DEFENSE_METHOD_CHEATSHEET.md`, especially the
  spatio-temporal and image-processing answers.
