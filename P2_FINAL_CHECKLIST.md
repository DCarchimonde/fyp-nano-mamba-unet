# P2 Final Submission Checklist

## Submission gate

| Item | Status | Evidence / final action |
|---|---|---|
| Registered title exact and unchanged | Pass | Canonical LaTeX title and PDF cover |
| Chapters 1--8 and references present | Pass | 73-page canonical PDF |
| Main table tied to rigorous CSV | Pass | Aggregate-consistency audit and committed evidence copies |
| Patient split is deterministic, disjoint, and complete | Pass | Seed 42; 80/20 patients; 160/40 cases |
| Quantitative figures use audited evidence | Pass | Three summary-derived PNGs, one six-log training plot, and provenance manifest |
| Recovered training curves | Pass | Six byte-hashed 150-epoch logs; every best row matches summary/checkpoint |
| Invalid qualitative validation figure removed | Pass | Restoration requires explicit checkpoint and validation case |
| False/malformed reference removed | Pass | 27 active citations; 27 verified bibliography keys |
| Clean thesis build | Pass | Exit 0; A4; no undefined references/citations; no overfull boxes |
| Full PDF visual QA | Pass | 73 pages rendered and reviewed |
| Final artifact manifest | Pass | Source commit, hashes, build command, versions, page/slide counts |
| 15-minute defence package | Pass | 13-slide deck with notes plus 25-question viva bank |
| Public-repository privacy cleanup | Partial | Current copies removed; Git-history purge remains separate |
| Original per-case rows and training logs | Pass | Six 40-case tables and six 150-epoch logs recovered and cross-validated |
| Full historical 200-case discovery manifest | Missing | Historical report retained only five examples; future pipeline records all cases |
| Checkpoint hashes/metadata | Partial | Six unique hashes and matching metadata recovered; historical set identity not confirmed |
| Confirmed historical environment/command | Missing | Current environment captured; historical identity and exact command remain unconfirmed |
| Patient-level CIs/paired differences | Pass | 10,000-replicate patient bootstrap, seed 20260820; explicitly post-hoc/descriptive |
| Empty-empty Dice impact | Pass | Zero exact-unity values among 720 recovered class scores; branch did not affect table |
| Independent/official test evaluation | Not performed | Explicit limitation/future work |

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
python -m unittest discover -s tests -v
python src/16_thesis_visualization.py
bash scripts/build_thesis.sh
```

For the evidence-complete gate on the original experiment machine:

```powershell
python src\22_p2_evidence_audit.py --strict-closure
```

## Upload package

- final private thesis PDF;
- final presentation deck;
- any faculty forms required outside the public repository;
- branch/commit identifier and reproducibility guide if requested;
- closure evidence ZIP retained privately or supplied to the supervisor, with
  checkpoint weights and ACDC data excluded unless an approved channel is used.

## Final manual review

- Check portal filename, file-size limit, deadline, and successful upload.
- Open the uploaded PDF from the portal and inspect cover, declaration,
  abstracts, contents, result table/figures, limitations, conclusion, and final
  reference page.
- Confirm that presentation numbers and wording match the thesis exactly.
- Keep the fixed scientific boundaries visible during the viva.
