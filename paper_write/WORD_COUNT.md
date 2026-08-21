# Thesis Word-Count Record

**Audit date:** 2026-08-21

**Canonical entry point:** `Universiti_Malaya_Thesis_Template/thesis.tex`

The five active body-chapter sources contain approximately **12,411 words** by
`detex | wc -w`:

| Active chapter source | Words |
|---|---:|
| `sample-chap-intro.tex` | 2,329 |
| `sample-chap-litreview.tex` | 2,162 |
| `sample-chap-methodology.tex` | 2,850 |
| `sample-chap-results-p2.tex` | 3,096 |
| `sample-chap-conclusion.tex` | 1,974 |
| **Total** | **12,411** |

Reproduction command from the template directory:

```bash
detex sample-chap-intro.tex sample-chap-litreview.tex \
  sample-chap-methodology.tex sample-chap-results-p2.tex \
  sample-chap-conclusion.tex | wc -w
```

The `detex` count excludes abstracts, front matter, references, and inactive
legacy/P1 files. PDF-text counts are expected to be slightly higher because
they include headings, captions, and page artefacts. If the faculty mandates
`texcount`, record its final output separately because counting rules differ.
