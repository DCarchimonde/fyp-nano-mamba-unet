# Thesis Word-Count Record

**Audit date:** 2026-08-23

**Canonical entry point:** `Universiti_Malaya_Thesis_Template/thesis.tex`

The five active body-chapter sources contain approximately **16,400 words** by
`detex | wc -w`:

| Active chapter source | Words |
|---|---:|
| `sample-chap-intro.tex` | 2,679 |
| `sample-chap-litreview.tex` | 2,539 |
| `sample-chap-methodology.tex` | 4,249 |
| `sample-chap-results-p2.tex` | 4,516 |
| `sample-chap-conclusion.tex` | 2,417 |
| **Total** | **16,400** |

Reproduction command from the template directory:

```bash
detex sample-chap-intro.tex sample-chap-litreview.tex \
  sample-chap-methodology.tex sample-chap-results-p2.tex \
  sample-chap-conclusion.tex | wc -w
```

The English and Malay abstracts contain approximately 463 and 446 words,
respectively, by the same command and are each below 500 words. The body
`detex` count excludes abstracts, front matter, references, and inactive
legacy/P1 files. PDF-text counts are expected to be slightly higher because
they include headings, captions, and page artefacts. If the faculty mandates
`texcount`, record its final output separately because counting rules differ.
