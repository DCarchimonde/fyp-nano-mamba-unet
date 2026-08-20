# Thesis Word-Count Record

**Audit date:** 2026-08-21

**Canonical entry point:** `Universiti_Malaya_Thesis_Template/thesis.tex`

The five active body-chapter sources contain approximately **12,112 words** by
`detex | wc -w`:

| Active chapter source | Words |
|---|---:|
| `sample-chap-intro.tex` | 2,328 |
| `sample-chap-litreview.tex` | 2,162 |
| `sample-chap-methodology.tex` | 2,698 |
| `sample-chap-results-p2.tex` | 2,923 |
| `sample-chap-conclusion.tex` | 2,001 |
| **Total** | **12,112** |

Reproduction command from the template directory:

```bash
detex sample-chap-intro.tex sample-chap-litreview.tex \
  sample-chap-methodology.tex sample-chap-results-p2.tex \
  sample-chap-conclusion.tex | wc -w
```

As a cross-check, `pdftotext` over the body-page range returned approximately
12,366 tokens separated as words; that method includes headings, captions, and
page artefacts. The `detex` count excludes abstracts, front matter, references,
and inactive legacy/P1 files. If the faculty mandates `texcount`, record its
final output separately because counting rules differ.
