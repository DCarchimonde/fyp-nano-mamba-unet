#!/usr/bin/env bash
set -euo pipefail

p2_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
p2_source_dir="$p2_repo_root/paper_write/Universiti_Malaya_Thesis_Template"
p2_build_dir="$(mktemp -d /tmp/nano-mamba-thesis.XXXXXX)"
p2_texmf_var="$p2_build_dir/texmf-var"
p2_texmf_config="$p2_build_dir/texmf-config"
p2_texmf_home="$p2_build_dir/texmf-home"
mkdir -p "$p2_texmf_var" "$p2_texmf_config" "$p2_texmf_home"
export TEXMFVAR="$p2_texmf_var"
export TEXMFCONFIG="$p2_texmf_config"
export TEXMFHOME="$p2_texmf_home"

for p2_command in pdflatex bibtex pdfinfo sha256sum rg; do
  command -v "$p2_command" >/dev/null 2>&1 || {
    echo "Required command is unavailable: $p2_command" >&2
    exit 1
  }
done

cd "$p2_source_dir"

pdflatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -output-directory="$p2_build_dir" \
  thesis.tex >"$p2_build_dir/pdflatex-1.log"

(
  cd "$p2_build_dir"
  BIBINPUTS="$p2_source_dir:" bibtex thesis >bibtex.log
)

for p2_pass in 2 3 4; do
  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$p2_build_dir" \
    thesis.tex >"$p2_build_dir/pdflatex-$p2_pass.log"
done

if rg -n -F \
  -e "undefined citations" \
  -e "There were undefined references" \
  -e "Label(s) may have changed" \
  -e "Overfull \\hbox" \
  -e "Overfull \\vbox" \
  -e "No \\printgloss" \
  "$p2_build_dir/thesis.log"; then
  echo "Thesis acceptance check failed; build retained at $p2_build_dir" >&2
  exit 1
fi

pdfinfo "$p2_build_dir/thesis.pdf" | rg "^(Pages|Page size|File size|PDF version)"
sha256sum "$p2_build_dir/thesis.pdf"
cp "$p2_build_dir/thesis.pdf" "$p2_source_dir/thesis.pdf"

echo "Canonical thesis written to $p2_source_dir/thesis.pdf"
echo "Build logs retained at $p2_build_dir"
