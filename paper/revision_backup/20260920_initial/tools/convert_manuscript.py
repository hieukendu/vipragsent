"""Mechanically convert the author Markdown snapshot into ACL section files.

This converter owns no scientific rewriting. It only maps Markdown headings,
lists, tables, inline code, bold spans, display math, and citations to LaTeX;
the figure insertions are production-owned placement snippets.
"""

from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript.md"
SECTION_DIR = ROOT / "sections"

TOP_LEVEL = {
    "Abstract": "abstract",
    "1 Introduction": "introduction",
    "2 Related Work": "related_work",
    "3 Method": "method",
    "4 Experimental Setup": "experiments",
    "5 Results": "results",
    "6 Analysis and Discussion": "discussion",
    "7 Limitations": "limitations",
    "8 Ethics and Responsible Use": "ethics",
    "9 Conclusion": "conclusion",
}

FIGURES = {
    "3.2 XLM-R-large encoder and prediction heads": r"""
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/figure_architecture_schematic.png}
  \caption{ViPragSent architecture and training objective. XLM-R-large produces a hidden sequence that feeds first-nonpadding pooling for six pragmatic binary heads, three-way polarity, and seven-way emotion prediction. The same hidden sequence is used as memory by a two-layer rationale decoder with teacher-forced shifted rationale tokens during training only; its rationale loss contributes to the objective and is not an inference output.}
  \label{fig:architecture}
\end{figure*}
""".strip(),
    "5.4 Q3: pragmatic label-budget behavior": r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/figure_q3_sarcasm_budget.png}
  \caption{Q3 sarcasm F1 across pragmatic-label budgets. Curves show artifact-derived means and sample SD over three logical runs ($n=3$; run labels 21, 22, and 23, corresponding to remote artifact seeds 20260521, 20260522, and 20260523). The target is shown with PhoBERT fine-tune and Vistral-7B SFT for context.}
  \label{fig:q3-sarcasm-budget}
\end{figure}
""".strip(),
    "5.5 Q4: calibration": r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/figure_q4_reliability.png}
  \caption{Q4 reliability diagrams for the calibration comparison. Curves aggregate the supplied per-label reliability bins over three logical runs ($n=3$; run labels 21, 22, and 23, corresponding to remote artifact seeds 20260521, 20260522, and 20260523). The dashed line denotes perfect calibration; the figure is descriptive and does not imply target dominance.}
  \label{fig:q4-reliability}
\end{figure}
""".strip(),
}

TABLE_CAPTIONS = {
    "4.1 Separate experimental cohorts": ("Experimental cohorts and protocols; quantitative entries elsewhere summarize three saved runs where reported ($n=3$, mean $\\pm$ sample SD).", None),
    "5.1 Q1a: in-domain pragmatic performance": ("Q1a in-domain pragmatic performance on the 2,000-example ID/gold test split. Entries are mean $\\pm$ sample SD over $n=3$ logical runs (21, 22, 23), in percentage points of binary macro-F1 per phenomenon and macro-pragmatic F1.", "tab:q1a"),
    "5.2 Q1b: external retention": ("External retention on the UIT-VSFC, UIT-VSMEC, and AIVIVN test cohorts. Entries are mean $\\pm$ sample SD over $n=3$ saved runs; columns report macro-F1 in percentage points and their unweighted ordinary-F1 aggregate.", "tab:q1b"),
    "5.3 Q2: auxiliary-task and uncertainty ablations": ("Q2 follow-up ablations. Macro-pragmatic F1 is evaluated on the Q2 test split; the saved \\texttt{polarity\\_dev\\_ece} field is a development-split three-way polarity ECE, shown multiplied by $10^3$; relative cost is normalized to the full Q2 XLM-R run. Entries are mean $\\pm$ sample SD over $n=3$ runs.", "tab:q2"),
    "5.4 Q3: pragmatic label-budget behavior": ("Q3 pragmatic label-budget behavior on the fixed test split. Entries are mean $\\pm$ sample SD over $n=3$ logical runs; sarcasm and macro columns are F1 in percentage points, and -- denotes an unavailable baseline cell.", "tab:q3"),
    "Appendix B. Table-specific protocol and missingness": ("Recorded table-specific protocol and missingness.", "tab:appendix-protocol"),
}


def latex_escape_code(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def latex_code(value: str) -> str:
    if len(value) <= 24:
        return rf"\texttt{{{latex_escape_code(value)}}}"
    chunks = [value[index : index + 8] for index in range(0, len(value), 8)]
    return r"\texttt{" + r"\allowbreak{}".join(latex_escape_code(chunk) for chunk in chunks) + "}"


def convert_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", lambda m: latex_code(m.group(1)), text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: rf"\textbf{{{m.group(1)}}}", text)
    text = re.sub(r"\s\+/-\s", r" $\\pm$ ", text)
    text = text.replace("Figure 1", r"Figure~\ref{fig:q3-sarcasm-budget}")
    return text


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [convert_inline(cell.strip()) for cell in stripped.split("|")]


def table_block(lines: list[str], section_key: str) -> list[str]:
    header = split_table_row(lines[0])
    rows = [split_table_row(line) for line in lines[2:]]
    count = len(header)
    if any(len(row) != count for row in rows):
        raise ValueError(f"Malformed table in {section_key}")
    wide = count >= 6 or section_key in {"5.2 Q1b: external retention", "5.4 Q3: pragmatic label-budget behavior"}
    env = "table*" if wide else "table"
    caption, label = TABLE_CAPTIONS.get(section_key, (None, None))
    if section_key.startswith("Appendix A.") and count == 2:
        output = [r"\begin{description}", r"\setlength{\itemsep}{2pt}"]
        output.extend(rf"\item[\textbf{{{row[0]}}}] {row[1]}" for row in rows)
        output.append(r"\end{description}")
        return output
    width = r"\textwidth" if wide else r"\columnwidth"
    if section_key == "5.1 Q1a: in-domain pragmatic performance":
        colspec = r">{\raggedright\arraybackslash}p{1.15in}" + (count - 1) * r" X"
        for row in rows:
            if row[0] == "Idiom/figurative":
                row[0] = r"Idiom/\\figurative"
            for index, cell in enumerate(row[1:], start=1):
                row[index] = re.sub(r"(?<![A-Za-z])([0-9]+\.[0-9]{2}) \$\\pm\$ ([0-9]+\.[0-9]{2})", r"\\mbox{\1 $\\pm$ \2}", cell)
    else:
        colspec = r">{\raggedright\arraybackslash}X" + (count - 1) * r" >{\raggedright\arraybackslash}X"
    output = [rf"\begin{{{env}}}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabularx}}{{{width}}}{{{colspec}}}", r"\hline"]
    output.append(" & ".join(header) + r" \\")
    output.append(r"\hline")
    output.extend(" & ".join(row) + r" \\" for row in rows)
    output.extend([r"\hline", r"\end{tabularx}"])
    if caption:
        if label is None:
            output.append(rf"\caption*{{{caption}}}")
        else:
            output.append(rf"\caption{{{caption}}}")
            output.append(rf"\label{{{label}}}")
    output.append(rf"\end{{{env}}}")
    return output


def paragraph(lines: list[str]) -> str:
    return convert_inline(" ".join(line.strip() for line in lines if line.strip()))


def heading_title(heading: str) -> str:
    return re.sub(r"^(?:\d+(?:\.\d+)*\s+|Appendix\s+[A-Z]\.\s*)", "", heading)


def convert_body(lines: list[str], section_key: str) -> list[str]:
    output: list[str] = []
    i = 0
    active_table_key = section_key
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip() == "$$":
            math_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Unclosed display math in {section_key}")
            output.append(r"\[")
            output.append(r"\begin{aligned}")
            if any(r"\mathcal{L}_{\mathrm{cls+rat}}" in math_line for math_line in math_lines):
                output.extend([
                    "s_t^{\\mathrm{c}}=\\operatorname{clip}(s_t,-5,5), \\\\",
                    "\\mathcal{L}_{\\mathrm{cls+rat}}=\\sum_{t\\in\\mathcal{T}}\\Bigl[\\tfrac12 e^{-s_t^{\\mathrm{c}}}(m_t\\ell_t)+\\tfrac12s_t^{\\mathrm{c}}\\Bigr] \\\\",
                    r"{}+\beta\mathcal{L}_{\mathrm{rat}},\qquad \beta=0.3.",
                ])
            else:
                for index, math_line in enumerate(math_lines):
                    math_line = math_line.replace(r"\left[", "[").replace(r"\right]", "]").replace(r"\left(", "(").replace(r"\right)", ")")
                    suffix = r" \\" if index < len(math_lines) - 1 else ""
                    output.append(math_line + suffix)
            output.extend([r"\end{aligned}", r"\]"])
            i += 1
            continue
        if line.startswith("### "):
            heading = line[4:].strip()
            active_table_key = heading
            output.append(rf"\subsection{{{convert_inline(heading_title(heading))}}}")
            if heading in FIGURES:
                output.extend(["", FIGURES[heading], ""])
            i += 1
            continue
        if line.lstrip().startswith("|"):
            table_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            output.extend(table_block(table_lines, active_table_key))
            output.append("")
            continue
        if re.match(r"^\s*\*\s+", line) or re.match(r"^\s*-\s+", line):
            items: list[str] = []
            while i < len(lines):
                match = re.match(r"^\s*[*-]\s+(.*)$", lines[i])
                if match:
                    items.append(convert_inline(match.group(1).strip()))
                    i += 1
                    while i < len(lines) and lines[i].strip() and not re.match(r"^\s*[*-]\s+", lines[i]) and not lines[i].startswith("### "):
                        items[-1] += " " + convert_inline(lines[i].strip())
                        i += 1
                else:
                    break
            output.extend([r"\begin{itemize}", *(rf"\item {item}" for item in items), r"\end{itemize}", ""])
            continue
        paragraph_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("### ") and not lines[i].lstrip().startswith("|") and lines[i].strip() != "$$" and not re.match(r"^\s*[*-]\s+", lines[i]):
            paragraph_lines.append(lines[i])
            i += 1
        output.append(paragraph(paragraph_lines))
        output.append("")
    return output


def parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^## (.+)$", line)
        if match:
            current = match.group(1).strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def main() -> None:
    sections = parse_sections(SOURCE.read_text(encoding="utf-8"))
    missing = [heading for heading in TOP_LEVEL if heading not in sections]
    if missing:
        raise SystemExit(f"Missing manuscript headings: {missing}")
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    for heading, filename in TOP_LEVEL.items():
        body = convert_body(sections[heading], heading)
        if heading != "Abstract":
            body = [rf"\section{{{convert_inline(heading_title(heading))}}}", ""] + body
        (SECTION_DIR / f"{filename}.tex").write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
    appendix_lines: list[str] = []
    for heading, lines in sections.items():
        if heading.startswith("Appendix "):
            appendix_lines.append(rf"\section{{{convert_inline(heading_title(heading))}}}")
            appendix_lines.extend(convert_body(lines, heading))
            appendix_lines.append("")
    (SECTION_DIR / "appendix.tex").write_text("\n".join(appendix_lines).rstrip() + "\n", encoding="utf-8")
    manifest = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper(),
        "section_files": sorted(path.name for path in SECTION_DIR.glob("*.tex")),
        "conversion_scope": "mechanical markdown-to-LaTeX conversion; no semantic prose edits",
    }
    (ROOT / "build" / "manuscript_conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(TOP_LEVEL)} required sections and appendix to {SECTION_DIR}")


if __name__ == "__main__":
    main()
