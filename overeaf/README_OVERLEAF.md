# ViPragSent paper package for Overleaf

This folder is a self-contained LaTeX source package for the current paper.

## Upload and compile

1. Upload the contents of this folder to a new Overleaf project.
2. Set `main.tex` as the project main document.
3. Use LuaLaTeX or XeLaTeX so the Vietnamese Unicode text is handled directly.
4. Compile with the normal Overleaf bibliography pass; `references.bib` and
   the root-level `acl_natbib.bst` are already included. The root-level copy is
   intentional because BibTeX does not recursively search `template/` on all
   Overleaf compiler configurations.

The manuscript is kept in the current anonymous review configuration from the
repository (`\usepackage[review]{template/acl}`). The generated section files
are the frozen LaTeX inputs used to produce the current PDF; the Markdown
manuscript converter and local PowerShell build script are intentionally not
needed on Overleaf.

Included files are limited to the entry point, section sources, bibliography,
ACL style/bibliography style, and the three figures referenced by the paper.
No `.env`, dataset, checkpoint, private report, log, or local build output is
included.
