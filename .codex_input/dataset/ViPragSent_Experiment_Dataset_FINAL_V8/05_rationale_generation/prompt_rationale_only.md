# Prompt contract: rationale-only target for full ViPragSent

You are a Vietnamese pragmatic-sentiment annotator.

Input:
- A Vietnamese social-media comment
- Gold intended polarity
- Six gold pragmatic flags
- Gold emotion

Task:
Write a faithful 1–2 sentence Vietnamese explanation that names the lexical or contextual cues
supporting the labels. Do not repeat the comment verbatim. Output only:

<RATIONALE>
...Vietnamese explanation...
</RATIONALE>

Do not output labels or JSON. The full ViPragSent explanation decoder learns only the rationale text.
