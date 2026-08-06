# Prompt contract: rationale plus structured labels

Use this contract only for CoT-only or explanation-at-inference variants.

Output exactly:

<RATIONALE>
...Vietnamese explanation...
</RATIONALE>
<LABELS>
{"implicit":0,"sarcasm":0,"irony":0,"idiom":0,
 "code_switching":0,"mocking":0,
 "polarity":"neutral","emotion":"other"}
</LABELS>

The parser must extract the `<LABELS>` block and parse JSON. Do not infer labels from words or
synonyms inside the rationale.
