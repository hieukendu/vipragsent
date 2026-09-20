"""Append source-derived Sailor/Vistral Q1b packs and rebuild CSV views.

All numbers below are copied from the live, pinned
`external_retention_metrics.json` files recorded in the source table. The
script uses csv.DictWriter for every rewritten CSV and is idempotent.
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
BASE = "campaigns/vipragsent-v7-6693e7e9eef08ef1/SHARD_GPU40_7B_INTEGRATION"
GOLD = {
    "aivivn": "F6F6A78EBADE3BC1038A3B1CFE05FF44E172AF024BBBC8A31079EFCF8133C6D3",
    "vsfc": "C27B61C70DE1090BD0A12F0F2824148D1691C68FB9CB18A07F23AE5282E75D96",
    "vsmec": "1D2E604BD06A4AAAAD33A3B98DE2FB910E9020F4D468EF8BC16DDCDEADB3D955",
}
PACKS = [
    # family, seed, overflow repo, Hub tree revision, source checkpoint SHA,
    # metric-file SHA, AIVIVN, VSFC, VSMEC, ordinary-F1
    ("sailor_multitask_8head", "20260521", "014", "c9de28043c9ead854e209517614428c78225556c", "33129E0387546CED1397CF9DD98CF67885E92F112B272BB3FC518E9524D35439", "914A54C9A59409A0395F0BB4E12331AE0480A6E97C62EE9B7311ED2D9883776A", .4708813901452613, .3395458883427234, .41539850836552955, .4086085956178381),
    ("sailor_multitask_8head", "20260522", "010", "42b3a6f32086a3bc7e0a833ae550a5fb0ff261ce", "D921D037CE09E8884EEA95688A7787455C3C45251DD5D766635115CDD1D74B7E", "F50837F12710F37F7BB5E67FEA3BD79BDA70ED5495D01638451925A0552A2269", .4933445704528252, .3142476250449751, .40121649558994044, .40293623036258025),
    ("sailor_multitask_8head", "20260523", "004", "82914ca97e58dc65d063f69ebe045db01ae1ae05", "8D57BFF43C87E0CB040FC242A11F84D2555D59F6A1759FCA1CF2F5996D6BD0C1", "9249EDA7852DE6C9876746907B04D1E7CC2E689B4A349C6B31CC261C0C467FE4", .4973243371352245, .37691885218914417, .4043902117880532, .4262111337041406),
    ("vistral_multitask_8head", "20260521", "011", "93d075edd8d2f9347edf41d5b2c28f330ce6f806", "BF7A3A98ABCD68F4959E3B86A515E4473DC6578EDB1BDDD10F197F092EFBC419", "38FA08E2659C6730539AFDE780A813C18EA6691A48E9496FE783ADE98838777B", .4829761744586365, .2807960819682657, .38173888931808636, .38183704858166284),
    ("vistral_multitask_8head", "20260522", "012", "fc492ad5184a6fde8401807ed3735a30851b0ec3", "26D562DB1D54F33DA6A6C819AC545E4F54CECC5039CC3D408C4311C8EC749E35", "307D8DF367215934D906B0B44D55C0EE57551CB340D6A57B23C95BFA3244F09B", .4988034295737695, .2596503108388579, .3911579602300378, .38320390021422174),
    ("vistral_multitask_8head", "20260523", "011", "93d075edd8d2f9347edf41d5b2c28f330ce6f806", "744483CB57B88CB3C029EA095BD53AFA4CFA7B160B57E98D7A22E8788F78A47C", "1F9F799CBE013CC0231606A6861C81C45BD0FA092417BE2169E001A7836F7C03", .48911825321204355, .38014451973299695, .4102755169307199, .4265127632919201),
]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


raw_path = ROOT / "q1b_raw_extraction.csv"
raw_fields = list(csv.DictReader(raw_path.open(encoding="utf-8-sig", newline="")).fieldnames)
raw = [r for r in read(raw_path) if r["actual_recipe"] not in {"sailor_multitask_8head", "vistral_multitask_8head"}]
for family, seed, overflow, rev, checkpoint, metric_sha, aivivn, vsfc, vsmec, ordinary in PACKS:
    raw.append({
        "actual_recipe": family, "run_id": f"q1b_{family}_{seed}", "seed": seed,
        "aivivn_macro_f1": str(aivivn), "vsfc_macro_f1": str(vsfc),
        "vsmec_macro_f1": str(vsmec), "ordinary_f1": str(ordinary),
        "remote_repo": f"Thundergod2007/vipragsent-experiment-artifacts-overflow-{overflow}",
        "remote_ref": f"hub:{rev};path:{BASE}/q1b_{family}_{seed}",
        "artifact_sha256": checkpoint, "metric_file_sha256": metric_sha,
        "status": "ANALYZED_EXTERNAL_PACK",
    })
write(raw_path, raw_fields, raw)

prov_path = ROOT / "q1b_provenance.csv"
prov_fields = list(csv.DictReader(prov_path.open(encoding="utf-8-sig", newline="")).fieldnames)
prov = [r for r in read(prov_path) if r["canonical_row"] not in {"Sailor-7B SFT", "Vistral-7B SFT"}]
for family, seed, overflow, rev, checkpoint, metric_sha, aivivn, vsfc, vsmec, ordinary in PACKS:
    display = "Sailor-7B SFT" if family.startswith("sailor") else "Vistral-7B SFT"
    model = "sail/Sailor-7B" if family.startswith("sailor") else "Viet-Mistral/Vistral-7B-Chat"
    prov.append({
        "canonical_row": display, "run_id": f"q1b_{family}_{seed}", "seed": seed,
        "remote_repo": f"Thundergod2007/vipragsent-experiment-artifacts-overflow-{overflow}",
        "remote_ref": f"hub:{rev};path:{BASE}/q1b_{family}_{seed}",
        "model_repository": model, "model_revision": "NOT_RECORDED",
        "code_commit": "NOT_RECORDED", "data_fingerprint": "NOT_RECORDED",
        "external_manifest_hash": "CF1B5504C898D0459A630250806089B5819F8B7A9B4D4D33943002DACBB988B5",
        "aivivn_macro_f1": str(aivivn), "vsfc_macro_f1": str(vsfc),
        "vsmec_macro_f1": str(vsmec), "ordinary_f1": str(ordinary),
        "aivivn_gold_hash": GOLD["aivivn"], "vsfc_gold_hash": GOLD["vsfc"],
        "vsmec_gold_hash": GOLD["vsmec"], "artifact_sha256": checkpoint,
        "metric_file_sha256": metric_sha, "status": "ANALYZED_EXTERNAL_PACK",
    })
write(prov_path, prov_fields, prov)

canonical_path = ROOT / "q1b_canonical_table.csv"
canonical_fields = list(csv.DictReader(canonical_path.open(encoding="utf-8-sig", newline="")).fieldnames)
canonical = [r for r in read(canonical_path) if r["canonical_row"] not in {"Sailor-7B SFT", "Vistral-7B SFT"}]
for family, label, ref in [("sailor_multitask_8head", "Sailor-7B SFT", "Q1b-04"), ("vistral_multitask_8head", "Vistral-7B SFT", "Q1b-05")]:
    vals = [(x[6], x[7], x[8], x[9]) for x in PACKS if x[0] == family]
    means = [statistics.mean(v[i] for v in vals) for i in range(4)]
    sds = [statistics.stdev(v[i] for v in vals) for i in range(4)]
    canonical.append({
        "canonical_row": label, "reference_row": ref, "actual_recipe": family,
        "seed_coverage": "20260521|20260522|20260523",
        "aivivn_macro_f1_mean": f"{means[0]:.10f}", "aivivn_macro_f1_sd": f"{sds[0]:.10f}",
        "vsfc_macro_f1_mean": f"{means[1]:.10f}", "vsfc_macro_f1_sd": f"{sds[1]:.10f}",
        "vsmec_macro_f1_mean": f"{means[2]:.10f}", "vsmec_macro_f1_sd": f"{sds[2]:.10f}",
        "ordinary_f1_mean": f"{means[3]:.10f}", "ordinary_f1_sd": f"{sds[3]:.10f}",
        "n": "3", "status": "ANALYZED_SEMANTIC_COHORT_VERIFIED",
        "identity_or_scope_note": "actual saved recipe; external pack explicitly routes AIVIVN/VSFC to polarity and VSMEC to emotion; not ViPragSent primary",
    })
for r in canonical:
    if r["canonical_row"] == "ViPragSent XLM-R-large primary":
        r["status"] = "ANALYZED_SEMANTIC_COHORT_VERIFIED"
        r["identity_or_scope_note"] = "primary ViPragSent XLM-R-large; observed ID/gold/raw-text/NFC parity passes against standard baseline; AIVIVN manifest hash discrepancy retained as provenance limitation"
if not any(r["reference_row"] == "Q1b-07" for r in canonical):
    canonical.append({
        "canonical_row": "GPT-4o-mini 8-shot", "reference_row": "Q1b-07", "actual_recipe": "NOT_FOUND",
        "seed_coverage": "NO_SEED", "aivivn_macro_f1_mean": "", "aivivn_macro_f1_sd": "",
        "vsfc_macro_f1_mean": "", "vsfc_macro_f1_sd": "", "vsmec_macro_f1_mean": "", "vsmec_macro_f1_sd": "",
        "ordinary_f1_mean": "", "ordinary_f1_sd": "", "n": "0", "status": "UNRESOLVED",
        "identity_or_scope_note": "No completed comparable Q1b 8-shot pack confirmed; do not substitute Azure deployment gpt-4.1-mini",
    })
write(canonical_path, canonical_fields, canonical)

# Rewrite the figure source from canonical values, retaining an explicit
# unresolved GPT-4o-mini 8-shot row and the contextual PhoBERT inventory row.
fig_path = PAPER / "figures" / "source_data" / "q1b_expanded_working_draft.csv"
fig_fields = list(csv.DictReader(fig_path.open(encoding="utf-8-sig", newline="")).fieldnames)
fig = []
row_map = {
    "ViPragSent XLM-R-large primary": ("ViPragSent XLM-R-large", "xlmr_followup_full", "FacebookAI/xlm-roberta-large", "primary", "ANALYZED_SEMANTIC_COHORT_VERIFIED", "one primary ViPragSent XLM-R-large row; semantic record parity verified versus standard XLM-R; AIVIVN manifest hash discrepancy is provenance-only", "revision_evidence/q1b_canonical_table.csv; revision_evidence/q1b_split_gold_comparability.md; revision_evidence/q1b_provenance.csv"),
    "ViPragSent full PhoBERT inventory": ("ViPragSent full PhoBERT (contextual inventory)", "vipragsent_full_phobert", "PhoBERT-base", "contextual_variant", "ANALYZED_CONTEXTUAL_VARIANT", "inventory only; not the primary row and not a reference-row substitute", "revision_evidence/q1b_canonical_table.csv"),
    "PhoBERT single-task routed": ("PhoBERT single-task", "phobert_pol_single + phobert_emo_single", "PhoBERT-base", "baseline", "ANALYZED", "same-seed composite; common F6F6A78E/C27B61C7/1D2E604B external hashes", "revision_evidence/q1b_canonical_table.csv"),
    "PhoBERT multitask 8-head": ("PhoBERT fine-tune (actual multitask recipe)", "phobert_multitask_8head", "PhoBERT-base", "baseline", "ANALYZED", "actual recipe retained; common external hashes; not silently relabelled", "revision_evidence/q1b_canonical_table.csv"),
    "XLM-R-large multitask 8-head": ("XLM-R-large standard classification baseline", "xlmr_multitask_8head", "XLM-R-large", "baseline", "ANALYZED_SEMANTIC_COHORT_VERIFIED", "standard classification baseline; observed same IDs/gold/text/NFC versus primary across 3 seeds", "revision_evidence/q1b_canonical_table.csv; revision_evidence/q1b_split_gold_comparability.md"),
    "Sailor-7B SFT": ("Sailor-7B SFT", "sailor_multitask_8head", "sail/Sailor-7B", "baseline", "ANALYZED_SEMANTIC_COHORT_VERIFIED", "three completed external packs; source model revision/code commit not recorded in manifest", "revision_evidence/q1b_canonical_table.csv; revision_evidence/q1b_provenance.csv"),
    "Vistral-7B SFT": ("Vistral-7B SFT", "vistral_multitask_8head", "Viet-Mistral/Vistral-7B-Chat", "baseline", "ANALYZED_SEMANTIC_COHORT_VERIFIED", "three completed external packs; source model revision/code commit not recorded in manifest", "revision_evidence/q1b_canonical_table.csv; revision_evidence/q1b_provenance.csv"),
    "GPT artifact (configured deployment)": ("Azure deployment gpt-4.1-mini (version unrecorded)", "azure_gpt41_mini dedicated_prompts", "Azure deployment gpt-4.1-mini", "baseline", "ANALYZED_IDENTITY_MISMATCH", "metric-ready; not reference GPT-4o-mini", "revision_evidence/q1b_canonical_table.csv; revision_evidence/q1b_azure_deployment_manifest.json"),
}
for r in canonical:
    key = r["canonical_row"]
    if key not in row_map:
        continue
    model, recipe, backbone, scope, status, note, source = row_map[key]
    def pct(value):
        return "—" if value in ("", "NOT_APPLICABLE") else f"{100*float(value):.6f}"
    fig.append({
        "model": model, "actual_recipe": recipe, "backbone": backbone, "scope": scope,
        "record_status": status, "source": source, "seed_coverage": r["seed_coverage"],
        "seeds": r["seed_coverage"].replace("|", ","), "n": r["n"],
        "identity_or_scope_note": note, "cohort_comparability_note": "same external test hashes and explicit label routing; primary AIVIVN source-manifest byte hash differs but observed record parity passes" if key == "ViPragSent XLM-R-large primary" else "same external test hashes and explicit label routing; no training-time claim added",
        "UIT_VSFC_mean": pct(r["vsfc_macro_f1_mean"]), "UIT_VSFC_sd": pct(r["vsfc_macro_f1_sd"]),
        "UIT_VSMEC_mean": pct(r["vsmec_macro_f1_mean"]), "UIT_VSMEC_sd": pct(r["vsmec_macro_f1_sd"]),
        "AIVIVN_mean": pct(r["aivivn_macro_f1_mean"]), "AIVIVN_sd": pct(r["aivivn_macro_f1_sd"]),
        "ordinary_f1_mean": pct(r["ordinary_f1_mean"]), "ordinary_f1_sd": pct(r["ordinary_f1_sd"]),
    })
fig.append({
    "model": "GPT-4o-mini 8-shot (reference identity)", "actual_recipe": "NOT_FOUND", "backbone": "GPT-4o-mini", "scope": "baseline", "record_status": "UNRESOLVED", "source": "revision_evidence/q1a_q1b_crosswalk.csv", "seed_coverage": "—", "seeds": "—", "n": "—", "identity_or_scope_note": "No completed comparable Q1b 8-shot pack confirmed; do not substitute gpt-4.1-mini", "cohort_comparability_note": "—", "UIT_VSFC_mean": "—", "UIT_VSFC_sd": "—", "UIT_VSMEC_mean": "—", "UIT_VSMEC_sd": "—", "AIVIVN_mean": "—", "AIVIVN_sd": "—", "ordinary_f1_mean": "—", "ordinary_f1_sd": "—"
})
write(fig_path, fig_fields, fig)

cross_path = ROOT / "q1a_q1b_crosswalk.csv"
cross = read(cross_path)
cross_fields = list(csv.DictReader(cross_path.open(encoding="utf-8-sig", newline="")).fieldnames)
for row in cross:
    if row["reference_row"] == "Q1b-03":
        row["evidence_state"] = "ANALYZED_SEMANTIC_COHORT_VERIFIED"
        row["identity_note"] = "standard XLM-R classification baseline; observed same IDs/gold/raw text/NFC against primary across all 3 seeds; do not conflate with primary"
        row["source_note"] = "HF overflow-002 actual science-result tar packs; q1b_split_gold_parity.py/log"
    elif row["reference_row"] == "Q1b-04":
        row["evidence_state"] = "ANALYZED_SEMANTIC_COHORT_VERIFIED"
        row["seed_coverage"] = "20260521|20260522|20260523"
        row["identity_note"] = "actual sailor_multitask_8head recipe; Sailor-7B SFT comparison, not primary; external label routing explicit"
        row["source_note"] = "HF overflow-014/010/004 pinned metrics and external manifests; q1b_provenance.csv"
    elif row["reference_row"] == "Q1b-05":
        row["evidence_state"] = "ANALYZED_SEMANTIC_COHORT_VERIFIED"
        row["seed_coverage"] = "20260521|20260522|20260523"
        row["identity_note"] = "actual vistral_multitask_8head recipe; Vistral-7B SFT comparison, not primary; external label routing explicit"
        row["source_note"] = "HF overflow-011/012 pinned metrics and external manifests; q1b_provenance.csv"
    elif row["reference_row"] == "Q1b-08":
        row["evidence_state"] = "ANALYZED_SEMANTIC_COHORT_VERIFIED"
        row["identity_note"] = "one primary ViPragSent XLM-R-large row; observed semantic cohort parity passes; AIVIVN manifest hash difference retained as provenance limitation"
        row["source_note"] = "HF overflow-018/025/017 pinned predictions, q1b_split_gold_comparability.md/log, q1b_provenance.csv"
write(cross_path, cross_fields, cross)

print("q1b_expanded_rows_written csvwriter")
print("sailor_and_vistral_per_seed_rows=6")
print("canonical_rows_written=", len(canonical))
print("figure_rows_written=", len(fig))
