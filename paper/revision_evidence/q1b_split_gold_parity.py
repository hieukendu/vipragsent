"""Reproduce Q1b same-cohort parity and macro-F1 from pinned Hub packs.

The standard XLM-R baseline is stored as a science-result tar pack, so this
script discovers the actual tar filename through the pinned Hub tree before
extracting it. Primary prediction JSONL files are fetched at pinned Hub
revisions. HTTP payloads are cached and hashed; no training or Hub writes occur.
"""
import csv
import hashlib
import io
import json
import math
import tarfile
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "q1b_http_cache"
CACHE.mkdir(exist_ok=True)
SEEDS = ("20260521", "20260522", "20260523")
DATASETS = {
    "aivivn": ("aivivn_test_predictions.jsonl", ("negative", "neutral", "positive"), 1609),
    "vsfc": ("uit_vsfc_test_predictions.jsonl", ("negative", "neutral", "positive"), 3166),
    "vsmec": ("uit_vsmec_test_predictions.jsonl", ("anger", "disgust", "enjoyment", "fear", "other", "sadness", "surprise"), 693),
}
BASE_REPO = "Thundergod2007/vipragsent-experiment-artifacts-overflow-002"
BASE_REV = {
    "20260521": "c40767d878d528fc7365b99d227b347e3edbabeb",
    "20260522": "52c711af58dc9d00a6f54916b4515f04d5aac083",
    "20260523": "972055fe673a7d48db0d416da9771e129c9c5e47",
}
BASE_DIR = "campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT/nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/science/final"
PRIMARY_REFS = {
    "20260521": ("vipragsent-experiment-artifacts-overflow-018", "cd0b0367ae24bfa5fac24316f5e153f01cd67f11"),
    "20260522": ("vipragsent-experiment-artifacts-overflow-025", "7810865b6c9162fbb01923dfb45656d5d49da60a"),
    "20260523": ("vipragsent-experiment-artifacts-overflow-017", "43a4e35edc969630fe33704a80fffea110f7f0a2"),
}
PAYLOADS = []
LOG = []


def repo_id(repo):
    return repo if "/" in repo else f"Thundergod2007/{repo}"


def url(repo, revision, path):
    return f"https://huggingface.co/{repo_id(repo)}/resolve/{revision}/{path}"


def cached_get(repo, revision, path, tag):
    cache_path = CACHE / f"{tag}.bin"
    endpoint = url(repo, revision, path)
    if cache_path.exists():
        payload = cache_path.read_bytes()
        status = "CACHE"
    else:
        response = requests.get(endpoint, timeout=180)
        response.raise_for_status()
        payload = response.content
        cache_path.write_bytes(payload)
        status = str(response.status_code)
    sha = hashlib.sha256(payload).hexdigest().upper()
    PAYLOADS.append((tag, status, len(payload), sha, endpoint, str(cache_path.relative_to(ROOT))))
    return payload


def tree(repo, revision, path):
    endpoint = f"https://huggingface.co/api/models/{repo_id(repo)}/tree/{revision}/{path}?recursive=true"
    response = requests.get(endpoint, timeout=120)
    response.raise_for_status()
    return response.json()


def load_jsonl(payload):
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def load_baseline(seed, filename):
    pack_dir = f"{BASE_DIR}/q1b_xlmr_multitask_8head_{seed}"
    entries = tree(BASE_REPO, BASE_REV[seed], pack_dir)
    tar_paths = [x["path"] for x in entries if x.get("type") == "file" and x["path"].endswith(".tar.gz")]
    if len(tar_paths) != 1:
        raise SystemExit(f"expected one actual baseline tar for {seed}, found {tar_paths}")
    payload = cached_get(BASE_REPO, BASE_REV[seed], tar_paths[0], f"baseline_{seed}_tar")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(filename)]
        if len(members) != 1:
            raise SystemExit(f"expected one {filename} in actual baseline tar, found {[m.name for m in members[:10]]}")
        return load_jsonl(archive.extractfile(members[0]).read())


def load_primary(seed, filename):
    repo, revision = PRIMARY_REFS[seed]
    path = f"campaigns/vipragsent-xlmr-q1b-q4-followup-v1/xlmr_followup_q1b_vipragsent_full_{seed}/predictions/{filename}"
    tag = f"primary_{seed}_{filename.replace('.jsonl', '')}"
    return load_jsonl(cached_get(repo, revision, path, tag))


def digest(records, field, normalize=False):
    values = []
    for record in records:
        value = str(record[field])
        values.append(unicodedata.normalize("NFC", value) if normalize else value)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest().upper()


def macro_f1(records, labels):
    scores = []
    for label in labels:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in records)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in records)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in records)
        scores.append(0.0 if 2 * tp + fp + fn == 0 else (2.0 * tp) / (2 * tp + fp + fn))
    return sum(scores) / len(labels)


with (ROOT / "q1b_raw_extraction.csv").open(encoding="utf-8-sig", newline="") as fh:
    stored = {(r["run_id"], dataset): float(r[f"{dataset}_macro_f1"])
              for r in csv.DictReader(fh) if r["actual_recipe"] in {"xlmr_multitask_8head", "xlmr_followup_full"}
              for dataset in DATASETS}

for seed in SEEDS:
    for dataset, (filename, labels, expected_n) in DATASETS.items():
        baseline = load_baseline(seed, filename)
        primary = load_primary(seed, filename)
        for name, records in (("baseline", baseline), ("primary", primary)):
            ids = [r.get("sample_id") for r in records]
            if len(records) != expected_n or any(not x for x in ids) or len(set(ids)) != expected_n:
                raise SystemExit(f"record count/ID failure {seed} {dataset} {name}: n={len(records)} unique={len(set(ids))}")
            observed = {value for r in records for value in (r.get("gold"), r.get("prediction"))}
            if not observed.issubset(set(labels)):
                raise SystemExit(f"label vocabulary failure {seed} {dataset} {name}: {sorted(observed - set(labels))}")
        for left, right in zip(baseline, primary):
            for field in ("sample_id", "gold", "text"):
                if left[field] != right[field]:
                    raise SystemExit(f"record mismatch {seed} {dataset} {field} {left['sample_id']}")
            if unicodedata.normalize("NFC", left["text"]) != unicodedata.normalize("NFC", right["text"]):
                raise SystemExit(f"NFC mismatch {seed} {dataset} {left['sample_id']}")
        b_f1, p_f1 = macro_f1(baseline, labels), macro_f1(primary, labels)
        run_b, run_p = f"q1b_xlmr_multitask_8head_{seed}", f"q1b_xlmr_followup_{seed}"
        for run_id, value in ((run_b, b_f1), (run_p, p_f1)):
            if not math.isclose(value, stored[(run_id, dataset)], rel_tol=0, abs_tol=5e-11):
                raise SystemExit(f"stored metric mismatch {run_id} {dataset}: {value} != {stored[(run_id, dataset)]}")
        pred_diffs = sum(x["prediction"] != y["prediction"] for x, y in zip(baseline, primary))
        line = f"seed={seed} dataset={dataset} n={expected_n} unique_ids=PASS gold=PASS raw_text=PASS nfc_text=PASS labels={','.join(labels)} prediction_diffs={pred_diffs} baseline_f1={b_f1:.15f} primary_f1={p_f1:.15f} stored=PASS"
        print(line)
        LOG.append(line)
        hash_line = f"  ids={digest(baseline, 'sample_id')} gold={digest(baseline, 'gold')} nfc_text={digest(baseline, 'text', normalize=True)}"
        print(hash_line)
        LOG.append(hash_line)
        if tuple(digest(baseline, field, normalize=(field == "text")) for field in ("sample_id", "gold", "text")) != tuple(digest(primary, field, normalize=(field == "text")) for field in ("sample_id", "gold", "text")):
            raise SystemExit(f"semantic hash mismatch {seed} {dataset}")

print("semantic_cohort_parity=PASS")
print("metric_recomputation_vs_saved_per_seed=PASS tolerance=5e-11")
print("manifest_aivivn_hash_difference=PROVENANCE_LIMITATION_NOT_SEMANTIC_COHORT_DIFFERENCE")
LOG.extend(["semantic_cohort_parity=PASS", "metric_recomputation_vs_saved_per_seed=PASS tolerance=5e-11", "manifest_aivivn_hash_difference=PROVENANCE_LIMITATION_NOT_SEMANTIC_COHORT_DIFFERENCE"])
payload_by_tag = {row[0]: row for row in PAYLOADS}
with (ROOT / "q1b_http_payload_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["tag", "status", "bytes", "payload_sha256", "url", "cache_path"])
    writer.writerows(payload_by_tag.values())
(ROOT / "q1b_split_gold_parity.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")

# Promote the full HTTP payload hashes into the baseline per-seed provenance
# rows. This resolves the earlier 63-character source filename field without
# treating the filename itself as a SHA-256 value.
baseline_payload_sha = {tag.split("_")[1]: row[3] for tag, row in payload_by_tag.items() if tag.startswith("baseline_")}
for filename in (ROOT / "q1b_raw_extraction.csv", ROOT / "q1b_provenance.csv"):
    with filename.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    key = "run_id"
    for row in rows:
        if row[key].startswith("q1b_xlmr_multitask_8head_"):
            seed = row[key].rsplit("_", 1)[-1]
            row["artifact_sha256"] = baseline_payload_sha[seed]
            row["status"] = "ANALYZED"
    with filename.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
