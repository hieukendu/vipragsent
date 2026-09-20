"""Bounded Q1a extra-artifact audit.

This is an artifact audit, not a training or result-generation script.  It uses
the pinned Hugging Face model revision returned by the API, fetches only saved
manifests/metrics/predictions, and writes every output below revision_q1a_extra.
The six existing local rows are recomputed from their saved JSONL predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "revision_q1a_extra"
RAW = OUT / "raw_sources"
HF = "https://huggingface.co"
HEADS = ["implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking"]
SEEDS = [21, 22, 23]
TOL = 1e-12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_url(url: str, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "q1a-extra-audit/1.0"})
            with urlopen(req, timeout=45) as r:
                return int(r.status), r.read(), dict(r.headers)
        except (HTTPError, URLError, TimeoutError) as e:
            last = str(e)
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    return 0, b"", {"error": last or "unknown"}


def model_api(repo: str):
    status, data, headers = get_url(f"{HF}/api/models/{repo}")
    if status != 200:
        return {"ok": False, "status": status, "error": headers.get("error", "")}
    obj = json.loads(data.decode("utf-8"))
    return {"ok": True, "status": status, "sha": obj.get("sha"), "model": obj.get("modelId", repo)}


def tree_api(repo: str, revision: str, path: str = ""):
    """Use the actual /api/models tree endpoint, not resolve/raw directory URLs."""
    rows = []
    cursor = None
    pages = 0
    while True:
        params = {"recursive": "true", "expand": "false", "limit": "1000"}
        if path:
            params["path"] = path
        if cursor:
            params["cursor"] = cursor
        url = f"{HF}/api/models/{repo}/tree/{revision}" + (f"/{quote(path, safe='/')}" if path else "") + "?" + urlencode(params)
        status, data, headers = get_url(url)
        pages += 1
        if status != 200:
            return {"ok": False, "repo": repo, "revision": revision, "path": path, "status": status,
                    "pages": pages, "entries": rows, "error": headers.get("error", "")}
        page = json.loads(data.decode("utf-8"))
        if isinstance(page, list):
            rows.extend(page)
        link = headers.get("Link", "")
        m = re.search(r"[?&]cursor=([^>;]+)", link)
        if not m:
            break
        cursor = m.group(1)
        if pages > 100:
            break
    return {"ok": True, "repo": repo, "revision": revision, "path": path, "status": 200,
            "pages": pages, "entries": rows}


def raw_fetch(repo: str, revision: str, path: str, label: str, required: bool = True):
    # Record the full repo/revision/path in the returned provenance, but keep
    # the Windows cache filename short enough for record-level HF paths.
    token = sha256_bytes(f"{repo}@{revision}:{path}".encode("utf-8"))[:20]
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:70].strip("_")
    safe = f"{safe_label}__{token}.bin"
    local = RAW / safe
    url = f"{HF}/{repo}/resolve/{revision}/{quote(path, safe='/')}?download=true"
    if local.exists():
        data = local.read_bytes()
        return {"ok": True, "cached": True, "status": 200, "repo": repo, "revision": revision,
                "path": path, "url": url, "local_path": str(local.relative_to(OUT)),
                "bytes": len(data), "sha256": sha256_bytes(data), "required": required}
    status, data, headers = get_url(url)
    rec = {"ok": status == 200, "cached": False, "status": status, "repo": repo, "revision": revision,
           "path": path, "url": url, "required": required}
    if status == 200:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        rec.update({"local_path": str(local.relative_to(OUT)), "bytes": len(data), "sha256": sha256_bytes(data)})
    else:
        rec["error"] = headers.get("error", "http failure")
    return rec


def load_local_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def f1_binary(gold, pred):
    vals = []
    for cls in (0, 1):
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(vals) / 2.0


def recompute(rows):
    per = {}
    for head in HEADS:
        gold = [int(r["gold"][head]) for r in rows]
        pred = [int(r["predictions"][head]) for r in rows]
        per[head] = f1_binary(gold, pred)
    return {"per_label_f1": per, "macro_f1": sum(per.values()) / len(HEADS), "prediction_count": len(rows)}


def ids_gold(rows):
    return {str(r.get("sample_id")): tuple(int(r["gold"][h]) for h in HEADS) for r in rows}


def cohort_rows():
    specs = [
        ("vipragsent_xlmr_large", "primary_vipragsent_xlmr_large", "target"),
        ("phobert_single_task", "phobert_single_task", "phobert_single"),
        ("phobert_finetune", "phobert_finetune", "phobert_finetune"),
        ("xlmr_large_baseline", "xlmr_large_baseline", "xlmr_baseline"),
        ("sailor_7b_sft", "sailor_7b_sft", "sailor"),
        ("vistral_7b_sft", "vistral_7b_sft", "vistral"),
    ]
    out = []
    for variant, label, stem in specs:
        for seed in SEEDS:
            pred_path = ROOT / "paper" / "raw_fairness" / f"{stem}_{seed}.jsonl"
            metric_path = ROOT / "paper" / "raw_fairness" / f"{stem}_{seed}_metrics.json"
            rows = load_local_jsonl(pred_path)
            comp = recompute(rows)
            recorded = json.loads(metric_path.read_text(encoding="utf-8"))
            rec_macro = recorded.get("macro_pragmatic_f1", recorded.get("selection_metric"))
            deltas = {h: comp["per_label_f1"][h] - recorded.get("per_label_f1", {}).get(h, math.nan) for h in HEADS}
            deltas["macro_f1"] = comp["macro_f1"] - rec_macro if rec_macro is not None else math.nan
            out.append({"row_id": f"{variant}__seed{seed}", "variant": variant, "display_name": label,
                        "seed": seed, "backbone": "xlm-roberta-large" if "xlmr" in variant else label,
                        "model_identity": label, "status": "VERIFIED_LOCAL_RECOMPUTED", "split": "test",
                        "prediction_count": len(rows), "per_label_f1": comp["per_label_f1"], "macro_f1": comp["macro_f1"],
                        "recorded_macro_f1": rec_macro, "recompute_deltas": deltas,
                        "exact_recorded_match": bool(rec_macro is not None and abs(deltas["macro_f1"]) <= TOL and all(abs(deltas[h]) <= TOL for h in HEADS)),
                        "id_gold_source": str(pred_path.relative_to(ROOT)), "prediction_sha256": sha256_file(pred_path),
                        "metric_source": str(metric_path.relative_to(ROOT)), "metric_sha256": sha256_file(metric_path),
                        "cohort_match": "self_reference"})
    return out


def find_file(entries, names):
    names = set(names)
    return sorted([e["path"] for e in entries if e.get("type") == "file" and Path(e.get("path", "")).name in names])


def first_file(entries, suffixes):
    for suffix in suffixes:
        vals = sorted([e["path"] for e in entries if e.get("type") == "file" and e.get("path", "").endswith(suffix)])
        if vals:
            return vals[0]
    return None


def manifest_source(repo, revision, root, label):
    t = tree_api(repo, revision, root)
    files = t.get("entries", [])
    selected = []
    for p in find_file(files, {"config_snapshot.yaml", "run_manifest.json", "SCIENCE_UNIT_MANIFEST.json", "review_summary.json", "approval_status.json", "metrics.json", "test_metrics.json", "test_reasoning_metrics.json"}):
        selected.append(raw_fetch(repo, revision, p, label, required=False))
    return t, selected


def read_json_cached(source):
    if not source.get("ok") or not source.get("local_path"):
        return None
    try:
        return json.loads((OUT / source["local_path"]).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # YAML/configuration and JSONL prediction payloads are still retained as
        # raw sources; they are not silently treated as JSON metadata.
        return None


def recorded_metric(source_records):
    preferred = [x for x in source_records if x.get("ok") and (x["path"].endswith("test_metrics.json") or x["path"].endswith("test_reasoning_metrics.json"))]
    preferred += [x for x in source_records if x.get("ok") and x["path"].endswith("metrics.json")]
    for src in preferred:
        obj = read_json_cached(src)
        if isinstance(obj, dict):
            for key in ("macro_pragmatic_f1", "primary_macro_f1", "selection_metric"):
                if isinstance(obj.get(key), (int, float)):
                    return obj, key, src
    return None, None, None


def direct_prediction_source(repo, revision, pred_path, label):
    src = raw_fetch(repo, revision, pred_path, label, required=True)
    if not src.get("ok"):
        return src, [], "download_failed"
    rows = load_local_jsonl(OUT / src["local_path"])
    return src, rows, "complete" if len(rows) == 2000 else f"row_count_{len(rows)}"


def record_prediction_source(record_sources, record_dir, label):
    all_records = {}
    tree_evidence = []
    for repo, revision in record_sources:
        t = tree_api(repo, revision, record_dir)
        tree_evidence.append(t)
        for e in t.get("entries", []):
            p = e.get("path", "")
            if e.get("type") == "file" and re.search(r"/\d{8}\.json$", p) and ".snapshots/" not in p:
                idx = int(Path(p).stem)
                all_records.setdefault(idx, (repo, revision, p))
    missing = [i for i in range(2000) if i not in all_records]
    duplicates = []
    # Only one source is fetched for an index; duplicates are retained in tree evidence.
    rows_by_idx = {}
    failures = []
    def fetch_one(item):
        idx, (repo, revision, path) = item
        return idx, raw_fetch(repo, revision, path, f"{label}__record_{idx:08d}", required=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(fetch_one, item) for item in sorted(all_records.items())]
        for fut in as_completed(futures):
            idx, src = fut.result()
            if src.get("ok"):
                try:
                    rows_by_idx[idx] = json.loads((OUT / src["local_path"]).read_text(encoding="utf-8"))
                except Exception as e:
                    failures.append({"index": idx, "error": f"parse:{e}", "source": src})
            else:
                failures.append({"index": idx, "error": "download", "source": src})
    rows = [rows_by_idx[i] for i in range(2000) if i in rows_by_idx]
    state = "complete" if len(rows) == 2000 and not missing and not failures else "incomplete_records"
    return rows, {"tree_evidence": tree_evidence, "record_count": len(all_records), "missing_indexes": missing,
                  "duplicate_indexes": duplicates, "download_failures": failures, "status": state}


def normalize_identity(objects, default):
    text = json.dumps(objects, ensure_ascii=False).lower()
    identity = default
    if "gpt-4.1-mini" in text or "gpt41_mini" in text:
        identity = "GPT-4.1-mini"
    return identity


def extra_specs():
    base = "campaigns/vipragsent-v7-6693e7e9eef08ef1/SHARD_GPU40_7B_INTEGRATION"
    return [
        {"variant": "vipragsent_no_auxiliary_vistral", "seed": s, "root": f"{base}/q1a_vipragsent_no_auxiliary_vistral_202605{s}/science", "repo": "Thundergod2007/vipragsent-experiment-artifacts", "kind": "direct"} for s in SEEDS
    ] + [
        {"variant": "cot_only_vistral", "seed": s, "root": f"{base}/q1a_cot_only_vistral_202605{s}/science", "repo": "Thundergod2007/vipragsent-experiment-artifacts", "kind": "direct"} for s in SEEDS
    ] + [
        {"variant": "explanation_only_vistral", "seed": s, "root": f"{base}/q1a_explanation_only_vistral_202605{s}", "repo": "Thundergod2007/vipragsent-experiment-artifacts-overflow-009", "kind": "split_explanation"} for s in SEEDS
    ] + [
        {"variant": "vipragsent_full_vistral", "seed": s, "root": f"{base}/q1a_vipragsent_full_vistral_202605{s}", "repo": "Thundergod2007/vipragsent-experiment-artifacts-overflow-009", "kind": "full_records"} for s in SEEDS
    ] + [
        {"variant": "gpt_zero_shot", "seed": None, "root": "live_runs/q1a_azure_gpt41_mini_zeroshot", "repo": "Thundergod2007/vipragsent-experiment-artifacts-overflow-022", "kind": "gpt"},
        {"variant": "gpt_8_shot", "seed": None, "root": "campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT/nb-170d4c47-8a39-47ad-9425-d231de44fc92-857677f648-2qn8f/science/runs/q1a_azure_gpt41_mini_8shot/45C4769EFDBF", "repo": "Thundergod2007/vipragsent-experiment-artifacts", "kind": "gpt"},
    ]


def audit_extra(spec):
    repo_info = model_api(spec["repo"])
    revision = repo_info.get("sha")
    rec = {"variant": spec["variant"], "seed": spec["seed"], "repo": spec["repo"], "root": spec["root"],
           "repo_api": repo_info, "revision": revision, "kind": spec["kind"], "source_records": [], "status": "UNVERIFIED"}
    if not revision:
        rec["status"] = "UNVERIFIED/NOT_AVAILABLE"
        rec["reason"] = "repository revision could not be resolved"
        return rec
    tree, sources = manifest_source(spec["repo"], revision, spec["root"], spec["variant"] + (f"__seed{spec['seed']}" if spec["seed"] else ""))
    rec["tree"] = {k: v for k, v in tree.items() if k != "entries"}
    rec["tree_entry_count"] = len(tree.get("entries", []))
    rec["source_records"].extend(sources)
    rec["identity"] = normalize_identity([read_json_cached(x) for x in sources if x.get("ok")], "Vistral-7B / saved run identity")
    metric_obj, metric_key, metric_src = recorded_metric(sources)
    rec["recorded_metric_source"] = metric_src
    rec["recorded_metric_key"] = metric_key
    rec["recorded_metric"] = metric_obj.get(metric_key) if metric_obj and metric_key else None
    entries = tree.get("entries", [])
    pred_paths = find_file(entries, {"test_predictions.jsonl"})
    rows = []
    pred_source = None
    if spec["kind"] in ("direct", "gpt") and pred_paths:
        pred_source, rows, pred_status = direct_prediction_source(spec["repo"], revision, pred_paths[0], spec["variant"] + (f"__seed{spec['seed']}" if spec["seed"] else ""))
        rec["source_records"].append(pred_source)
        rec["prediction_status"] = pred_status
    elif spec["kind"] == "split_explanation":
        # Known direct artifact locations are selected from the live tree, never invented.
        roots = {
            21: [("Thundergod2007/vipragsent-experiment-artifacts-overflow-025", "7810865b6c9162fbb01923dfb45656d5d49da60a"), ("Thundergod2007/vipragsent-experiment-artifacts-overflow-003", "70c04faa6e19b65d2caaea08d6d1b0a8dbcf9311")],
            22: [("Thundergod2007/vipragsent-experiment-artifacts-overflow-003", None), ("Thundergod2007/vipragsent-experiment-artifacts-overflow-004", None)],
            23: [("Thundergod2007/vipragsent-experiment-artifacts-overflow-003", None), ("Thundergod2007/vipragsent-experiment-artifacts-overflow-004", None)],
        }[spec["seed"]]
        for rr, pinned in roots:
            ri = {"sha": pinned} if pinned else model_api(rr)
            rv = ri.get("sha")
            if not rv:
                continue
            tt = tree_api(rr, rv, spec["root"])
            for pp in find_file(tt.get("entries", []), {"test_predictions.jsonl"}):
                ps, rows, pred_status = direct_prediction_source(rr, rv, pp, spec["variant"] + f"__seed{spec['seed']}")
                rec["source_records"].append(ps)
                rec.setdefault("prediction_candidates", []).append({"repo": rr, "revision": rv, "path": pp, "status": pred_status})
                if len(rows) == 2000:
                    pred_source = ps
                    break
            if len(rows) == 2000:
                break
    elif spec["kind"] == "full_records":
        record_repos = [("Thundergod2007/vipragsent-experiment-artifacts-overflow-009", model_api("Thundergod2007/vipragsent-experiment-artifacts-overflow-009").get("sha")),
                        ("Thundergod2007/vipragsent-experiment-artifacts-overflow-017", model_api("Thundergod2007/vipragsent-experiment-artifacts-overflow-017").get("sha")),
                        ("Thundergod2007/vipragsent-experiment-artifacts-overflow-016", model_api("Thundergod2007/vipragsent-experiment-artifacts-overflow-016").get("sha"))]
        record_repos = [(r, v) for r, v in record_repos if v]
        record_dir = spec["root"] + "/predictions/test_predictions.jsonl.records"
        rows, record_evidence = record_prediction_source(record_repos, record_dir, spec["variant"] + f"__seed{spec['seed']}")
        rec["record_prediction_evidence"] = record_evidence
        rec["prediction_status"] = record_evidence["status"]
    if not rows or len(rows) != 2000:
        rec["status"] = "UNVERIFIED/NOT_AVAILABLE"
        rec["reason"] = "no complete saved test six-head prediction artifact; inventory/preflight/partial dev output is not a Q1a result"
        rec["observed_prediction_count"] = len(rows)
        return rec
    comp = recompute(rows)
    rec["prediction_count"] = len(rows)
    rec["prediction_sha256"] = pred_source.get("sha256") if pred_source else None
    rec["recomputed"] = comp
    rec["status"] = "ANALYZED_RECOMPUTED"
    if rec.get("recorded_metric") is not None:
        rec["recompute_delta_macro"] = comp["macro_f1"] - float(rec["recorded_metric"])
        rec["exact_recorded_match"] = abs(rec["recompute_delta_macro"]) <= TOL
    else:
        rec["exact_recorded_match"] = False
        rec["recompute_delta_macro"] = None
    rec["id_gold"] = ids_gold(rows)
    return rec


def compare_to_cohort(extra, local):
    if extra.get("status") != "ANALYZED_RECOMPUTED" or not extra.get("id_gold"):
        return {"status": "NOT_APPLICABLE"}
    seed = extra.get("seed")
    if seed is None:
        return {"status": "NOT_APPLICABLE"}
    ref = [r for r in local if r["seed"] == seed]
    if not ref:
        return {"status": "NO_SAME_SEED_REFERENCE"}
    ref_ids = None
    for r in ref:
        p = ROOT / r["id_gold_source"]
        vals = ids_gold(load_local_jsonl(p))
        ref_ids = vals if ref_ids is None else ({k: vals[k] for k in ref_ids.keys() if k in vals and ref_ids[k] == vals[k]})
    eid = extra["id_gold"]
    common = set(eid) & set(ref_ids or {})
    mismatches = sorted(k for k in common if eid[k] != ref_ids[k])
    missing = sorted(set(ref_ids or {}) - set(eid))
    return {"status": "MATCH" if not mismatches and not missing and len(common) == 2000 else "MISMATCH",
            "common_count": len(common), "mismatch_count": len(mismatches), "missing_count": len(missing),
            "mismatch_examples": mismatches[:10], "missing_examples": missing[:10]}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    local = cohort_rows()
    save_json(OUT / "q1a_extra_local_cohort_recomputed.json", local)
    checkpoint = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "completed": [], "status": "RUNNING"}
    save_json(OUT / "q1a_extra_checkpoint.json", checkpoint)
    inventory = []
    for spec in extra_specs():
        rec = audit_extra(spec)
        if rec.get("status") == "ANALYZED_RECOMPUTED":
            rec["cohort_match"] = compare_to_cohort(rec, local)
            rec.pop("id_gold", None)  # do not duplicate 2000-row identifiers in the inventory
        inventory.append(rec)
        checkpoint["completed"].append({"variant": spec["variant"], "seed": spec["seed"], "status": rec.get("status"), "reason": rec.get("reason")})
        checkpoint["last_completed"] = checkpoint["completed"][-1]
        save_json(OUT / "q1a_extra_checkpoint.json", checkpoint)
        save_json(OUT / "q1a_extra_run_inventory.json", {"schema_version": 1, "status": "INCREMENTAL", "runs": inventory})
    checkpoint["status"] = "COMPLETE"
    checkpoint["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(OUT / "q1a_extra_checkpoint.json", checkpoint)
    save_json(OUT / "q1a_extra_run_inventory.json", {"schema_version": 1, "status": "ANALYZED", "runs": inventory})

    canonical = list(local)
    for rec in inventory:
        if rec.get("status") != "ANALYZED_RECOMPUTED":
            continue
        canonical.append({"row_id": f"{rec['variant']}__seed{rec['seed']}", "variant": rec["variant"], "display_name": rec["variant"],
                          "seed": rec["seed"], "backbone": "Vistral-7B-Chat" if "vistral" in rec["variant"] else rec.get("identity"),
                          "model_identity": rec.get("identity"), "status": rec["status"], "split": "test",
                          "prediction_count": rec["prediction_count"], "per_label_f1": rec["recomputed"]["per_label_f1"],
                          "macro_f1": rec["recomputed"]["macro_f1"], "recorded_macro_f1": rec.get("recorded_metric"),
                          "recompute_delta_macro": rec.get("recompute_delta_macro"), "exact_recorded_match": rec.get("exact_recorded_match"),
                          "cohort_match": rec.get("cohort_match"), "source_repo": rec["repo"], "source_revision": rec["revision"]})
    save_json(OUT / "q1a_extra_canonical_rows.json", canonical)
    csv_rows = []
    for r in canonical:
        q = {"row_id": r.get("row_id"), "variant": r.get("variant"), "display_name": r.get("display_name"), "seed": r.get("seed"),
             "backbone": r.get("backbone"), "model_identity": r.get("model_identity"), "status": r.get("status"), "split": r.get("split"),
             "prediction_count": r.get("prediction_count"), "macro_f1": r.get("macro_f1"), "recorded_macro_f1": r.get("recorded_macro_f1"),
             "recompute_delta_macro": r.get("recompute_delta_macro"), "exact_recorded_match": r.get("exact_recorded_match"),
             "cohort_match": json.dumps(r.get("cohort_match", ""), ensure_ascii=False), "source_repo": r.get("source_repo", ""), "source_revision": r.get("source_revision", "")}
        q.update({h: r.get("per_label_f1", {}).get(h, "") for h in HEADS})
        csv_rows.append(q)
    write_csv(OUT / "q1a_extra_canonical_rows.csv", csv_rows, ["row_id", "variant", "display_name", "seed", "backbone", "model_identity", "status", "split", "prediction_count"] + HEADS + ["macro_f1", "recorded_macro_f1", "recompute_delta_macro", "exact_recorded_match", "cohort_match", "source_repo", "source_revision"])

    ledger = []
    for r in inventory:
        ledger.append({"variant": r["variant"], "seed": r.get("seed"), "status": r.get("status"), "prediction_count": r.get("prediction_count", r.get("observed_prediction_count", "")),
                       "test_metric_present": bool(r.get("recorded_metric") is not None), "recorded_metric": r.get("recorded_metric", ""),
                       "reason": r.get("reason", ""), "repo": r.get("repo"), "revision": r.get("revision"), "root": r.get("root"),
                       "cohort_match": json.dumps(r.get("cohort_match", ""), ensure_ascii=False)})
    write_csv(OUT / "q1a_extra_coverage_ledger.csv", ledger, ["variant", "seed", "status", "prediction_count", "test_metric_present", "recorded_metric", "reason", "repo", "revision", "root", "cohort_match"])

    provenance = {"status": "ANALYZED", "metric_definition": "binary per-label macro-F1; macro is arithmetic mean of six pragmatic heads", "heads": HEADS,
                  "tolerance": TOL, "local_cohort": local, "extra_runs": [{k: v for k, v in r.items() if k != "id_gold"} for r in inventory],
                  "non_evidence_sources": ["paper/reference PDF", "reports/azure_job_inventory.json", "reports/approved_aggregation_q1a.json", "preflight-only or partial dev outputs"],
                  "note": "Inventory/preflight/approval gates are provenance only; no missing test six-head artifact is converted into a number."}
    save_json(OUT / "q1a_extra_source_proof.json", provenance)

    grouped = {}
    for r in canonical:
        grouped.setdefault(r["variant"], []).append(r["macro_f1"])
    summary = []
    for variant, vals in grouped.items():
        summary.append({"variant": variant, "complete_seed_count": len(vals), "seeds": sorted([r["seed"] for r in canonical if r["variant"] == variant]),
                        "mean_macro_f1": mean(vals), "sd_macro_f1": stdev(vals) if len(vals) > 1 else None})
    save_json(OUT / "q1a_extra_variant_summary.json", summary)
    report = ["# Q1a extra artifact audit", "", "Status: ANALYZED (artifact audit; no training).", "",
              "The canonical CSV/JSON contains the six existing local cohort rows recomputed from saved test JSONL plus only extra rows with a complete saved test six-head prediction source.", "",
              "## Gate interpretation", "",
              "`reports/azure_job_inventory.json` identifies GPT-4.1-mini jobs but has no output/status/approval proof. `reports/approved_aggregation_q1a.json` is BLOCKED with accepted_run_count=0. These are not result evidence. The local CoT seed21 run is NOT_STARTED/PREFLIGHT_ONLY and its dev number is excluded.", "",
              "## Missing-row rule", "",
              "Rows marked `UNVERIFIED/NOT_AVAILABLE` have no complete saved test six-head prediction artifact. Partial dev files, preflight, inventory, approval gates, and metric-only files are not used as Q1a numbers. GPT identity is retained as GPT-4.1-mini where the saved/live metadata says so; no GPT-4o-mini relabeling is made.", "",
              "## Validation", "",
              "Run `python paper/revision_q1a_extra/validate_q1a_extra.py` after the audit. It checks output hashes, six-head recomputation, count=2000, and same-seed sample_id/gold matching. Raw source cache entries include pinned revision, URL, status, byte count, and SHA-256.", ""]
    (OUT / "q1a_extra_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "LEIBNIZ_HANDOFF.md").write_text("# Q1a extra handoff\n\nUse `q1a_extra_canonical_rows.csv`, `q1a_extra_coverage_ledger.csv`, `q1a_extra_run_inventory.json`, and `q1a_extra_source_proof.json`. Only `ANALYZED_RECOMPUTED` rows are numeric. All other expected rows are explicitly `UNVERIFIED/NOT_AVAILABLE` with a reason. No manuscript files were edited.\n", encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "canonical_rows": len(canonical), "extra_runs": len(inventory), "complete_extra": sum(r.get("status") == "ANALYZED_RECOMPUTED" for r in inventory)}, indent=2))


if __name__ == "__main__":
    main()
