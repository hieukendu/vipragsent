"""Validate retained aggregate experimental results; never rerun training or bootstrap."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ("implicit_sentiment","sarcasm","irony","idiom_figurative","code_switching","mocking")
SEEDS = (20260521,20260522,20260523)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))

def f1(matrix):
    a=np.asarray(matrix)
    require(a.ndim==2 and a.shape[0]==a.shape[1] and np.isfinite(a).all() and (a>=0).all() and np.equal(a,a.astype(int)).all(), "Invalid confusion counts")
    den=a.sum(axis=0)+a.sum(axis=1)
    scores=np.divide(2*np.diag(a),den,out=np.zeros(len(a)),where=den!=0)
    return float(scores.mean())

def same(a,b):
    return abs(float(a)-float(b))<1e-10

def ece(bins,n,pkey,ykey):
    require(len(bins)==10 and [b["bin"] for b in bins]==list(range(10)), "Invalid calibration bins")
    require(sum(b["count"] for b in bins)==n, "Calibration cohort size")
    require(all(isinstance(b["count"],int) and b["count"]>=0 and 0<=b[pkey]<=1 and 0<=b[ykey]<=1 for b in bins), "Invalid calibration aggregates")
    return sum(b["count"]/n*abs(b[pkey]-b[ykey]) for b in bins)

def matrix_check(values,paper):
    m=np.asarray(values);means=m.mean(axis=0)
    sd=m.std(axis=0,ddof=1) if len(m)>1 else None
    indices=paper.get("indices",list(range(m.shape[1])))
    require(all(round(float(means[i]),paper["digits"])==v for i,v in zip(indices,paper["means"],strict=True)), "Paper mean mismatch")
    if paper.get("sds"):
        require(sd is not None and all(round(float(sd[i]),paper["digits"])==v for i,v in zip(indices,paper["sds"],strict=True)), "Paper sample SD mismatch")
    return means,sd

def main():
    payload=read("replay/AGGREGATE_RESULTS.json")
    require(payload["individual_predictions_included"] is False and len(payload["runs"])==235, "Aggregate inventory mismatch")
    runs={}
    for r in payload["runs"]:
        key=(r["target"],r["split"],r.get("dataset"))
        require(key not in runs, "Duplicate aggregate run");runs[key]=r
        n=r["n"]
        if "dataset" in r:
            expected={"uit_vsfc":3166,"uit_vsmec":693,"aivivn_human_derived_3way":1609}[r["dataset"]]
            require(n==expected and np.asarray(r["confusion_matrix"]).sum()==n and same(f1(r["confusion_matrix"]),r["macro_f1"]), "External aggregate F1 mismatch")
        else:
            require(n==(1999 if r["split"]=="dev" else 2000), "Aggregate cohort size mismatch")
            for k in LABELS:
                require(np.asarray(r["confusion_matrices"][k]).sum()==n and same(f1(r["confusion_matrices"][k]),r["head_f1"][k]), "Pragmatic aggregate F1 mismatch")
            require(same(np.mean(list(r["head_f1"].values())),r["macro_f1"]), "Macro averaging mismatch")
            if "calibration" in r:
                cal=r["calibration"]
                scores=[ece(cal["reliability"][k],n,"mean_probability","empirical_positive_rate") for k in LABELS]
                require(all(same(v,cal["head_ece"][k]) for k,v in zip(LABELS,scores)) and same(np.mean(scores),cal["macro_ece"]), "Aggregate ECE mismatch")
            if "polarity_calibration" in r:
                cal=r["polarity_calibration"]
                require(same(ece(cal["bins"],n,"mean_confidence","accuracy"),cal["ece"]), "Polarity ECE mismatch")
    contracts=[]
    for name in ("RECOVERED_INPUTS.json","Q3_reported/INPUT_MANIFEST.json","Q3_reported/BASELINE_MANIFEST.json","Q3_reported/VISTRAL32_MANIFEST.json","Q3_reported/PHOBERT_COMPACT_MANIFEST.json"):
        contracts.extend(read("replay/"+name)["accepted"])
    require(len(contracts)==30 and len({g["group"] for g in contracts})==30, "Paper cohort contract mismatch")
    checked=[]
    for g in contracts:
        rr=g["runs"]
        require(len(rr)==(1 if g["group"].startswith("Q1a_azure") else 3), "Incomplete run cohort")
        if len(rr)==3:
            require({int(r["seed"]) for r in rr}==set(SEEDS), "Seed cohort mismatch")
        values=[];polarity=[];pragmatic_ece=[]
        for run in rr:
            test=runs[(run["target"],"test",None)]
            heads=[100*test["head_f1"][k] for k in LABELS]
            values.append(heads+[100*test["macro_f1"]])
            if "calibration" in test:
                pragmatic_ece.append(test["calibration"]["macro_ece"])
            if g["paper"].get("polarity_dev_ece_x1000"):
                polarity.append(1000*runs[(run["target"],"dev",None)]["polarity_calibration"]["ece"])
        means,sd=matrix_check(values,g["paper"])
        if polarity:
            require([round(float(np.mean(polarity)),1),round(float(np.std(polarity,ddof=1)),1)]==g["paper"]["polarity_dev_ece_x1000"], "Paper Q2 calibration mismatch")
        if g["group"]=="Q1a_vistral_pragmatic_sft":
            require([round(float(np.mean(pragmatic_ece)),4),round(float(np.std(pragmatic_ece,ddof=1)),4)]==[.0345,.0041], "Paper Q4 calibration mismatch")
        checked.append({"group":g["group"],"macro_f1_percent_mean":float(means[-1]),"paper_match":True})
    pair=read("replay/REPLAY_REPORT.json")
    for original in pair["per_seed"]:
        actual=runs[(f'runs/{original["model"]}/{original["seed"]}',"test",None)]
        require(same(actual["macro_f1"],original["macro_f1"]) and all(same(actual["head_f1"][k],original["head_f1"][k]) for k in LABELS), "Retained pair F1 mismatch")
        require(same(actual["calibration"]["macro_ece"],original["calibration"]["macro_ece"]), "Retained pair calibration mismatch")
    for model,means,sds in (("proposed_multitask",[92.67,89.36,98.43,93.35,94.84,93.35,93.67],[.24,.90,.20,.66,.56,.73,.19]),("baseline_pragmatic_only",[92.25,88.40,98.18,92.59,94.34,92.45,93.04],[.10,.36,.21,.39,.19,.52,.04])):
        selected=[runs[(f"runs/{model}/{s}","test",None)] for s in SEEDS]
        matrix_check([[*[100*r["head_f1"][k] for k in LABELS],100*r["macro_f1"]] for r in selected],{"digits":2,"means":means,"sds":sds})
    with (ROOT/"replay/reference/paired_bootstrap_ci.csv").open(encoding="utf-8",newline="") as stream:
        reference={r["metric"]:r for r in csv.DictReader(stream)}
    require(len(pair["paired_bootstrap"])==len(reference)==7,"Retained CI inventory")
    for r in pair["paired_bootstrap"]:
        require(r["resamples"]==10000 and r["bootstrap_seed"]==20260525,"Retained bootstrap settings")
        require(all(same(r[k],reference[r["metric"]][k]) for k in ("observed_delta_pp","ci_low_pp","ci_high_pp","p_value")),"Retained CI/reference disagreement")
        require(r["ci_low_pp"]<=r["ci_high_pp"] and 0<=r["p_value"]<=1,"Invalid saved CI")
    q1b_expected={"proposed_multitask":([37.9,40.0,47.9,42.0],[2.8,.7,2.2,1.8]),"phobert_single_task":([29.7,38.8,45.4,38.0],[.5,1.5,2.6,.3]),"phobert_multitask":([33.6,36.4,44.7,38.2],[3.0,.7,1.3,1.5]),"xlmr_eight_head":([39.1,40.2,48.3,42.5],[1.1,.2,1.2,.5])}
    datasets=("uit_vsfc","uit_vsmec","aivivn_human_derived_3way")
    q1b_expected.update({"sailor_multitask":([34.4,40.7,48.7,41.3],[3.2,.7,1.4,1.2]),
                         "vistral_multitask":([30.7,39.4,49.0,39.7],[6.4,1.5,.8,2.5])})
    for system,(means,sds) in q1b_expected.items():
        values=[]
        for seed in SEEDS:
            scores=[100*runs[(f"Q1b/{system}/{seed}","test",d)]["macro_f1"] for d in datasets]
            values.append(scores+[float(np.mean(scores))])
        matrix_check(values,{"digits":1,"means":means,"sds":sds})
    for g in read("replay/Q1b/EXTERNAL_RESULTS.json")["groups"]:
        require(len(g["runs"])==3 and {r["seed"] for r in g["runs"]}==set(SEEDS),"External seed cohort")
        values=[]
        for r in g["runs"]:
            a=[100*r[k] for k in ("vsfc_macro_f1","vsmec_macro_f1","aivivn_macro_f1")]
            require(same(np.mean(a),100*r["ord_f1"]),"External mean definition")
            for dataset,key in zip(datasets,("vsfc_macro_f1","vsmec_macro_f1","aivivn_macro_f1"),strict=True):
                actual=runs[(f'Q1b/{g["system"]}/{r["seed"]}',"test",dataset)]
                require(same(actual["macro_f1"],r[key]),"External count/reference F1 mismatch")
            values.append(a+[100*r["ord_f1"]])
        matrix_check(values,{"digits":1,**g["paper"]})
    usage=read("replay/Q2_followup/RESOURCE_USAGE.json")
    hours={k:float(np.mean([r["successful_gpu_hours"] for r in usage["runs"] if r["variant"]==k])) for k in usage["expected_relative_gpu_hours"]}
    require(all(round(hours[k]/hours["full"],2)==v for k,v in usage["expected_relative_gpu_hours"].items()),"Paper relative compute mismatch")
    usage=read("replay/Q1a/RESOURCE_USAGE.json")
    for model in usage["expected_gpu_hours"]:
        selected=[r for r in usage["runs"] if r["model"]==model]
        require(len(selected)==3 and {r["seed"] for r in selected}==set(SEEDS),"Compute seed cohort")
        require(round(sum(r["wall_seconds"] for r in selected)/3600,2)==usage["expected_gpu_hours"][model] and round(max(r["peak_allocated_gb"] for r in selected),2)==usage["expected_peak_gb"][model],"Paper compute mismatch")
    evidence=read("replay/Q3_reported/VISTRAL32_SELECTION_RESULTS.json")
    require(evidence["selection_epoch_confirmed_in_saved_best_checkpoint"]==2,"Saved best checkpoint epoch mismatch")
    require(evidence["independent_dev_threshold_reselection_verified"] is True and len(evidence["versions"])==1,"Q3 selection evidence scope")
    for v in evidence["versions"]:
        history=v["training_history"];best=max(history,key=lambda r:r["dev_metric"])
        require(best["epoch"]==2 and same(best["dev_metric"],evidence["selected_epoch_dev_sarcasm_f1"]),"Historical epoch selection mismatch")
        require(v["freeze_manifest"]["frozen"] is True and v["freeze_manifest"]["thresholds"]==v["thresholds"],"Historical threshold freeze mismatch")
    search=read("replay/Q3_reported/VISTRAL32_THRESHOLD_SEARCH.json")
    require(search["independent_dev_threshold_reselection_verified"] and search["same_checkpoint_bytes_verified"] and search["same_configuration_verified"],"Recovered selection lineage mismatch")
    require(search["verified_selection_version"]=="original_selection" and len(search["versions"])==1,"Selection version mismatch")
    with (ROOT/"dev.csv").open(encoding="utf-8-sig",newline="") as stream:
        dev=list(csv.DictReader(stream))
    expected_grid=[round(.05+i*.01,2) for i in range(91)]
    for version in search["versions"]:
        require(version["n"]==1999 and version["dev_file_matches_freeze_hash"] and version["gold_and_IDs_match_original_dev"],"Recovered dev cohort mismatch")
        require(set(version["heads"])==set(LABELS),"Incomplete threshold heads")
        for label,head in version["heads"].items():
            require([x["threshold"] for x in head["grid"]]==expected_grid,"Threshold grid mismatch")
            positive=sum(int(r[label]) for r in dev)
            for item in head["grid"]:
                matrix=np.asarray(item["confusion_matrix"])
                require(matrix.shape==(2,2) and matrix.sum(axis=1).tolist()==[1999-positive,positive],"Threshold-grid gold/count mismatch")
                f1(matrix)
            best=min(head["grid"],key=lambda r:(-f1(r["confusion_matrix"]),abs(r["threshold"]-.5),r["threshold"]))
            require(same(best["threshold"],head["reselected_threshold"]),"Independent threshold reselection mismatch")
            require(head["matches_frozen_threshold"]==same(best["threshold"],head["frozen_threshold"]),"Threshold match flag mismatch")
            if version["version"]==search["verified_selection_version"]:
                require(head["matches_frozen_threshold"] and head["saved_dev_prediction_mismatch_at_frozen_threshold"]==0,"Original dev thresholds differ from freeze")
    statistics=read("replay/PAPER_STATISTICS.json")
    for label,systems in statistics["pooled_test_confusion_matrices"].items():
        for model,expected in systems.items():
            matrices=[runs[(f"runs/{model}/{seed}","test",None)]["confusion_matrices"][label] for seed in SEEDS]
            require(np.sum(matrices,axis=0).tolist()==expected,"Pooled error-analysis confusion mismatch")
    result={
        "status":"PASS_AGGREGATE_VALIDATION", "verification_status":"ANALYZED",
        "aggregate_run_splits":len(runs), "paper_matched_groups":checked,
        "Q1a_pair_F1_ECE_match":True,
        "Q1b_all_six_rows_paper_mean_SD_match":True,
        "Q1b_all_six_rows_F1_from_confusion_counts":True,
        "Q1b_Sailor_Vistral_independent_F1_recomputation":True,
        "Q2_F1_ECE_relative_cost_match":True, "Q3_all_reported_groups_match":True,
        "Q4_Vistral_match":True, "saved_CI_matches_reference":True,
        "Q3_Vistral32_epoch_selection_and_threshold_freeze_saved_results_match":True,
        "Q3_Vistral32_independent_dev_threshold_reselection_verified":True,
        "bootstrap_reexecuted":False, "individual_predictions_included":False,
        "full_paper_reproduction":False, "training_or_API_rerun":False,
    }
    require(not list((ROOT/"replay").rglob("*_predictions.jsonl")), "Individual predictions still present; aggregate arithmetic checked, cleanup pending")
    (ROOT/"replay/VALIDATION_REPORT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
