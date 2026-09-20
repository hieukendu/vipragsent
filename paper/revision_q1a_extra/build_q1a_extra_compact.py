"""Build the compact Q1a-extra handoff from the persisted best-F1 report.

The report is used only as an artifact/provenance source.  Per-seed values are
loaded and re-aggregated here with arithmetic mean and sample SD.  Its
bootstrap CI half-width is retained as a separate reported field and is never
used as an SD.
"""
from __future__ import annotations
import csv, hashlib, json, math, shutil, time
from pathlib import Path
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'paper'/'revision_q1a_extra'; REPORT=ROOT/'reports'/'q1a_best_f1_table.json'; RAW=OUT/'raw_sources'
HEADS=['implicit_sentiment','sarcasm','irony','idiom_figurative','code_switching','mocking']; METRICS=HEADS+['macro_pragmatic_f1']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def load_rows(path):
    with path.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def f1(g,p):
    vals=[]
    for c in (0,1):
        tp=sum(a==c and b==c for a,b in zip(g,p)); fp=sum(a!=c and b==c for a,b in zip(g,p)); fn=sum(a==c and b!=c for a,b in zip(g,p))
        pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; vals.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
    return sum(vals)/2

def recompute(rows):
    per={h:f1([int(x['gold'][h]) for x in rows],[int(x['predictions'][h]) for x in rows]) for h in HEADS}
    per['macro_pragmatic_f1']=sum(per.values())/len(HEADS); return per

def local_row(system_id,display,stem,seed):
    pred=ROOT/'paper'/'raw_fairness'/f'{stem}_{seed}.jsonl'; metric=ROOT/'paper'/'raw_fairness'/f'{stem}_{seed}_metrics.json'
    rows=load_rows(pred); calc=recompute(rows); recorded=json.loads(metric.read_text(encoding='utf-8'))
    deltas={m:calc[m]-float(recorded.get('per_label_f1',{}).get(m,recorded.get('macro_pragmatic_f1')) if m!='macro_pragmatic_f1' else recorded.get('macro_pragmatic_f1')) for m in METRICS}
    return {'system_id':system_id,'display_name':display,'source_id':f'local_{stem}_{seed}','run_id':f'local_{stem}_{seed}','seed':str(seed),'status':'ANALYZED_RECOMPUTED','source_type':'local_saved_prediction_jsonl','f1':calc,'row_count':len(rows),'prediction_path':str(pred.relative_to(ROOT)),'metric_path':str(metric.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_sha256':sha(metric),'recompute_deltas':deltas,'exact_recomputed_match':all(abs(x)<=1e-12 for x in deltas.values())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=json.loads(REPORT.read_text(encoding='utf-8')); report_hash=sha(REPORT)
    src_files={p.name:sha(p) for p in [REPORT,REPORT.with_suffix('.csv'),REPORT.with_suffix('.md'),REPORT.with_suffix('.tex')]}
    source_by_id={x['source_id']:x for x in d['source_catalog']}
    compact_by_id={x['source_id']:x for x in d['per_seed_rows']}
    # Recompute all six existing local rows plus the requested primary.
    local=[]
    for system_id,display,stem in [('phobert_pragmatic_single_task','PhoBERT (single-task)','phobert_single'),('phobert_pragmatic_finetune','PhoBERT fine-tune','phobert_finetune'),('xlmr_pragmatic_finetune','XLM-R-large baseline (artifact: XLM-R-large fine-tune)','xlmr_baseline'),('sailor_pragmatic_sft','Sailor-7B SFT','sailor'),('vistral_pragmatic_sft','Vistral-7B SFT','vistral'),('vipragsent_no_auxiliary_vistral','ViPragSent - no auxiliary loss','__remote_noaux__'),('explanation_only_vistral','ViPragSent - explanation only','__remote_expl__')]:
        if stem.startswith('__'): continue
        for s in [21,22,23]: local.append(local_row(system_id,display,stem,s))
    for s in [21,22,23]: local.append(local_row('vipragsent_xlmr_large','ViPragSent XLM-R-large (primary)','target',s))
    # Upgrade rows with directly fetched prediction JSONL from the previous
    # bounded pass.  The inventory is only a local-cache index; compact source
    # paths and hashes remain in the resulting proof.
    old_inv=OUT/'q1a_extra_run_inventory.json'; old={}
    if old_inv.exists():
        try:
            for r in json.loads(old_inv.read_text(encoding='utf-8')).get('runs',[]): old[(r.get('variant'),r.get('seed'))]=r
        except Exception: pass
    for variant in ('vipragsent_no_auxiliary_vistral','explanation_only_vistral'):
        for s in [21,22,23]:
            r=old.get((variant,s),{}); predsrc=next((x for x in r.get('source_records',[]) if 'test_predictions.jsonl' in x.get('path','') and x.get('ok')),None)
            if predsrc and predsrc.get('local_path'):
                rows=load_rows(OUT/predsrc['local_path']); calc=recompute(rows)
                compact=compact_by_id.get(f'q1a_{variant}_{20260500+s}') or compact_by_id.get(f'q1a_{variant}_{20260521 if s==21 else 20260522 if s==22 else 20260523}')
                # source IDs in the report use the full YYYYMMDD seed.
                sid=f'q1a_{variant}_{20260500+s}'
                c=compact_by_id.get(sid)
                deltas={m:calc[m]-float(c['f1'][m]) for m in METRICS} if c else {}
                local.append({'system_id':variant,'display_name': 'ViPragSent - no auxiliary loss' if variant.startswith('vip') else 'ViPragSent - explanation only','source_id':sid,'run_id':sid,'seed':str(20260500+s),'status':'ANALYZED_RECOMPUTED','source_type':'live_HF_prediction_JSONL_cached','f1':calc,'row_count':len(rows),'prediction_path':predsrc.get('path'),'local_cache_path':predsrc.get('local_path'),'prediction_sha256':predsrc.get('sha256'),'recompute_deltas_vs_compact':deltas,'exact_compact_match':bool(c and all(abs(x)<=1e-12 for x in deltas.values()))})
    local_keys={(x['system_id'],str(x['seed'])) for x in local}
    # Compact per-seed rows are the complete remote source catalogue.  Preserve
    # its source IDs/paths and classify non-local rows as reported remote.
    seed_rows=[]
    for x in d['per_seed_rows']:
        sid=x['source_id']; f1vals=x['f1']; base={'system_id':x['system_id'],'display_name':x['display_name'],'run_id':x.get('run_id'),'source_id':sid,'seed':x.get('seed'),'f1':f1vals,'source':x.get('source'),'report_status':x.get('status')}
        # Local rows use local provenance IDs, while the compact report uses
        # canonical remote IDs.  Match by system and seed, preferring the
        # direct-cache override for no-auxiliary/explanation variants.
        local_match=next((z for z in local if z['system_id']==x['system_id'] and str(z['seed'])==str(x.get('seed')) and z.get('source_id')==sid),None)
        if local_match is None:
            local_match=next((z for z in local if z['system_id']==x['system_id'] and str(z['seed'])==str(x.get('seed'))),None)
        if local_match:
            base.update({'status':'ANALYZED_RECOMPUTED','recomputed_f1':local_match['f1'],'recompute_deltas_vs_report':{m:local_match['f1'][m]-f1vals[m] for m in METRICS},'prediction_sha256':local_match.get('prediction_sha256'),'row_count':local_match.get('row_count')})
        elif x.get('status')=='MISSING_TEST_SCORE':
            source=x.get('source',{})
            base.update({'status':'ANALYZED_REMOTE_DEV_ONLY' if source.get('dev_metric_path') else 'UNVERIFIED/NOT_AVAILABLE','reason':source.get('note','missing test metric/prediction artifact')})
        else:
            base.update({'status':'ANALYZED_REPORTED_REMOTE','recomputed_f1':None,'recompute_deltas_vs_report':None,'prediction_sha256':(d.get('validation',{}).get(x['system_id'],{}).get('prediction_sha256') or [None])[0] if len(d.get('validation',{}).get(x['system_id'],{}).get('prediction_sha256',[]))==1 else None,'row_count':(d.get('validation',{}).get(x['system_id'],{}).get('row_counts') or [None])[0] if len(d.get('validation',{}).get(x['system_id'],{}).get('row_counts',[]))==1 else None})
        seed_rows.append(base)
    # The compact report has no primary XLM-R row; append its locally verified
    # seed rows so the requested primary system is not omitted.
    seed_rows += [x for x in local if x['system_id']=='vipragsent_xlmr_large']
    aggregate=[]
    # preserve report order, then primary as the requested extension
    order=[x['system_id'] for x in d['table_rows']]+['vipragsent_xlmr_large']
    for system_id in order:
        rows=[x for x in seed_rows if x['system_id']==system_id and x.get('f1',{}).get('macro_pragmatic_f1') is not None]
        report_row=next((x for x in d['table_rows'] if x['system_id']==system_id),None)
        if not rows: continue
        means={m:mean([x['f1'][m] for x in rows]) for m in METRICS}; sds={m:(stdev([x['f1'][m] for x in rows]) if len(rows)>1 else None) for m in METRICS}
        if system_id=='cot_only_vistral': status='PROVISIONAL_2_OF_3_MISSING_TEST_SCORE'
        elif all(x['status']=='ANALYZED_RECOMPUTED' for x in rows): status='ANALYZED_RECOMPUTED'
        else: status='ANALYZED_REPORTED_REMOTE'
        agg={'system_id':system_id,'display_name':report_row['display_name'] if report_row else 'ViPragSent XLM-R-large (primary)','status':status,'seed_count':len(rows),'expected_seed_count':3 if system_id!='azure_pragmatic_zero_shot' and system_id!='azure_gpt41_mini_8shot' and system_id!='vipragsent_xlmr_large' else (3 if system_id=='vipragsent_xlmr_large' else 1),'metric_definition':d['score_definition'],'mean':means,'sample_sd':sds,'ci_half_width_from_report':({m:report_row['metrics'][m].get('half_width') for m in METRICS} if report_row else None),'seed_source_ids':[x['source_id'] for x in rows],'note':'sample_sd is computed from per-seed F1 values; report CI half_width is a separate bootstrap quantity.'}
        if system_id=='cot_only_vistral': agg['missing_seed_source_id']='q1a_cot_only_vistral_clean_rerun_003__seed_20260522'
        if system_id in ('azure_pragmatic_zero_shot','azure_gpt41_mini_8shot'): agg['model_identity']='GPT-4.1-mini (source identity retained; not GPT-4o-mini)'
        aggregate.append(agg)
    # source proof: compact inputs, local raw hashes, catalog, and cache note.
    proof={'status':'ANALYZED','compact_input':{'path':str(REPORT.relative_to(ROOT)),'generated_at_utc':d['generated_at_utc'],'sha256':report_hash,'companion_hashes':src_files},'metric_definition':d['score_definition'],'ci_definition':d['confidence_interval'],'sample_sd_formula':'statistics.stdev over available per-seed F1 values (n-1 denominator); no CI half-width substitution','source_catalog':d['source_catalog'],'source_validation':d['validation'],'local_recomputed_sources':[x for x in local if x.get('status')=='ANALYZED_RECOMPUTED'],'local_cache_note':'raw_sources is a generated local audit cache; it is not a manuscript/PR payload. Full-record download was stopped at the user-requested checkpoint; compact persisted per-seed source evidence is used for reported-remote rows.','non_evidence_gates':['reports/azure_job_inventory.json (inventory only; empty output/status/approval fields)','reports/approved_aggregation_q1a.json (BLOCKED; accepted_run_count=0)','results/runs/q1a_cot_only_vistral_20260521 (NOT_STARTED/PREFLIGHT_ONLY; dev-only number excluded)']}
    payload={'schema_version':1,'status':'ANALYZED','source_report':str(REPORT.relative_to(ROOT)),'aggregate_rows':aggregate,'per_seed_rows':seed_rows,'coverage_note':'ANALYZED_REMOTE_DEV_ONLY means the exact remote run exists with development/checkpoint evidence but no test six-head metric/prediction evidence; UNVERIFIED/NOT_AVAILABLE remains for rows with no usable remote evidence. No value is imputed.'}
    save(OUT/'q1a_extra_canonical_rows.json',payload)
    fields=['system_id','display_name','status','seed_count','expected_seed_count']+[f'{m}_mean' for m in METRICS]+[f'{m}_sample_sd' for m in METRICS]+['seed_source_ids','note']
    with (OUT/'q1a_extra_canonical_rows.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for a in aggregate:
            q={'system_id':a['system_id'],'display_name':a['display_name'],'status':a['status'],'seed_count':a['seed_count'],'expected_seed_count':a['expected_seed_count'],'seed_source_ids':json.dumps(a['seed_source_ids'],ensure_ascii=False),'note':a['note']}
            q.update({f'{m}_mean':a['mean'][m] for m in METRICS}); q.update({f'{m}_sample_sd':a['sample_sd'][m] for m in METRICS}); w.writerow(q)
    with (OUT/'q1a_extra_per_seed_rows.csv').open('w',newline='',encoding='utf-8') as f:
        fields2=['system_id','display_name','source_id','seed','status','row_count','prediction_sha256']+METRICS+['source_repo','metric_path','prediction_path','reason']
        w=csv.DictWriter(f,fieldnames=fields2); w.writeheader()
        for x in seed_rows:
            s=x.get('source') or {}; q={k:x.get(k,'') for k in fields2}; q.update({m:x.get('f1',{}).get(m,'') for m in METRICS}); q.update({'source_repo':s.get('repo',''),'metric_path':s.get('metric_path',''),'prediction_path':s.get('prediction_path',''),'reason':x.get('reason','')}); w.writerow(q)
    ledger=[]
    for x in seed_rows: ledger.append({'scope':'seed','system_id':x['system_id'],'source_id':x['source_id'],'seed':x.get('seed'),'status':x['status'],'test_metric_available':x.get('f1',{}).get('macro_pragmatic_f1') is not None,'row_count':x.get('row_count',''),'prediction_sha256':x.get('prediction_sha256',''),'reason':x.get('reason','')})
    for a in aggregate: ledger.append({'scope':'aggregate','system_id':a['system_id'],'source_id':'|'.join(a['seed_source_ids']),'seed':f"{a['seed_count']}/{a['expected_seed_count']}",'status':a['status'],'test_metric_available':True,'row_count':'','prediction_sha256':'','reason':a.get('note','')})
    with (OUT/'q1a_extra_coverage_ledger.csv').open('w',newline='',encoding='utf-8') as f:
        fields3=['scope','system_id','source_id','seed','status','test_metric_available','row_count','prediction_sha256','reason']; w=csv.DictWriter(f,fieldnames=fields3); w.writeheader(); w.writerows(ledger)
    save(OUT/'q1a_extra_source_proof.json',proof)
    save(OUT/'q1a_extra_checkpoint.json',{'status':'COMPACT_HANDOFF_COMPLETE','stopped_remote_record_download':True,'raw_cache_retained':True,'source':'reports/q1a_best_f1_table.json','aggregate_count':len(aggregate),'seed_row_count':len(seed_rows),'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    save(OUT/'q1a_extra_run_inventory.json',{'status':'COMPACT_SOURCE_AUDITED','note':'Supersedes incomplete record-download inventory for handoff; raw cache retained separately.','source_report_hash':report_hash,'runs':seed_rows})
    report=['# Q1a extra compact audit','',f'- Compact source: `reports/q1a_best_f1_table.json` generated {d["generated_at_utc"]}; SHA-256 `{report_hash}`.','- Metric: six binary per-head macro-F1 values; macro is their arithmetic mean.','- Aggregates in `q1a_extra_canonical_rows.csv` use arithmetic mean and sample SD across available per-seed rows. The source report bootstrap CI `half_width` is retained separately and is not SD.','- `ANALYZED_RECOMPUTED`: local six-head prediction JSONL was loaded and recomputed; direct no-auxiliary/explanation cache rows are compared to compact per-seed values.','- `ANALYZED_REPORTED_REMOTE`: persisted remote per-seed metric/prediction evidence is preserved with source ID/path/hash, without pretending this worker independently downloaded every source.','- `ANALYZED_REMOTE_DEV_ONLY`: the exact remote run exists with development/checkpoint evidence, but no test metric/prediction evidence exists; dev values are excluded.','- CoT-only is `PROVISIONAL_2_OF_3_MISSING_TEST_SCORE`; seed 20260522 is `ANALYZED_REMOTE_DEV_ONLY`, not absent. No test value is imputed.','- GPT rows retain exact identity `GPT-4.1-mini`; no GPT-4o-mini relabeling.','- `raw_sources/` is generated local audit cache only; full-Vistral record download was stopped at the requested checkpoint (~1.7k files/~85 MB) and is not required by this compact handoff.','- Validation command: `python paper/revision_q1a_extra/validate_q1a_extra_compact.py`.']
    (OUT/'q1a_extra_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (OUT/'LEIBNIZ_HANDOFF.md').write_text('# Q1a extra compact handoff\n\nUse `q1a_extra_canonical_rows.csv/json`, `q1a_extra_per_seed_rows.csv`, `q1a_extra_coverage_ledger.csv`, and `q1a_extra_source_proof.json`. The exact source IDs/remote paths come from `reports/q1a_best_f1_table.json`; `q1a_extra_canonical_rows.csv` reports mean + sample SD, not bootstrap CI half-width. Raw `raw_sources/` is generated local audit cache only. CoT is provisional 2/3: seed 20260522 exists on current HF trees with dev/checkpoint evidence but no test score, so no test value is imputed. GPT identity is GPT-4.1-mini.\n',encoding='utf-8')
    print(json.dumps({'status':'COMPACT_HANDOFF_COMPLETE','aggregate_count':len(aggregate),'seed_rows':len(seed_rows),'report_sha256':report_hash},indent=2))

if __name__=='__main__': main()
