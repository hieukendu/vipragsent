"""Verify the standalone ViPragSent release offline using Python's stdlib."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math

ROOT=Path(__file__).resolve().parent
LABELS=['implicit_sentiment','sarcasm','irony','idiom_figurative','code_switching','mocking','polarity','emotion']
BINARY=LABELS[:6]
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jsonl(name):
    return [json.loads(line) for line in (ROOT/name).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def require(condition,message):
    if not condition:raise ValueError(message)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    manifest=json.loads((ROOT/'FILE_MANIFEST.json').read_text())
    listed=set()
    for item in manifest['files']:
        relative=Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe manifest path')
        p=ROOT/relative
        require(p.is_file(),'Missing release payload: '+str(relative))
        require(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Checksum mismatch: '+str(relative))
        listed.add(relative.as_posix())
    excluded=set(manifest.get('excluded_working_directories',[]))
    require(excluded <= {'submission'},'Unsafe exclusion')
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.relative_to(ROOT).parts[0] not in excluded
            and p.relative_to(ROOT).as_posix() not in ['FILE_MANIFEST.json','CHECKSUMS.sha256']}
    require(actual==listed,'Unlisted or missing release file')
    checksums={}
    for line in (ROOT/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest,name=line.split(maxsplit=1);checksums[name.lstrip('*')]=digest.lower()
    require(checksums=={entry['path']:entry['sha256'] for entry in manifest['files']},'Checksum/manifest disagreement')
    schema=json.loads((ROOT/'label_schema.json').read_text())
    full=read('vipragsent_full.csv');gold={r['sample_id']:r for r in full}
    require(len(full)==len(gold)==11997,'Full dataset count/uniqueness')
    require(len({r['source_id'] for r in full})==11997,'Source ID uniqueness')
    extraction_order=sorted(full,key=lambda r:r['sample_id'])
    require(all(int(a['source_id'])<int(b['source_id']) for a,b in zip(extraction_order,extraction_order[1:])),'Source ordering by sample ID')
    for r in full:
        require(r['text'].strip()!='','Empty text')
        require(all(r[k] in ['0','1'] for k in BINARY),'Invalid binary label')
        require(r['polarity'] in schema['polarity'] and r['emotion'] in schema['emotion'],'Invalid affect label')
        require(r['timestamp']==r['collected_at']=='2026-08-21T07:00:00+07:00','Collection timestamp')
        require(r['timestamp_type']=='collection_time' and r['timestamp_precision']=='approximate'
                and r['timestamp_source']=='user_reported' and r['timestamp_timezone']=='Asia/Ho_Chi_Minh','Time semantics')
        require(r['source_timestamp']==r['source_platform']=='','Unavailable metadata populated')
        require(r['source_timestamp_status']==r['source_platform_status']=='not_provided_by_upstream','Missingness semantics')
    all_ids=set();sizes={};splits={}
    for name,n in [('train',7998),('dev',1999),('test',2000)]:
        rows=read(name+'.csv');ids={r['sample_id'] for r in rows}
        require(len(rows)==len(ids)==n,'Split count/uniqueness: '+name)
        require(not all_ids.intersection(ids),'Split overlap')
        require(all(r==gold[r['sample_id']] and r['split']==name for r in rows),'Split/full mismatch')
        require(ids=={r['sample_id'] for r in full if r['split']==name},'Split membership')
        all_ids.update(ids);splits[name]={r['sample_id']:r for r in rows};sizes[name]=n
    require(all_ids==set(gold),'Incomplete splits')
    split_log=read('split_manifest.csv')
    require(len(split_log)==11997 and {r['sample_id'] for r in split_log}==set(gold),'Split log coverage')
    require(all(r['split']==gold[r['sample_id']]['split'] for r in split_log),'Split log assignment')
    annotators=[{r['sample_id']:r for r in read('annotations/annotator_'+str(i)+'.csv')} for i in [1,2]]
    require(all(len(a)==11997 and set(a)==set(gold) for a in annotators),'Annotator coverage')
    train=splits['train'];rats=jsonl('rationales/approved_generated_rationales_train.jsonl')
    inputs=jsonl('rationales/azure_rationale_input_train.jsonl')
    require(len(rats)==len({r['sample_id'] for r in rats})==7998 and {r['sample_id'] for r in rats}==set(train),'Rationale IDs')
    require(len(inputs)==7998 and {r['sample_id'] for r in inputs}==set(train),'Generator input IDs')
    for r in inputs:
        g=train[r['sample_id']]
        require(r['comment']==g['text'],'Generator input text')
        require(all(str(r['gold_labels'][k])==g[k] for k in LABELS),'Generator input gold')
    positive={sid for sid,g in train.items() if g['sarcasm']=='1'};negative=set(train)-positive
    require(len(positive)==545 and len(negative)==7453,'Q3 train counts')
    previous=set();budgets={}
    for budget in ['32','64','128','256','512','full']:
        rows=read('q3_masks/budget_'+budget+'_masks.csv')
        require(len(rows)==len({r['sample_id'] for r in rows})==7998 and {r['sample_id'] for r in rows}==set(train),'Q3 coverage')
        chosen=set()
        for r in rows:
            sid=r['sample_id'];p=int(sid in positive);selected=int(r['positive_selected_for_budget'])
            require(int(r['is_sarcasm_positive'])==p,'Q3 positive flag')
            require(selected in [0,1] and (p or selected==0),'Q3 selection')
            require(int(r['sarcasm_target_mask'])==(selected if p else 1),'Q3 sarcasm mask')
            require(int(r['rationale_loss_mask'])==p,'Paper Q3 rationale mask')
            if selected:chosen.add(sid)
        require(len(chosen)==(545 if budget=='full' else int(budget)) and previous.issubset(chosen),'Q3 nesting/budget')
        previous=chosen;budgets[budget]=len(chosen)
    masks=json.loads((ROOT/'q3_masks/manifest.json').read_text())
    require(masks['rationale_mask_policy']=='paper_reported','Q3 manifest policy')
    require(all(sha(ROOT/f'q3_masks/budget_{b}_masks.csv')==v.lower() for b,v in masks['mask_hashes'].items()),'Q3 input byte locks')
    dataset=json.loads((ROOT/'dataset_manifest.json').read_text())
    require(sha(ROOT/'rationales/azure_rationale_input_train.jsonl')==dataset['active_rationale_sha256'].lower(),'Rationale input byte lock')
    require(sha(ROOT/'rationales/approved_generated_rationales_train.jsonl')==dataset['approved_generated_rationale_sha256'].lower(),'Generated rationale byte lock')
    weights=json.loads((ROOT/'loss_weights_train.json').read_text())
    for k in BINARY:
        p=sum(int(r[k]) for r in train.values())
        require(math.isclose(weights['pragmatic_pos_weight'][k],(7998-p)/p,rel_tol=1e-12),'Loss weight: '+k)
    require(not list((ROOT/'replay').rglob('*_predictions.jsonl')),'Individual prediction records must not be packaged')
    require((ROOT/'replay/AGGREGATE_RESULTS.json').is_file(),'Aggregate experimental results missing')
    receipt=json.loads((ROOT/'SOURCE_RECEIPT.json').read_text(encoding='utf-8'))
    require(receipt['status']=='RECONSTRUCTED_CURRENT' and receipt['historical_extraction_receipt_available'] is False,'Receipt overclaim')
    require(all(sha(ROOT/name)==value for name,value in receipt['delivered_file_sha256'].items()),'Receipt byte locks')
    print(json.dumps({'status':'PASS','checksum_files':len(listed),'rows':len(full),'splits':sizes,
        'q3_positive_budgets':budgets,
        'individual_predictions_included':False,'full_paper_reproduction':False,
        'q3_rationale_policy':'paper_reported',
        'excluded_generated_directories':sorted(excluded),
        'upstream_online_verification':False,'historical_snapshot_comparison_repeated':False,
        'training_or_API_rerun':False},indent=2))
if __name__=='__main__':main()
