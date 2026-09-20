"""Offline checks for compact Q1a extra handoff."""
import csv,json,math
from pathlib import Path
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'paper'/'revision_q1a_extra'; HEADS=['implicit_sentiment','sarcasm','irony','idiom_figurative','code_switching','mocking']; METRICS=HEADS+['macro_pragmatic_f1']; TOL=1e-12
def main():
 d=json.loads((OUT/'q1a_extra_canonical_rows.json').read_text(encoding='utf-8')); errors=[]
 if d.get('status')!='ANALYZED': errors.append('payload status')
 if len(d.get('aggregate_rows',[]))<12: errors.append(f"aggregate rows={len(d.get('aggregate_rows',[]))}")
 for a in d['aggregate_rows']:
  vals={m:[x['f1'][m] for x in d['per_seed_rows'] if x['system_id']==a['system_id'] and x.get('f1',{}).get(m) is not None] for m in METRICS}
  for m in METRICS:
   if vals[m] and abs(a['mean'][m]-mean(vals[m]))>TOL: errors.append(f"mean:{a['system_id']}:{m}")
   if len(vals[m])>1 and abs(a['sample_sd'][m]-stdev(vals[m]))>TOL: errors.append(f"sd:{a['system_id']}:{m}")
  if a['system_id']=='cot_only_vistral' and a['status']!='PROVISIONAL_2_OF_3_MISSING_TEST_SCORE': errors.append('cot status')
 for x in d['per_seed_rows']:
  if x['system_id'] in ('azure_pragmatic_zero_shot','azure_gpt41_mini_8shot') and 'GPT-4.1-mini' not in json.dumps(x.get('source','')) and x.get('status')!='ANALYZED_REPORTED_REMOTE': errors.append('GPT identity')
  if x['status']!='ANALYZED_RECOMPUTED' and x.get('status')!='ANALYZED_REPORTED_REMOTE' and x.get('status')!='ANALYZED_REMOTE_DEV_ONLY' and x.get('status')!='UNVERIFIED/NOT_AVAILABLE': errors.append(f"unknown status:{x['source_id']}")
 for p in ['q1a_extra_canonical_rows.csv','q1a_extra_per_seed_rows.csv','q1a_extra_coverage_ledger.csv','q1a_extra_source_proof.json','q1a_extra_checkpoint.json']:
  if not (OUT/p).exists(): errors.append(f'missing:{p}')
 result={'status':'PASS' if not errors else 'FAIL','aggregate_rows':len(d['aggregate_rows']),'seed_rows':len(d['per_seed_rows']),'errors':errors}
 (OUT/'q1a_extra_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2)); raise SystemExit(1 if errors else 0)
if __name__=='__main__': main()
