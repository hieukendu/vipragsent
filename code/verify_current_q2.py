"""Recompute the current paper's primary-recipe Q2 table from frozen predictions."""
from pathlib import Path
from release_paths import default_release
import argparse,csv,json,os,re
import numpy as np
from sklearn.metrics import f1_score

LABELS=('implicit_sentiment','sarcasm','irony','idiom_figurative','code_switching','mocking')
VARIANTS=('full','no_emotion_auxiliary','no_polarity_auxiliary','no_rationale',
          'no_uncertainty_weighting','all_multipliers_1','beta_01','beta_05')
SEEDS=(20260521,20260522,20260523)

def ece(gold, probabilities):
    # Ten equal-width bins; p=1 belongs to the last bin, as in Appendix C.
    index=np.minimum((probabilities*10).astype(int),9)
    return sum(abs(float(np.mean(probabilities[index==b]-gold[index==b])))
               *float(np.mean(index==b)) for b in range(10) if np.any(index==b))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,help='Separately supplied saved-prediction bundle (not included)')
    parser.add_argument('--release',type=Path,default=default_release())
    args=parser.parse_args()
    if args.bundle is None:
        parser.error('BLOCKED: saved predictions are not included; supply --bundle explicitly.')
    bundle=args.bundle.resolve()
    if os.name=='nt' and not str(bundle).startswith('\\\\?\\'):bundle=Path('\\\\?\\'+str(bundle))
    if not (bundle/'paper_tables/tab_q2.csv').is_file():
        parser.error('BLOCKED: the supplied bundle has no primary-recipe Q2 table.')
    with (args.release.resolve()/'test.csv').open(encoding='utf-8-sig',newline='') as f:
        gold_by_id={r['sample_id']:r for r in csv.DictReader(f)}
    with (bundle/'paper_tables/tab_q2.csv').open(encoding='utf-8-sig',newline='') as f:
        table=[r for r in list(csv.reader(f))[1:] if r[0]!='Variant']
    with (bundle/'tables/Q1a_all_30_runs_recomputed.csv').open(encoding='utf-8-sig',newline='') as f:
        primary_full={int(r['seed_or_repeat']):r for r in csv.DictReader(f) if 'v37' in r['prediction_path']}
    assert len(table)==len(VARIANTS)
    checked=0
    for variant,row in zip(VARIANTS,table):
        values=[]
        for seed in SEEDS:
            pred=bundle/f'canonical_runs/priority12_v37_q1_{variant}_{seed}/predictions/test_predictions.jsonl'
            if variant=='full':
                pred=bundle/primary_full[seed]['prediction_path']
            elif not pred.exists():
                pred=bundle/f'artifact_archive/reports/priority12_v37/priority1/{variant}/seed_{seed}/predictions/test_predictions.jsonl'
            data=[json.loads(line) for line in pred.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            ids=[r['sample_id'] for r in data]
            assert len(ids)==len(set(ids))==len(gold_by_id)==2000
            assert set(ids)==set(gold_by_id)
            assert all(int(r['gold'][k])==int(gold_by_id[r['sample_id']][k]) for r in data for k in LABELS)
            scores=[f1_score([int(r['gold'][k]) for r in data],[int(r['predictions'][k]) for r in data],
                             labels=[0,1],average='macro',zero_division=0) for k in LABELS]
            calibration=np.mean([ece(np.array([int(r['gold'][k]) for r in data]),
                                     np.array([float(r['probabilities'][k]) for r in data])) for k in LABELS])
            values.append((100*np.mean(scores),100*scores[1],calibration));checked+=1
        means=np.mean(values,axis=0);std=np.std(values,axis=0,ddof=1)
        for col,(mean,sd) in enumerate(zip(means,std),start=1):
            expected=[float(v) for v in re.findall(r'\d+\.\d+',row[col])]
            assert len(expected)==2
            tolerance=5e-5 if col<3 else 5e-7
            assert abs(mean-expected[0])<=tolerance and abs(sd-expected[1])<=tolerance,(variant,col,mean,sd,expected)
    print(json.dumps({'status':'PASS','primary_Q2_prediction_files':checked,'variants':len(VARIANTS),
                      'binary_macro_F1_sarcasm_ECE_mean_and_sample_SD_match':True,
                      'test_IDs_and_gold_match_ARR_clean':True,'historical_followup_used':False,
                      'training_or_API_rerun':False},indent=2))

if __name__=='__main__':main()
