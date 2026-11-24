"""Recompute the delivered Q1a/Q1b/Q3 F1 values from saved predictions."""
import argparse
import csv
import json
import os
from pathlib import Path
from release_paths import default_release
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Separately supplied saved-prediction bundle (not included)')
    parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parent.parent / 'data/ARR_clean')
    args = parser.parse_args()
    if args.bundle is None:
        parser.error('BLOCKED: saved predictions are not included; supply --bundle explicitly.')
    bundle = args.bundle.resolve()
    if os.name == 'nt' and not str(bundle).startswith('\\\\?\\'):
        bundle = Path('\\\\?\\' + str(bundle))
    labels = ['implicit_sentiment', 'sarcasm', 'irony', 'idiom_figurative', 'code_switching', 'mocking']
    if not (bundle / 'tables/Q1a_all_30_runs_recomputed.csv').is_file():
        parser.error('BLOCKED: the supplied bundle has no Q1a prediction table.')
    with (args.release.resolve() / 'test.csv').open(encoding='utf-8-sig', newline='') as stream:
        canonical_test = {row['sample_id']: row for row in csv.DictReader(stream)}
    assert len(canonical_test) == 2000
    total = {}
    for family, filename in [('Q1a', 'Q1a_all_30_runs_recomputed.csv'), ('Q1b', 'Q1b_prediction_validation.csv'), ('Q3', 'Q3_all_54_runs.csv')]:
        with (bundle / 'tables' / filename).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            relative = row.get('prediction_path') or f"canonical_runs/{row['run_id']}/predictions/test_predictions.jsonl"
            data = [json.loads(line) for line in (bundle / relative).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            ids = [item['sample_id'] for item in data]
            assert len(ids) == len(set(ids)), (family, row['run_id'])
            if family == 'Q1b':
                gold = [item['gold'] for item in data]
                predicted = [item['prediction'] for item in data]
                score = f1_score(gold, predicted, labels=sorted(set(gold)), average='macro', zero_division=0)
                assert abs(score - float(row['macro_f1'])) < 1e-12
            else:
                assert set(ids) == set(canonical_test), (family, row['run_id'], 'test membership differs')
                assert all(int(item['gold'][label]) == int(canonical_test[item['sample_id']][label])
                           for item in data for label in labels), (family, row['run_id'], 'gold differs')
                scores = {label: f1_score([int(item['gold'][label]) for item in data], [int(item['predictions'][label]) for item in data], labels=[0, 1], average='macro', zero_division=0) for label in labels}
                assert len(data) == 2000
                assert abs(sum(scores.values()) / len(labels) - float(row['macro_pragmatic_f1'])) < 1e-12
                if family == 'Q1a':
                    assert all(abs(value - float(row[label])) < 1e-12 for label, value in scores.items())
                else:
                    assert abs(scores['sarcasm'] - float(row['sarcasm_binary_macro_f1'])) < 1e-12
        total[family] = len(rows)
    print(json.dumps({'status': 'PASS', 'verified_prediction_files': total,
                      'Q1a_Q3_test_membership_and_gold_match_ARR_clean': True,
                      'training_or_API_rerun': False}, indent=2))

if __name__ == '__main__':
    main()
