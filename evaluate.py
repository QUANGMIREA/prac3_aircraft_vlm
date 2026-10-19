"""Metrics are computed only on photos with a valid response from BOTH models."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main import BASE, DATASET, RESULTS, load_dataset


def metrics(pairs: list[tuple[str, str]]) -> dict[str, float]:
    if not pairs:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    tp = sum(y == 'MILITARY' and p == 'MILITARY' for y, p in pairs)
    fp = sum(y == 'CIVILIAN' and p == 'MILITARY' for y, p in pairs)
    tn = sum(y == 'CIVILIAN' and p == 'CIVILIAN' for y, p in pairs)
    fn = sum(y == 'MILITARY' and p == 'CIVILIAN' for y, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': (tp + tn) / len(pairs), 'precision': precision,
            'recall': recall, 'f1': (2 * precision * recall / (precision + recall)
                                   if precision + recall else 0.0),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def evaluate() -> None:
    if not RESULTS.exists():
        raise SystemExit('No results/predictions.csv. Run main.py first.')
    dataset = {x['image']: x for x in load_dataset(DATASET)}
    with RESULTS.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    models = list(dict.fromkeys(r['model'] for r in rows))
    if len(models) != 2:
        raise SystemExit('Evaluation requires predictions from exactly 2 models. '
                         'Use a clean results/predictions.csv for each model pair.')
    lookup = {(r['image'], r['model']): r for r in rows}
    common = [name for name in dataset if all(
        (name, m) in lookup and lookup[(name, m)]['status'] == 'ok'
        for m in models)]
    print(f'Dataset images: {len(dataset)} | Common valid predictions: {len(common)}')
    if len(common) < len(dataset):
        print('WARNING: incomplete paired evaluation; missing/invalid predictions '
              'are excluded. Do not claim results for all 30.')
    if not common:
        raise SystemExit('No image has valid predictions from both models.')

    report = []
    for model in models:
        pairs = [(dataset[name]['label'], lookup[(name, model)]['prediction'])
                 for name in common]
        scores = metrics(pairs)
        report.append({'model': model, 'n': len(common), **scores})
        print(f'\n{model}\n  accuracy={scores["accuracy"]:.3f}, '
              f'precision={scores["precision"]:.3f}, recall={scores["recall"]:.3f}, '
              f'F1={scores["f1"]:.3f}\n'
              f'  TP={scores["tp"]} FP={scores["fp"]} '
              f'TN={scores["tn"]} FN={scores["fn"]}')

    output = BASE / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'n', 'accuracy', 'precision',
                                               'recall', 'f1', 'tp', 'fp', 'tn', 'fn'])
        writer.writeheader()
        writer.writerows(report)

    width = 0.36
    import numpy as np
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    names = ['Accuracy', 'Precision', 'Recall', 'F1']
    for i, entry in enumerate(report):
        values = [entry[x] for x in ('accuracy', 'precision', 'recall', 'f1')]
        ax.bar(x + (i - 0.5) * width, values, width,
               label=entry['model'].split('/')[-1])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score (0–1)')
    ax.set_title(f'Two-model comparison on {len(common)} paired aircraft photos')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'metrics_chart.png', dpi=180)
    plt.close(fig)

    with (output / 'paired_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ground_truth', models[0], models[1], 'difficulty', 'source_url'])
        for name in common:
            item = dataset[name]
            writer.writerow([name, item['label'], lookup[(name, models[0])]['prediction'],
                             lookup[(name, models[1])]['prediction'], item['difficulty'],
                             item['source_url']])
    print('\nSaved results/metrics.csv, results/metrics_chart.png, '
          'results/paired_predictions.csv')


if __name__ == '__main__':
    evaluate()
