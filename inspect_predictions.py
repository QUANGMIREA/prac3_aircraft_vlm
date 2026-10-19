"""Offline, zero-cost summary of predictions.csv; never calls the Hugging Face API."""
import csv
from collections import Counter
from pathlib import Path
p = Path(__file__).resolve().parent / 'results' / 'predictions.csv'
if not p.exists():
    raise SystemExit(f'Not found: {p}')
with p.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
print('Total rows:', len(rows))
print('By status:', dict(Counter(r.get('status', '') for r in rows)))
empty = [r for r in rows if r.get('status') == 'invalid' and not (r.get('raw_response') or '').strip()]
nonempty = [r for r in rows if r.get('status') == 'invalid' and (r.get('raw_response') or '').strip()]
print('INVALID with blank raw_response:', len(empty))
print('INVALID with nonblank raw_response:', len(nonempty))
for row in nonempty[:3]:
    print('Nonblank example:', row.get('image'), row.get('model'), repr(row.get('raw_response', '')[:150]))
