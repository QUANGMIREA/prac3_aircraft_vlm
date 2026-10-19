"""Check prepared photos and metadata without API requests."""
from collections import Counter
from pathlib import Path
from PIL import Image
from main import BASE, DATASET, load_dataset


def main():
    items = load_dataset(DATASET)
    label_counts = Counter(row['label'] for row in items)
    difficulty_counts = Counter(row['difficulty'] for row in items)
    print(f'Rows: {len(items)}')
    print('Labels:', dict(label_counts))
    print('Difficulty:', dict(difficulty_counts))
    missing = []
    bad = []
    missing_sources = []
    for item in items:
        path = BASE / 'images' / item['image']
        if not path.exists():
            missing.append(item['image'])
            continue
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            bad.append((item['image'], str(exc)))
        if not item['source_url']:
            missing_sources.append(item['image'])
    print(f'Missing files: {len(missing)}')
    print(f'Unreadable images: {len(bad)}')
    print(f'Existing images without source_url: {len(missing_sources)}')
    if missing:
        print('Missing:', ', '.join(missing[:12]))
    if bad:
        print('Unreadable:', bad[:4])
    if not missing and not bad and len(items) == 30:
        print('Ready to run inference. Check the labels and image licenses manually.')
    else:
        print('Dataset not complete yet.')


if __name__ == '__main__':
    main()
