"""Практическая работа № 3: сравнение двух VLM на фотографиях самолётов."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import re
import time
from pathlib import Path

from PIL import Image, ImageOps
from huggingface_hub import InferenceClient

BASE = Path(__file__).resolve().parent
DATASET = BASE / 'dataset.csv'
RESULTS = BASE / 'results' / 'predictions.csv'
MODELS = (
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'zai-org/GLM-4.5V',
)
PROMPT = (
    "Look at the aircraft in the image. "
    "Classify its operational role as MILITARY or CIVILIAN. "
    "Military transport aircraft, tankers, fighters and "
    "military surveillance aircraft are MILITARY. "
    "Commercial, private and civilian aircraft are CIVILIAN. "
    "Return only one label: MILITARY or CIVILIAN. "
    "Do not explain your answer."
)


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'Cannot find dataset: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image', 'label', 'difficulty', 'source_url'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'dataset.csv needs columns {sorted(required)}')
        data = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            item = {k: (v or '').strip() for k, v in row.items()}
            image = item['image']
            if not image or image in seen:
                raise ValueError(f'Empty/duplicate image filename at line {line}')
            if '/' in image or '\\' in image or image in {'.', '..'}:
                raise ValueError(f'Image name must be a filename (line {line})')
            if item['label'] not in {'MILITARY', 'CIVILIAN'}:
                raise ValueError(f'Invalid label at line {line}: {item["label"]}')
            if item['difficulty'] not in {'easy', 'medium', 'hard'}:
                raise ValueError(f'Invalid difficulty at line {line}: {item["difficulty"]}')
            seen.add(image)
            data.append(item)
    if not data:
        raise ValueError('dataset.csv has no rows')
    return data


def image_data_url(path: Path) -> str:
    """Reduce size and strip metadata before sending image to an inference provider."""
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        rgb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def parse_prediction(text: str) -> str:
    """Extract a single unambiguous category from a VLM text answer.

    Some providers return a sentence even when asked for a one-word answer.
    Never guess when both category names appear or no label is present.
    """
    cleaned = (text or '').strip().upper()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if re.fullmatch(r'(?:MILITARY|CIVILIAN)[.!]?', cleaned):
        return 'MILITARY' if cleaned.startswith('MILITARY') else 'CIVILIAN'

    # Accept typical natural-language responses with exactly one label.
    labels = set(re.findall(r'\b(?:MILITARY|CIVILIAN)\b', cleaned))
    if len(labels) != 1:
        return 'INVALID'
    label = next(iter(labels))

    # Avoid the most obvious incorrect interpretations of negation.
    if re.search(r'\b(?:NOT|NEVER|ISN\'T|IS NOT|NON)\s+(?:AN?\s+)?' + label + r'\b', cleaned):
        return 'INVALID'
    return label


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['image', 'model', 'ground_truth', 'prediction', 'status',
                  'raw_response', 'seconds', 'error', 'finish_reason',
                  'completion_tokens']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict(client: InferenceClient, model: str, path: Path, max_tokens: int) -> tuple[str, str, float, str, str]:
    """Return classification, visible output, duration, stop reason and token usage."""
    url = image_data_url(path)
    start = time.perf_counter()
    
    response = client.chat.completions.create(
    model=model,
    messages=[{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': PROMPT},
            {'type': 'image_url', 'image_url': {'url': url}},
        ],
    }],
    max_tokens=max_tokens,
    temperature=0,
    extra_body={
    "chat_template_kwargs": {
        "enable_thinking": False
    }
    },
)

    elapsed = time.perf_counter() - start
    choice = response.choices[0]

    

    content = choice.message.content
    if isinstance(content, str):
        answer = content
    elif isinstance(content, list):
        answer = ' '.join(
            item.get('text', '') if isinstance(item, dict)
            else getattr(item, 'text', '') or ''
            for item in content
        )
    else:
        answer = ''
    finish_reason = str(getattr(choice, 'finish_reason', None) or '')
    usage = getattr(response, 'usage', None)
    completion_tokens = str(getattr(usage, 'completion_tokens', '') or '')
    return parse_prediction(answer), answer, elapsed, finish_reason, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare two Hugging Face VLMs on airplane photos')
    parser.add_argument('--model-a', default=MODELS[0])
    parser.add_argument('--model-b', default=MODELS[1])
    parser.add_argument('--image', help='Only run this image filename, e.g. military_08.jpg')
    parser.add_argument('--only-model', help='Run only one of the two models')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max output tokens per call (default 512, billed by usage)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Run only N first existing dataset images (for a small smoke test)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run successful predictions; may incur extra costs')
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be >= 1')
    if args.only_model and args.only_model not in (args.model_a, args.model_b):
        parser.error('--only-model must match --model-a or --model-b')
    if args.model_a == args.model_b:
        parser.error('Please use two different models')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    token = os.environ.get('HF_TOKEN')
    if not token:
        parser.error('HF_TOKEN not set. Set it in PowerShell first.')
    items = load_dataset(DATASET)
    if len(items) != 30:
        print(f'WARNING: dataset contains {len(items)} entries; task suggests about 30.')
    items_with_file = [item for item in items if (BASE / 'images' / item['image']).is_file()]
    missing = [item['image'] for item in items if not (BASE / 'images' / item['image']).is_file()]
    if missing:
        print(f'INFO: {len(missing)} image files are missing and will be skipped. '
              f'Example: {", ".join(missing[:3])}')
    if not items_with_file:
        parser.error('No images available in images/. Add photos listed in dataset.csv.')
    if args.image:
        items_with_file = [x for x in items_with_file if x['image'] == args.image]
        if not items_with_file:
            parser.error(f'Image not found in dataset/images: {args.image}')
    if args.limit is not None:
        items_with_file = items_with_file[:args.limit]

    results = read_existing(RESULTS)
    success = {(r['image'], r['model']) for r in results if r.get('status') == 'ok'}
    client = InferenceClient(api_key=token, provider='auto', timeout=90)
    models_to_run = [args.only_model] if args.only_model else [args.model_a, args.model_b]
    for item in items_with_file:
        for model in models_to_run:
            key = (item['image'], model)
            if key in success and not args.force:
                print(f'[SKIP] {item["image"]} {model}: already completed')
                continue
            if args.force:
                results = [r for r in results if (r['image'], r['model']) != key]
            else:
                results = [r for r in results if (r['image'], r['model']) != key]
            print(f'[RUN] {item["image"]} | {model}', flush=True)
            try:
                prediction, raw, seconds, finish_reason, token_count = predict(
                    client, model, BASE / 'images' / item['image'], args.max_tokens
                )
                status = 'ok' if prediction in {'MILITARY', 'CIVILIAN'} else 'invalid'
                result = {'image': item['image'], 'model': model,
                          'ground_truth': item['label'], 'prediction': prediction,
                          'status': status, 'raw_response': raw,
                          'seconds': f'{seconds:.2f}', 'error': '',
                          'finish_reason': finish_reason, 'completion_tokens': token_count}
                print(f'  -> {prediction} ({seconds:.2f}s)')
                if status == 'invalid':
                    print(f'  Diagnostic: finish_reason={finish_reason!r}; '
                          f'completion_tokens={token_count!r}; raw_response={raw[:250]!r}')
                if status == 'ok':
                    success.add(key)
            except Exception as exc:
                msg = str(exc)
                result = {'image': item['image'], 'model': model,
                          'ground_truth': item['label'], 'prediction': '',
                          'status': 'error', 'raw_response': '', 'seconds': '',
                          'error': msg[:500],
                          'finish_reason': '', 'completion_tokens': ''}
                print(f'  ERROR: {msg[:250]}')
            results.append(result)
            save_results(RESULTS, results)
            if result['status'] in {'error', 'invalid'}:
                print('Stopped after unsuccessful response to avoid more paid calls. '
                      'Inspect diagnostic information before retrying.')
                return
    print(f'Finished. Saved to: {RESULTS}')
    print('Next: python evaluate.py')


if __name__ == '__main__':
    main()
