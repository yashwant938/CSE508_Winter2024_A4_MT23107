"""Evaluate review-only prompts. The original notebook checkpoint is not included."""
import argparse
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path
import re
import time

DATA_SHA256 = '7221e36e72c6d817dc8bc9ea30e8f274998dbb702fc82d8cc0fc6514ee76235c'
ALLOWED_GLOBALS = {
    ('pandas.core.frame', 'DataFrame'), ('pandas.core.internals.managers', 'BlockManager'),
    ('pandas._libs.internals', '_unpickle_block'), ('numpy.core.multiarray', '_reconstruct'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.numeric', '_frombuffer'),
    ('pandas.core.indexes.base', '_new_Index'), ('pandas.core.indexes.base', 'Index'),
    ('builtins', 'slice'),
}


class DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f'Unexpected global: {module}.{name}')
        return super().find_class(module, name)


def load_original_dataset(path):
    # Only this repository's exact, known dataset is accepted. Never use this
    # loader for arbitrary downloaded pickles: pickle is executable data.
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError('Dataset hash differs from the expected repository artifact')
    import io
    frame = DatasetUnpickler(io.BytesIO(payload)).load()
    if list(frame.columns) != ['Summary', 'Text']:
        raise ValueError('Unexpected dataset columns')
    return frame


def words(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


def f1(overlap, prediction_length, reference_length):
    if not overlap or not prediction_length or not reference_length:
        return 0.0
    return 2 * overlap / (prediction_length + reference_length)


def rouge_scores(prediction, reference):
    predicted, expected = words(prediction), words(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    # Rolling-row longest common subsequence; deterministic, no stemming.
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for j, ref in enumerate(expected, 1):
            current.append(previous[j - 1] + 1 if token == ref else max(previous[j], current[-1]))
        previous = current
    return {'rouge1': f1(overlap, len(predicted), len(expected)),
            'rougeL': f1(previous[-1], len(predicted), len(expected))}


def baseline(review, limit=16):
    return ' '.join(str(review).split()[:limit])


def generate_batch(model, tokenizer, reviews, input_tokens, output_tokens):
    # Targets are not arguments to this function. Keep the separator even for
    # long reviews and left-pad decoder-only prompts for batched generation.
    import torch
    suffix = tokenizer.encode(' TL;DR', add_special_tokens=False)
    if len(suffix) >= input_tokens:
        raise ValueError('Input token budget is too small for the prompt suffix')
    rows = []
    for review in reviews:
        ids = tokenizer.encode(str(review), add_special_tokens=False)[:input_tokens - len(suffix)]
        rows.append({'input_ids': ids + suffix})
    batch = tokenizer.pad(rows, padding=True, return_tensors='pt')
    batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=output_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    completions = generated[:, batch['input_ids'].shape[1]:]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in completions]


def mean(scores, name):
    return round(sum(item[name] for item in scores) / len(scores), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='preprocessed_data.pkl', help='Exact repository dataset (SHA-256 checked)')
    parser.add_argument('--model', required=True, help='Local Hugging Face model directory or model ID')
    parser.add_argument('--revision', default=None, help='Immutable Hugging Face revision when using a model ID')
    parser.add_argument('--state-dict', default=None, help='Optional trusted notebook-style model.pt weights')
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=24)
    parser.add_argument('--output', default='results/pilot.json')
    args = parser.parse_args()
    if min(args.sample_size, args.batch_size, args.input_tokens, args.output_tokens) < 1:
        parser.error('Sample size, batch size and token limits must be positive')

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    frame = load_original_dataset(args.data)
    _, held_out = train_test_split(np.arange(len(frame)), test_size=0.25, random_state=42)
    if args.sample_size > len(held_out):
        parser.error('Sample size exceeds the held-out split')
    selected = held_out[:args.sample_size]
    rows = frame.iloc[selected]
    reviews = rows['Text'].fillna('').astype(str).tolist()
    references = rows['Summary'].fillna('').astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    weights_sha256 = None
    if args.state_dict:
        weights = Path(args.state_dict)
        weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    started = time.perf_counter()
    predictions = []
    for start in range(0, len(reviews), args.batch_size):
        predictions.extend(generate_batch(model, tokenizer, reviews[start:start + args.batch_size], args.input_tokens, args.output_tokens))
    elapsed = time.perf_counter() - started
    if len(predictions) != len(references):
        raise RuntimeError('Generation did not cover the entire held-out sample')
    baseline_predictions = [baseline(review) for review in reviews]
    model_scores = [rouge_scores(p, r) for p, r in zip(predictions, references)]
    base_scores = [rouge_scores(p, r) for p, r in zip(baseline_predictions, references)]
    items = []
    for row, review, reference, pred, base, score in zip(selected, reviews, references, predictions, baseline_predictions, model_scores):
        items.append({'source_row': int(row), 'review_excerpt': review[:240], 'reference': reference,
                      'gpt2': pred, 'lead_16_words': base, 'gpt2_rougeL': round(score['rougeL'], 4)})
    result = {'data_sha256': DATA_SHA256, 'split': '25% test, random_state=42; first test indices',
              'sample_size': len(rows), 'model': args.model, 'revision': args.revision,
              'state_dict_sha256': weights_sha256,
              'batch_size': args.batch_size, 'input_tokens': args.input_tokens, 'output_tokens': args.output_tokens,
              'generation': 'greedy', 'metric': 'mean per-example ROUGE-1 and ROUGE-L F1; lowercase alphanumeric tokens, no stemming',
              'transformers': transformers.__version__, 'torch': torch.__version__,
              'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
              'generation_seconds': round(elapsed, 3),
              'scores': {'lead_16_words': {k: mean(base_scores, k) for k in ('rouge1', 'rougeL')},
                         'gpt2': {k: mean(model_scores, k) for k in ('rouge1', 'rougeL')}}, 'examples': items}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
