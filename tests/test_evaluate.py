import pickle
import unittest
from evaluate import DatasetUnpickler, baseline, generate_batch, rouge_scores


class EvaluationTests(unittest.TestCase):
    def test_rouge_uses_all_tokens_and_order(self):
        self.assertEqual(rouge_scores('one two', 'one two'), {'rouge1': 1, 'rougeL': 1})
        self.assertEqual(rouge_scores('two one', 'one two'), {'rouge1': 1, 'rougeL': .5})
        self.assertEqual(rouge_scores('', 'one'), {'rouge1': 0, 'rougeL': 0})

    def test_baseline_is_deterministic_and_only_reads_review(self):
        self.assertEqual(baseline('one two three', limit=2), 'one two')

    def test_unexpected_pickle_global_is_rejected(self):
        import io
        payload = pickle.dumps(eval)
        with self.assertRaises(pickle.UnpicklingError):
            DatasetUnpickler(io.BytesIO(payload)).load()

    def test_generation_covers_batch_and_decodes_only_completions(self):
        import torch

        class Tokenizer:
            eos_token_id = 0
            def __init__(self): self.inputs = []
            def encode(self, text, add_special_tokens=False):
                self.inputs.append(text)
                return [4] if text == ' TL;DR' else [1, 2, 3]
            def pad(self, rows, padding=True, return_tensors='pt'):
                return {'input_ids': torch.tensor([row['input_ids'] for row in rows]),
                        'attention_mask': torch.ones((len(rows), len(rows[0]['input_ids'])), dtype=torch.long)}
            def decode(self, row, skip_special_tokens=True): return str(row.tolist())

        class Model:
            device = 'cpu'
            def generate(self, input_ids, attention_mask, **kwargs):
                return torch.cat((input_ids, torch.tensor([[7], [8]])), dim=1)

        tokenizer = Tokenizer()
        outputs = generate_batch(Model(), tokenizer, ['review one', 'review two'], 8, 1)
        self.assertEqual(outputs, ['[7]', '[8]'])
        self.assertEqual(tokenizer.inputs, [' TL;DR', 'review one', 'review two'])


if __name__ == '__main__':
    unittest.main()
