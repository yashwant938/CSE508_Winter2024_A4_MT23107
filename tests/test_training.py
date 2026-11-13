import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_data import collate, make_manifest, training_example, validate_manifest


class Tokenizer:
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class TrainingTests(unittest.TestCase):
    def test_long_review_cannot_displace_target_and_prompt_has_no_loss(self):
        row = training_example(Tokenizer(), 'x' * 100, 'OK', 12, 6)
        self.assertEqual(row['input_ids'][:12], [ord(c) for c in 'xxxxxx TL;DR'])
        self.assertEqual(row['labels'][:12], [-100] * 12)
        self.assertEqual(row['labels'][12:], [ord(' '), ord('O'), ord('K'), 0])
        self.assertEqual(len(row['input_ids']), 16)
        self.assertEqual(row['input_ids'][12:], row['labels'][12:])

    def test_padding_is_ignored_but_real_eos_is_supervised(self):
        a = training_example(Tokenizer(), 'a', 'A', 20, 10)
        b = training_example(Tokenizer(), 'longer review', 'a longer summary', 20, 10)
        batch = collate([a, b], 0)
        end = len(a['input_ids'])
        self.assertEqual(batch['labels'][0, end - 1], 0)
        self.assertTrue((batch['labels'][0, end:] == -100).all())
        self.assertTrue((batch['attention_mask'][0, end:] == 0).all())
        self.assertTrue((batch['attention_mask'][0, :end] == 1).all())

    def test_empty_target_and_invalid_budgets_are_rejected(self):
        for args in [('x', '', 12, 5), ('x', 'y', 2, 5), ('x', 'y', 12, 1)]:
            with self.assertRaises(ValueError):
                training_example(Tokenizer(), *args)

    def test_deduplication_and_split_reproduction(self):
        import pandas as pd
        frame = pd.DataFrame([{'Text': f'{i:04} review ending', 'Summary': 'good'} for i in range(200)])
        frame = pd.concat([frame, frame.iloc[:2], pd.DataFrame([{'Text': '0003 REVIEW ENDING!', 'Summary': 'other'},
                                                             {'Text': '0004 review ending changed', 'Summary': 'other'}])], ignore_index=True)
        config = {'seed': 4, 'input_tokens': 16, 'train_examples': 30, 'validation_examples': 10, 'test_examples': 10}
        manifest = make_manifest(frame, Tokenizer(), config, 'test-hash')
        self.assertEqual(manifest, make_manifest(frame, Tokenizer(), config, 'test-hash'))
        self.assertEqual(manifest['skipped']['normalized_review_duplicate'], 3)
        self.assertEqual(manifest['skipped']['prompt_duplicate'], 1)
        bad = copy.deepcopy(manifest)
        bad['splits']['test'][0] = bad['splits']['train'][0]
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            validate_manifest(bad)

    def test_model_loss_ignores_padding_and_shifts_target_once(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(7)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                         resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        a = training_example(Tokenizer(), 'x', 'ok', 16, 10)
        b = training_example(Tokenizer(), 'longer review', 'fine', 20, 10)
        batch = collate([a, b], 0)
        with torch.no_grad():
            output = model(**batch)
        manual = torch.nn.functional.cross_entropy(output.logits[:, :-1, :].reshape(-1, 128), batch['labels'][:, 1:].reshape(-1), ignore_index=-100)
        self.assertAlmostEqual(float(output.loss), float(manual), places=5)
        count = [(torch.tensor(row['labels'][1:]) != -100).sum().item() for row in (a, b)]
        with torch.no_grad():
            individual = [float(model(**collate([row], 0)).loss) for row in (a, b)]
        self.assertAlmostEqual(float(output.loss), sum(x * n for x, n in zip(individual, count)) / sum(count), places=5)

    def test_evaluation_rejects_config_or_split_changes_before_loading_model(self):
        from evaluate_checkpoint import main
        from train import sha256_file
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for changed in ('run_config.json', 'split_manifest.json'):
                for name in ('run_config.json', 'split_manifest.json', 'training_complete.json'):
                    (run / name).write_text('{}', encoding='utf-8')
                environment = {'config_sha256': sha256_file(run / 'run_config.json'),
                               'split_manifest_sha256': sha256_file(run / 'split_manifest.json')}
                (run / 'environment.json').write_text(json.dumps(environment), encoding='utf-8')
                (run / changed).write_text('{"changed": true}', encoding='utf-8')
                with patch('sys.argv', ['evaluate_checkpoint.py', '--run', str(run), '--output', str(run / 'evaluation.json')]):
                    with self.assertRaisesRegex(ValueError, 'modified after training'):
                        main()


if __name__ == '__main__':
    unittest.main()
