import copy
import json
from pathlib import Path
import tempfile
import unittest

from comparison_data import (NearDuplicates, assert_fresh, assert_hashes, seq2seq_collate,
                             t5_example, t5_input)
from train import sha256_file
from compare_summarizers import verify_partial


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        ids = list(range(2, len(text.split()) + 2))
        return ids + [1] if add_special_tokens else ids


class ComparisonTests(unittest.TestCase):
    def test_seq2seq_padding_eos_and_independent_loss(self):
        import torch
        import torch.nn.functional as F
        from transformers import T5Config, T5ForConditionalGeneration
        config = {"t5_input_tokens": 32, "target_tokens": 4}
        rows = [t5_example(Tokenizer(), "nice food", "a b c d e f", config),
                t5_example(Tokenizer(), "bad", "stale", config)]
        batch = seq2seq_collate(rows, 0)
        self.assertEqual(batch["labels"].tolist(), [[2, 3, 4, 1], [2, 1, -100, -100]])
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)
        model = T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_ff=32,
            num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
            pad_token_id=0, eos_token_id=1, dropout_rate=0)).eval()
        result = model(**batch)
        independent = F.cross_entropy(result.logits.reshape(-1, 32), batch["labels"].reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(result.loss, independent))
        self.assertEqual((batch["labels"] != -100).sum().item(), 6)

    def test_input_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "truncated evidence"):
            t5_input(Tokenizer(), "one two three four", 3)

    def test_fresh_excludes_each_identity_and_self_duplicates(self):
        old = {k: "old" for k in ("row", "review_sha256", "prompt_sha256", "t5_prompt_sha256", "visible_sha256")}
        new = {k: "new" for k in old}
        good = {"historical_exclusions": [old], "splits": {"fresh_test": [new]}}
        assert_fresh(good)
        for key in old:
            changed = copy.deepcopy(good)
            changed["splits"]["fresh_test"][0][key] = old[key]
            with self.assertRaises(ValueError):
                assert_fresh(changed)
        good["splits"]["fresh_test"].append(new)
        with self.assertRaises(ValueError):
            assert_fresh(good)

    def test_near_duplicate_search(self):
        index = NearDuplicates(.8)
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        index.add(text)
        self.assertTrue(index.matches(text.upper()))
        self.assertTrue(index.matches(text + " sixteen"))
        self.assertFalse(index.matches("different product and no shared five word chunks here"))
        self.assertFalse(index.matches("short"))

    def test_model_tamper_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model"
            path.write_text("before", encoding="utf-8")
            expected = {"model": sha256_file(path)}
            assert_hashes(folder, expected)
            path.write_text("after", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_hashes(folder, expected)

    def test_resume_rejects_changed_experiment_or_row_order(self):
        identity = {"checkpoint": "fixed", "beams": 4}
        rows = [{"row": 5}, {"row": 9}]
        partial = {"identity": identity.copy(), "examples": [{"source_row": 5, "prediction": "good", "batch_seconds": 1}]}
        verify_partial(partial, identity, rows)
        changed = copy.deepcopy(partial)
        changed["identity"]["beams"] = 1
        with self.assertRaises(ValueError):
            verify_partial(changed, identity, rows)
        partial["examples"][0]["source_row"] = 9
        with self.assertRaises(ValueError):
            verify_partial(partial, identity, rows)

    def test_frozen_protocol_integrity(self):
        from comparison_data import load_protocol, source_hashes
        root = Path(__file__).resolve().parents[1]
        config, manifest, freeze = load_protocol(root / "results/t5-comparison/protocol")
        self.assertEqual(len(manifest["splits"]["fresh_test"]), config["fresh_test_examples"])
        self.assertEqual(len(manifest["detail_audit_rows"]), config["detail_audit_examples"])
        old = json.loads((root / "results/reproducible-pilot/split_manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            self.assertEqual([r["row"] for r in manifest["splits"][split]], [r["row"] for r in old["splits"][split]])


if __name__ == "__main__":
    unittest.main()
