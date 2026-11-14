"""Audit public experimental evidence offline; never load pretrained weights."""
import copy
from pathlib import Path
import unittest

from comparison_data import read_json, source_hashes
from evaluate import rouge_scores
from score_detail_audit import compile_annotations, score_audit
from train import sha256_file
from training_data import digest

ROOT = Path(__file__).resolve().parents[1] / "results/t5-comparison"


class RecordedComparisonTests(unittest.TestCase):
    def test_selection_and_test_lock_are_consistent(self):
        from compare_summarizers import SOURCE_NAMES
        validation = read_json(ROOT / "validation_selection.json")
        result = read_json(ROOT / "fresh_test.json")
        self.assertEqual(result["identity"], read_json(ROOT / "test_lock.json"))
        self.assertEqual(result["identity"]["selection_sha256"], sha256_file(ROOT / "validation_selection.json"))
        self.assertEqual(result["identity"]["facts_sha256"], sha256_file(ROOT / "detail_facts.json"))
        for key in ("protocol", "checkpoints", "source_sha256", "device", "environment"):
            self.assertEqual(result["identity"][key], validation["identity"][key])
        self.assertEqual(result["identity"]["source_sha256"], source_hashes(SOURCE_NAMES))
        selected = min((1, 4), key=lambda b: (-validation["scores"][f"t5_beam{b}"]["rougeL"]["mean"], b))
        self.assertEqual(selected, validation["selected_beams"])

    def test_scores_recompute_and_every_fresh_row_is_present(self):
        result = read_json(ROOT / "fresh_test.json")
        manifest = read_json(ROOT / "protocol/manifest.json")
        self.assertEqual([r["source_row"] for r in result["examples"]], [r["row"] for r in manifest["splits"]["fresh_test"]])
        self.assertEqual(result["sample_size"], len(result["examples"]))
        for system, recorded in result["scores"].items():
            scores = []
            for example in result["examples"]:
                computed = rouge_scores(example["predictions"][system], example["reference"])
                self.assertEqual(computed, example["scores"][system])
                scores.append(computed)
            for metric in ("rouge1", "rougeL"):
                self.assertAlmostEqual(sum(x[metric] for x in scores) / len(scores), recorded[metric]["mean"], places=12)

    def test_recorded_fact_labels_and_denominators_recompute(self):
        result = read_json(ROOT / "fresh_test.json")
        facts = read_json(ROOT / "detail_facts.json")
        audit = read_json(ROOT / "detail_audit.json")
        scores = read_json(ROOT / "detail_scores.json")
        self.assertEqual(compile_annotations(result, facts, read_json(ROOT / "audit_annotations.json")), audit)
        self.assertEqual(score_audit(result, facts, audit), scores["scores"])
        for key, filename in (("results", "fresh_test.json"), ("facts", "detail_facts.json"), ("audit", "detail_audit.json")):
            self.assertEqual(scores["input_sha256"][key], sha256_file(ROOT / filename))
        by_row = {r["source_row"]: r for r in result["examples"]}
        for row in facts["examples"]:
            for fact in row["facts"]:
                self.assertIn(fact["evidence"], by_row[row["source_row"]]["visible_review"])

    def test_audit_rejects_missing_duplicate_and_changed_predictions(self):
        result, facts, audit = [read_json(ROOT / name) for name in ("fresh_test.json", "detail_facts.json", "detail_audit.json")]
        bad = copy.deepcopy(audit)
        bad["reviews"].pop()
        with self.assertRaises(ValueError):
            score_audit(result, facts, bad)
        bad = copy.deepcopy(audit)
        bad["reviews"].append(bad["reviews"][0])
        with self.assertRaises(ValueError):
            score_audit(result, facts, bad)
        bad = copy.deepcopy(audit)
        bad["reviews"][0]["prediction_sha256"] = digest("changed")
        with self.assertRaises(ValueError):
            score_audit(result, facts, bad)

    def test_supported_generic_output_is_not_detail_coverage(self):
        result = {"scores": {"model": {}}, "examples": [{"source_row": 1, "visible_review": "good food with no salt",
                  "predictions": {"model": "Good food"}}]}
        facts = {"examples": [{"source_row": 1}]}
        item = {"source_row": 1, "system": "model", "prediction_sha256": digest("Good food"),
                "facts_preserved": [False, False], "coverage_rationale": "Generic praise misses the selected facts.",
                "claims": [{"claim": "Food is good", "verdict": "supported", "evidence": "good food", "reason": "Explicit praise."}]}
        score = score_audit(result, facts, {"reviews": [item]})["model"]
        self.assertEqual(score["fully_supported_summaries"], 1)
        self.assertEqual(score["facts_preserved"], 0)
        item["claims"][0]["evidence"] = "invented source quote"
        with self.assertRaises(ValueError):
            score_audit(result, facts, {"reviews": [item]})


if __name__ == "__main__":
    unittest.main()
