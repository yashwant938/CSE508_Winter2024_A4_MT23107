"""Validate a source-grounded review and compute explicit audit denominators."""
import argparse
from collections import Counter
from pathlib import Path

from comparison_data import read_json
from train import sha256_file
from training_data import digest, write_json


def compile_annotations(result, facts, annotations):
    """Expand the explicitly authored labels; this does not infer or auto-grade facts."""
    by_row = {x["source_row"]: x for x in result["examples"]}
    systems = {"t": "t5_beam1", "g": "gpt2", "l": "lead_16_words"}
    expected = {str(x["source_row"]) for x in facts["examples"]}
    if set(annotations["rows"]) != expected:
        raise ValueError("Annotation row set differs from the frozen audit sample")
    reviews = []
    for source in facts["examples"]:
        row = source["source_row"]
        authored = annotations["rows"][str(row)]
        if set(authored) != set(systems):
            raise ValueError("Missing or unexpected annotated system")
        for code, system in systems.items():
            covered, reason, claims = authored[code]
            reviews.append({"source_row": row, "system": system,
                "prediction_sha256": digest(by_row[row]["predictions"][system]),
                "facts_preserved": covered, "coverage_rationale": reason,
                "claims": [{"claim": c, "verdict": v, "evidence": e, "reason": r} for c, v, e, r in claims]})
    audit = {"reviewer": facts["reviewer"], "reviews": reviews}
    score_audit(result, facts, audit)
    return audit


def score_audit(result, facts, audit):
    systems = set(result["scores"])
    expected = {(x["source_row"], system) for x in facts["examples"] for system in systems}
    actual = [(x["source_row"], x["system"]) for x in audit["reviews"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Audit must cover every prescribed review/system pair exactly once")
    by_row = {x["source_row"]: x for x in result["examples"]}
    summaries = {system: Counter() for system in systems}
    for item in audit["reviews"]:
        row = by_row[item["source_row"]]
        prediction = row["predictions"][item["system"]]
        if item["prediction_sha256"] != digest(prediction):
            raise ValueError("Reviewed prediction changed")
        if len(item["facts_preserved"]) != 2 or any(type(x) is not bool for x in item["facts_preserved"]):
            raise ValueError("Two boolean fact-coverage labels required")
        if not item.get("coverage_rationale", "").strip():
            raise ValueError("Coverage requires an explanation")
        claims = item["claims"]
        if bool(prediction.strip()) != bool(claims):
            raise ValueError("Every nonempty summary requires a claim review")
        counts = Counter()
        for claim in claims:
            verdict = claim["verdict"]
            if verdict not in ("supported", "unsupported", "contradicted", "uncertain"):
                raise ValueError("Unknown support label")
            if not claim["claim"].strip() or not claim["reason"].strip():
                raise ValueError("Every claim needs text and rationale")
            if verdict == "supported" and (not claim.get("evidence") or claim["evidence"] not in row["visible_review"]):
                raise ValueError("Supported claim needs an exact visible-source quote")
            counts[verdict] += 1
        counter = summaries[item["system"]]
        counter["summaries"] += 1
        counter["selected_facts"] += 2
        counter["facts_preserved"] += sum(item["facts_preserved"])
        counter["both_facts_preserved"] += all(item["facts_preserved"])
        counter["at_least_one_fact_preserved"] += any(item["facts_preserved"])
        counter["fully_supported_summaries"] += bool(claims) and counts["supported"] == len(claims)
        counter["summaries_with_unsupported_or_contradicted_claim"] += bool(counts["unsupported"] + counts["contradicted"])
        for label in ("supported", "unsupported", "contradicted", "uncertain"):
            counter[label + "_claims"] += counts[label]
        counter["claims"] += len(claims)
    return {system: {**dict(counter), "fact_preservation_rate": counter["facts_preserved"] / counter["selected_facts"],
        "fully_supported_summary_rate": counter["fully_supported_summaries"] / counter["summaries"]}
        for system, counter in sorted(summaries.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/t5-comparison/fresh_test.json")
    p.add_argument("--facts", default="results/t5-comparison/detail_facts.json")
    p.add_argument("--audit", default="results/t5-comparison/detail_audit.json")
    p.add_argument("--output", default="results/t5-comparison/detail_scores.json")
    p.add_argument("--compile-annotations", help="Expand explicitly authored labels into the review artifact")
    args = p.parse_args()
    if args.compile_annotations:
        if Path(args.audit).exists():
            p.error("Existing review is preserved; use a new audit path")
        write_json(args.audit, compile_annotations(read_json(args.results), read_json(args.facts), read_json(args.compile_annotations)))
    result, facts, audit = map(read_json, (args.results, args.facts, args.audit))
    if sha256_file(args.facts) != result["identity"]["facts_sha256"]:
        raise ValueError("Selected facts changed after final test began")
    if audit["reviewer"] != facts["reviewer"]:
        raise ValueError("Reviewer provenance mismatch")
    write_json(args.output, {"reviewer": audit["reviewer"], "sample_size": len(facts["examples"]),
        "input_sha256": {name: sha256_file(path) for name, path in (("results", args.results), ("facts", args.facts), ("audit", args.audit))},
        "scores": score_audit(result, facts, audit),
        "limitations": ["Single agent, non-independent and non-blind; small descriptive sample.",
                        "Source support is not independently verified product truth.",
                        "Generic supported statements can still omit the selected details."]})
    print(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
