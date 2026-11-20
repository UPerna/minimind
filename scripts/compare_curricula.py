"""Prepare the fixed multilingual evaluation, or summarize four local evaluation reports."""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_quality import matches_answer
from scripts.grounded_curriculum import FIELDS, LANGUAGES, make_cases


MODELS = ("base", "previous", "repeated", "expanded")
PREFIX = re.compile(r"^(?:(?:the (?:answer|result|value) is|(?:la risposta|il risultato|il valore) è)\s+|(?:答案|结果|数值)[是为：:]\s*)", re.IGNORECASE)


def evaluation_cases():
    cases = []
    for split in ("development", "holdout"):
        cases.extend(dict(c, split=split) for c in make_cases(split))
    for name in ("quality_cases", "transfer_cases"):
        public = json.loads((ROOT / f"evals/{name}.json").read_bytes())
        cases.extend(dict(c, split="public") for c in public)
    return cases


def case_bytes(cases):
    return (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def count(rows, name):
    return [sum(bool(r[name]) for r in rows), len(rows)]


def scores(rows, cases):
    pairs = defaultdict(list)
    for row in rows:
        case = cases[row["case"]]
        if "pair" in case:
            pairs[(case["pair"], row["language"])].append(row["correct"])
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Incomplete paired scenario")
    missing = [r for r in rows if cases[r["case"]].get("missing")]
    present = [r for r in rows if cases[r["case"]]["category"] in FIELDS and not cases[r["case"]]["missing"]]
    return {
        "exact": count(rows, "correct"), "exact_or_plain_answer_sentence": count(rows, "sentence_correct"),
        "eos": [sum(r["stop"] == "eos" for r in rows), len(rows)],
        "languages": {lang: count([r for r in rows if r["language"] == lang], "correct") for lang in LANGUAGES},
        "categories": {cat: count([r for r in rows if cases[r["case"]]["category"] == cat], "correct")
                       for cat in sorted({cases[r["case"]]["category"] for r in rows})},
        "both_variants_correct": [sum(all(pair) for pair in pairs.values()), len(pairs)],
        "missing_correct": count(missing, "correct"),
        "unnecessary_unknown": [sum(r["answer"].strip().rstrip(".。").casefold() == "unknown" for r in present), len(present)],
    }


def summarize(reports, output):
    cases = evaluation_cases()
    digest = hashlib.sha256(case_bytes(cases)).hexdigest()
    by_id = {case["id"]: case for case in cases}
    expected = {(c["id"], lang) for c in cases for lang in c["prompts"]}
    raw, metrics = {}, {"cases_sha256": digest, "models": {}, "paired_expanded_vs_repeated": {}}
    for name in MODELS:
        report = json.loads((reports / f"{name}.json").read_bytes())
        if report["cases_sha256"] != digest:
            raise ValueError(f"Question file differs for {name}")
        rows = report["results"]
        keys = {(r["case"], r["language"]) for r in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(f"Missing or duplicated items for {name}")
        for row in rows:
            case, lang = by_id[row["case"]], row["language"]
            if row["correct"] != matches_answer(row["answer"], case, lang):
                raise ValueError(f"Stored score disagrees with grader for {name} {case['id']} {lang}")
            plain = PREFIX.sub("", row["answer"].strip(), count=1)
            row["sentence_correct"] = matches_answer(plain, case, lang)
        metadata = {k: v for k, v in report.items() if k not in ("results", "summary", "cases_sha256")}
        metrics["models"][name] = {"metadata": metadata, "scores": {
            split: scores([r for r in rows if by_id[r["case"]]["split"] == split], by_id)
            for split in ("development", "holdout", "public")}}
        raw[name] = {(r["case"], r["language"]): r for r in rows}
    for split in ("development", "holdout", "public"):
        keys = [k for k in expected if by_id[k[0]]["split"] == split]
        changes = [(raw["repeated"][k]["correct"], raw["expanded"][k]["correct"]) for k in keys]
        metrics["paired_expanded_vs_repeated"][split] = {
            "gains": changes.count((False, True)), "losses": changes.count((True, False)),
            "both_correct": changes.count((True, True)), "both_wrong": changes.count((False, False))}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target, lineterminator="\n")
        columns = ["split", "case", "pair", "category", "language", "prompt", "expected"]
        writer.writerow(columns + [f"{name}_{column}" for name in MODELS for column in ("answer_json", "correct", "stop", "tokens")])
        for case in cases:
            for lang, prompt in case["prompts"].items():
                answer = case.get("number")
                if answer is None:
                    accepted = case["answers"]
                    answer = " / ".join(accepted[lang] if isinstance(accepted, dict) else accepted)
                values = [case["split"], case["id"], case.get("pair", ""), case["category"], lang, prompt, answer]
                for name in MODELS:
                    r = raw[name][(case["id"], lang)]
                    values.extend([json.dumps(r["answer"], ensure_ascii=False), int(r["correct"]), r["stop"], r["generated_tokens"]])
                writer.writerow(values)
    (output / "results.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    cases_parser = commands.add_parser("cases")
    cases_parser.add_argument("output", type=Path)
    score_parser = commands.add_parser("summarize")
    score_parser.add_argument("reports", type=Path, help="Directory containing base, previous, repeated and expanded JSON files")
    score_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "cases":
        args.output.write_bytes(case_bytes(evaluation_cases()))
    else:
        summarize(args.reports, args.output)
