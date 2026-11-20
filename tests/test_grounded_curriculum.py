import csv
import hashlib
import json
import re
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from model.model_lora import apply_lora, load_lora, merge_lora, save_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.compare_curricula import case_bytes, evaluation_cases
from scripts.eval_belebele_slice import LANGUAGES as READING_LANGUAGES, answer_scores, prompt as reading_prompt, selected_keys
from scripts.eval_quality import matches_answer
from scripts.grounded_curriculum import FIELDS, LANGUAGES, PAIRS, SPLITS, make_cases, write_datasets


class GroundedCurriculumTests(unittest.TestCase):
    def test_pilot_adapter_formats_produce_identical_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1))
            apply_lora(model, rank=8, lora_alpha=16)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith("lora.B.weight"):
                        parameter.fill_(0.05)
            for ext in ("pth", "safetensors"):
                save_lora(model, root / f"adapter.{ext}")
            for ext in ("pth", "safetensors"):
                merge_lora(model, root / f"adapter.{ext}", root / f"merged-{ext}.pth", strict=True)
            original = torch.load(root / "merged-pth.pth", weights_only=True)
            restored = torch.load(root / "merged-safetensors.pth", weights_only=True)
            torch.testing.assert_close(restored, original, rtol=0, atol=0)
            torch.save({}, root / "incomplete.pth")
            with self.assertRaises(ValueError):
                load_lora(model, root / "incomplete.pth", strict=True)

    def test_published_scores_against_answers(self):
        evidence = Path(__file__).resolve().parents[1] / "evals" / "grounded-v1"
        metrics = json.loads((evidence / "results.json").read_bytes())["models"]
        cases = {case["id"]: case for case in evaluation_cases()}
        with (evidence / "predictions.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        expected = {(c["id"], lang) for c in cases.values() for lang in c["prompts"]}
        self.assertEqual(len(rows), len(expected))
        self.assertEqual({(r["case"], r["language"]) for r in rows}, expected)
        for name, model in metrics.items():
            for split, reported in model["scores"].items():
                group = [r for r in rows if r["split"] == split]
                correct = []
                for row in group:
                    case, lang = cases[row["case"]], row["language"]
                    self.assertEqual(row["prompt"], case["prompts"][lang])
                    passed = matches_answer(json.loads(row[name + "_answer_json"]), case, lang)
                    self.assertEqual(int(passed), int(row[name + "_correct"]))
                    correct.append(passed)
                self.assertEqual(reported["exact"], [sum(correct), len(group)])

    def test_reading_evidence_uses_same_items_and_disjoint_followup(self):
        folder = Path(__file__).resolve().parents[1] / "evals" / "grounded-v1" / "reading"
        item_sets = {}
        for path in folder.glob("*.json"):
            report = json.loads(path.read_bytes())
            rows = report["results"]
            keys = {(r["link"], r["question_number"], r["language"], r["gold"]) for r in rows}
            self.assertEqual((len(rows), len(keys), len({r["link"] for r in rows})), (72, 72, 24))
            likelihood = "likelihood" in path.stem
            self.assertEqual(item_sets.setdefault(likelihood, keys), keys)
            for lang, totals in report["summary"].items():
                group = rows if lang == "all" else [r for r in rows if r["language"] == lang]
                choices = (("with_passage", "passage"), ("question_only", "question_only")) if likelihood else (
                    ("original_correct", "original"), ("rotated_correct", "rotated"))
                for metric, condition in choices:
                    self.assertEqual(totals[metric], sum(r[condition] == r["gold"] for r in group))
                if likelihood:
                    self.assertEqual(totals["with_passage"] - totals["question_only"],
                                     totals["gained_with_passage"] - totals["lost_with_passage"])
                else:
                    self.assertEqual(totals["both_correct"], sum(r["original"] == r["rotated"] == r["gold"] for r in group))
        self.assertTrue({k[0] for k in item_sets[False]}.isdisjoint(k[0] for k in item_sets[True]))

    def test_conditional_answer_scores_match_unpadded_cross_entropy(self):
        tokenizer_path = Path(__file__).resolve().parents[1] / "model"
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tok.padding_side = "left"
        config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128)
        model = Qwen3ForCausalLM(config).eval()
        batch = [([1, 5, 7, 9], [20, 74]), ([1, 12], [11, 24, 31])]
        observed = answer_scores(model, tok, batch, "cpu")
        with torch.no_grad():
            for score, (prefix, answer) in zip(observed, batch):
                logits = model(torch.tensor([prefix + answer])).logits[0, len(prefix)-1:-1]
                expected = -torch.nn.functional.cross_entropy(logits, torch.tensor(answer)).item()
                self.assertAlmostEqual(score, expected, places=5)

    def test_reading_items_are_balanced_and_rotation_changes_positions(self):
        data = {}
        for lang in READING_LANGUAGES:
            data[lang] = {(f"source-{i}", q): {"correct_answer_num": str(1 + i % 4),
                          "flores_passage": f"Passage {i}"} for i in range(70) for q in (1, 2)}
        keys = selected_keys(data)
        self.assertEqual(len({key[0] for key in keys}), 24)
        new_keys = selected_keys(data, "minimind-likelihood-v1", excluded=keys)
        self.assertTrue({k[0] for k in keys}.isdisjoint(k[0] for k in new_keys))
        for i in range(1, 5):
            self.assertEqual(sum(data["en"][key]["correct_answer_num"] == str(i) for key in keys), 6)
        row = {"flores_passage": "Text", "question": "Which?", **{f"mc_answer{i}": str(i) for i in range(1, 5)}}
        self.assertEqual(reading_prompt(row, "en", 1).splitlines()[-4:], ["A. 2", "B. 3", "C. 4", "D. 1"])

    def test_frozen_files_and_equal_budget(self):
        expected_hashes = {
            "expanded.jsonl": "976fdd8e5a799188cf7790c96e06d070d6fa5d9c48e897031947bb5c0ad4270c",
            "repeated.jsonl": "4845e4898ffb3a8a289fd8fc0a75f4a8ac66f3b6f1df5d711f5ff192ef830f15",
            "development.json": "78f8b7e141dd4351e837b49bf6d1f580aa85bf52cbe6296113c06ea3bf174537",
            "holdout.json": "7ced28709f479b6d8a342189cead662e335b135f5f0e0bd1ba95bddf1ae83606",
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_datasets(folder)
            for name, expected in expected_hashes.items():
                self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)
            for name, unique in (("expanded", 432), ("repeated", 90)):
                rows = (folder / (name + ".jsonl")).read_text(encoding="utf-8").splitlines()
                self.assertEqual((len(rows), len(set(rows))), (432, unique))
                training = {json.loads(r)["conversations"][0]["content"] for r in rows}
                evaluation = {p for case in evaluation_cases() for p in case["prompts"].values()}
                self.assertTrue(training.isdisjoint(evaluation))
        self.assertEqual(hashlib.sha256(case_bytes(evaluation_cases())).hexdigest(),
                         "18c06d291494c8f36d5191a4560bfe7ece69ab09fa812a063e27feafb22b0083")

    def test_targets_against_rendered_records_and_arithmetic(self):
        for split in SPLITS:
            cases = make_cases(split)
            self.assertEqual(len(cases), PAIRS[split] * 6 * 2)
            for first, second in zip(cases[::2], cases[1::2]):
                self.assertEqual(first["pair"], second["pair"])
                self.assertNotEqual(first.get("answers", first.get("number")), second.get("answers", second.get("number")))
            for case in cases:
                category = case["category"]
                if category in FIELDS:
                    for j, lang in enumerate(LANGUAGES):
                        prompt = case["prompts"][lang]
                        block = re.search(r"[\[【](.*?)[\]】]", prompt)
                        rows = dict(pair.split(" = ") for pair in block.group(1).split(" | "))
                        request = prompt[:block.start()] + prompt[block.end():]
                        fields = [names[j] for names in FIELDS[category] if names[j] in request]
                        self.assertEqual(len(fields), 1)
                        self.assertEqual(case["answers"][lang], [rows.get(fields[0], "UNKNOWN")])
                    continue
                text = case["prompts"]["en"]
                values = [Fraction(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
                if category == "subtraction":
                    a, b = values
                    expected = b - a if "from" in text else a - b
                elif category == "coverage":
                    a, b = values
                    expected = b / a if text.startswith(("A project pays", "Debt payments total")) else a / b
                else:
                    source, target = re.findall(r"millilitres|kilograms|grams|litres", text)
                    if text.startswith("How many"):
                        source, target = target, source
                    expected = values[0] * (1000 if source in ("kilograms", "litres") else Fraction(1, 1000))
                self.assertEqual(Fraction(case["number"]), expected, case["id"])
