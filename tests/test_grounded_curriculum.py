import hashlib
import json
import re
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from scripts.compare_curricula import case_bytes, evaluation_cases
from scripts.eval_belebele_slice import LANGUAGES as READING_LANGUAGES, answer_scores, prompt as reading_prompt, selected_keys
from scripts.grounded_curriculum import FIELDS, LANGUAGES, PAIRS, SPLITS, make_cases, write_datasets


class GroundedCurriculumTests(unittest.TestCase):
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
