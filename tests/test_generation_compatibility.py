import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts import convert_model
from scripts.eval_quality import DEFAULT_CASES, file_sha256, matches_answer
from scripts.short_answer_pilot import training_rows


class GenerationCompatibilityTests(unittest.TestCase):
    def test_dense_and_moe_exports_stop_on_eos_and_match_native_logits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for use_moe in (False, True):
                with self.subTest(moe=use_moe):
                    torch.manual_seed(31)
                    config = MiniMindConfig(hidden_size=32, num_hidden_layers=1, max_position_embeddings=128,
                                            use_moe=use_moe)
                    native = MiniMindForCausalLM(config).eval()
                    weights, export = root / "native.pth", root / ("moe" if use_moe else "dense")
                    torch.save(native.state_dict(), weights)
                    with patch.object(convert_model, "lm_config", config, create=True), contextlib.redirect_stdout(io.StringIO()):
                        convert_model.convert_torch2transformers(weights, export, torch.float32)
                    hf = AutoModelForCausalLM.from_pretrained(export, local_files_only=True).eval()
                    self.assertEqual([hf.generation_config.bos_token_id, hf.generation_config.eos_token_id,
                                      hf.generation_config.pad_token_id], [1, 2, 0])
                    ids = torch.tensor([[1, 5, 9]])
                    with torch.no_grad():
                        torch.testing.assert_close(hf(ids).logits, native(ids).logits, rtol=1e-5, atol=1e-6)
                    generated = hf.generate(ids, do_sample=False, max_new_tokens=4,
                                            prefix_allowed_tokens_fn=lambda *args: [2])
                    self.assertEqual(generated[0, ids.size(1):].tolist(), [2])

    def test_legacy_config_repair_preserves_explicit_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (root / name).write_bytes((convert_model.TOKENIZER_PATH / name).read_bytes())
            (root / "config.json").write_text('{"model_type":"qwen3","bos_token_id":13}')
            path = root / "generation_config.json"
            path.write_text('{"eos_token_id":[2,19],"temperature":0.78}')
            path.chmod(0o640)
            self.assertEqual(convert_model.repair_generation_config(root), path)
            saved = path.read_bytes()
            self.assertEqual(json.loads(saved), {"eos_token_id": [2, 19], "bos_token_id": 13,
                                                "pad_token_id": 0, "temperature": 0.78})
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            convert_model.repair_generation_config(root)
            self.assertEqual(saved, path.read_bytes())
            path.unlink()
            convert_model.repair_generation_config(root)
            self.assertEqual(json.loads(path.read_bytes())["eos_token_id"], 2)

    def test_diagnostic_grader_rejects_wrong_and_ambiguous_answers(self):
        case = {"answers": {"en": ["Rome"]}}
        self.assertTrue(matches_answer("rome.", case, "en"))
        for text in ("It is not Rome.", "Rome or Milan", "Romero", ""):
            self.assertFalse(matches_answer(text, case, "en"))
        self.assertTrue(matches_answer("1,20", {"number": "1.2"}, "it"))
        self.assertFalse(matches_answer("1,20", {"number": "1.2"}, "en"))
        self.assertFalse(matches_answer("1.2 or 8", {"number": "1.2"}, "en"))
        self.assertTrue(matches_answer("H₂O。", {"answers": ["H2O"], "case_sensitive": True}, "zh"))
        self.assertFalse(matches_answer("veko", {"answers": ["VEKO"], "case_sensitive": True}, "it"))

    def test_published_results_regrade_and_use_separate_questions(self):
        questions = {row["conversations"][0]["content"] for row in training_rows()}
        self.assertEqual(len(questions), 90)
        for case_file, reports in ((DEFAULT_CASES, ["minimind3-dense-baseline", "short-answer-pilot"]),
                                   (DEFAULT_CASES.with_name("transfer_cases.json"), ["minimind3-dense-transfer", "short-answer-transfer"])):
            cases = {c["id"]: c for c in json.loads(case_file.read_bytes())}
            expected_items = {(c["id"], lang) for c in cases.values() for lang in c["prompts"]}
            for case in cases.values():
                self.assertTrue(questions.isdisjoint(case["prompts"].values()))
            for name in reports:
                report = json.loads(case_file.with_name(name + ".json").read_bytes())
                self.assertEqual(report["cases_sha256"], file_sha256(case_file))
                self.assertEqual(len(report["results"]), len(expected_items))
                self.assertEqual({(r["case"], r["language"]) for r in report["results"]}, expected_items)
                for row in report["results"]:
                    self.assertEqual(row["correct"], matches_answer(row["answer"], cases[row["case"]], row["language"]))
                for language, summary in report["summary"].items():
                    rows = [r for r in report["results"] if r["language"] == language]
                    self.assertEqual(summary, {"correct": sum(r["correct"] for r in rows), "total": len(rows),
                                               "ended_with_eos": sum(r["stop"] == "eos" for r in rows)})
