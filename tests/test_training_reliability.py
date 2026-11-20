import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer import trainer_utils
from trainer.training_state import TokenAccumulator, resume_batch

ROOT = Path(__file__).resolve().parents[1]


class TrainingReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_masked_mean_matches_cross_entropy_and_has_finite_empty_gradient(self):
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=32, num_hidden_layers=1, vocab_size=64))
        ids = torch.tensor([[1, 12, 11, 2], [1, 24, 6, 2]])
        labels = torch.tensor([[-100, -100, 11, 2], [-100, 24, -100, 2]])
        out = model(ids, labels=labels)
        expected = F.cross_entropy(out.logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(out.loss, expected)
        for length in (1, 4):
            model.zero_grad(set_to_none=True)
            loss = model(ids[:, :length], labels=torch.full_like(ids[:, :length], -100)).loss
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(all((p.grad == 0).all() for p in model.parameters() if p.grad is not None))

    def test_sft_window_keeps_answer_and_masks_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sft.jsonl"
            rows = [
                [{"role": "system", "content": "Be brief."}, {"role": "user", "content": " Context" * 1600},
                 {"role": "assistant", "content": "The answer is cobalt."}],
                [{"role": "user", "content": "No answer supplied"}],
                [],
            ]
            path.write_text(''.join(json.dumps({"conversations": row}) + '\n' for row in rows))
            data = SFTDataset(str(path), self.tokenizer, max_length=36)
            data.epoch_seed = 15
            ids, labels = data[0]
            active = labels[1:] != -100
            self.assertEqual(len(ids), 36)
            self.assertIn("cobalt", self.tokenizer.decode(labels[1:][active]))
            self.assertEqual(ids[1:][active].tolist(), labels[1:][active].tolist())
            for index in (1, 2):
                ids, labels = data[index]
                self.assertEqual(len(ids), 36)
                self.assertTrue((labels == -100).all())

    def test_token_accumulation_matches_logical_batch_and_skips_empty_adamw(self):
        # The two constituent microbatches have different numbers of target tokens.
        model = torch.nn.Linear(1, 1, bias=False)
        control = torch.nn.Linear(1, 1, bias=False)
        control.load_state_dict(model.state_dict())
        options = dict(lr=0.02, weight_decay=0.1)
        opt = torch.optim.AdamW(model.parameters(), **options)
        opt_control = torch.optim.AdamW(control.parameters(), **options)
        args = SimpleNamespace(device="cpu", batch_size=1, max_seq_len=8, accumulation_steps=4, grad_clip=0)
        window = TokenAccumulator(opt, torch.amp.GradScaler("cpu", enabled=False), args)
        samples = torch.tensor([[1.], [3.], [-4.], [2.]])
        target = torch.tensor([[3.], [1.], [2.], [-1.]])
        for x, y in zip(samples.split([1, 3]), target.split([1, 3])):
            window.backward((model(x) - y).square().mean(), torch.tensor(len(x)))
        window.step()
        (control(samples) - target).square().mean().backward()
        opt_control.step()
        torch.testing.assert_close(model.weight, control.weight)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(opt.state[model.weight][key], opt_control.state[control.weight][key])
        before = model.weight.detach().clone()
        window.backward(model(samples).sum() * 0, torch.tensor(0))
        window.step()
        self.assertTrue(torch.equal(model.weight, before))
        self.assertEqual(opt.state[model.weight]["step"].item(), 1)

    def test_legacy_and_resized_resume_positions(self):
        self.assertEqual(resume_batch({"step": 3}, 10, 2, 2, 1), 2)
        self.assertEqual(resume_batch({"step": 5}, 10, 2, 2, 1), 5)
        state = {"step": 3, "world_size": 2, "progress": {"batch_size": 2, "completed_step": 2, "epoch_batches": 5}}
        self.assertEqual(resume_batch(state, 18, 2, 3, 1), 4)
        state["progress"]["completed_step"] = 5
        self.assertEqual(resume_batch(state, 18, 2, 3, 1), 9)

    def test_trainers_resume_from_real_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "model", root / "model", ignore=shutil.ignore_patterns("__pycache__"))
            work = root / "trainer"
            work.mkdir()
            (root / "out").mkdir()
            torch.manual_seed(19)
            config = MiniMindConfig(hidden_size=32, num_hidden_layers=1)
            torch.save(MiniMindForCausalLM(config).state_dict(), root / "out/fixture_32.pth")
            original_cwd = Path.cwd()
            try:
                os.chdir(work)
                for stage in ("pretrain", "full_sft", "lora", "dpo", "distillation", "grpo", "agent", "ppo"):
                    with self.subTest(stage=stage):
                        self.check_stage_resume(root, stage)
            finally:
                os.chdir(original_cwd)

    def check_stage_resume(self, root, stage):
        rows = []
        for i in range(5):
            question = {"role": "user", "content": f"Question {i}?"}
            chosen = [question, {"role": "assistant", "content": f"Answer {i}."}]
            if stage == "pretrain":
                rows.append({"text": f"Sample {i} " * (i + 1)})
            elif stage == "dpo":
                rows.append({"chosen": chosen, "rejected": [question, {"role": "assistant", "content": "Wrong."}]})
            else:
                rows.append({"conversations": chosen, "gt": []} if stage == "agent" else {"conversations": chosen})
        data = root / f"{stage}.jsonl"
        data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        resume = root / "checkpoints" / f"{stage}_32_resume.pth"
        real_checkpoint = trainer_utils.lm_checkpoint
        checkpoint_paths = {}

        def capture(*args, **kwargs):
            result = real_checkpoint(*args, **kwargs)
            if kwargs.get("model") is not None:
                step = kwargs["step"]
                saved = root / f"{stage}-{step}.pth"
                shutil.copyfile(resume, saved)
                checkpoint_paths[step] = saved
            return result

        script = ROOT / "trainer" / f"train_{stage}.py"
        argv = [str(script), "--epochs", "1", "--batch_size", "1", "--accumulation_steps", "2",
                "--device", "cpu", "--num_workers", "0", "--max_seq_len", "32", "--data_path", str(data),
                "--save_interval", "1", "--log_interval", "5", "--seed", "83"]
        if stage == "distillation":
            argv += ["--student_hidden_size", "32", "--teacher_hidden_size", "32", "--student_num_layers", "1",
                     "--teacher_num_layers", "1", "--teacher_use_moe", "0", "--from_student_weight", "none",
                     "--from_teacher_weight", "none", "--save_weight", stage]
        else:
            argv += ["--hidden_size", "32", "--num_hidden_layers", "1", "--from_weight", "fixture"]
            if stage == "lora":
                argv += ["--lora_name", stage]
            if stage in ("grpo", "agent", "ppo"):
                argv += ["--save_weight", stage, "--max_gen_len", "3", "--rollout_engine", "torch"]
                argv += ["--mini_batch_size", "1", "--ppo_update_iters", "3"] if stage == "ppo" else ["--num_generations", "2"]

        def run(extra):
            # The RL paths sample real tokens. A fixed local reward isolates optimizer and replay behavior.
            reward = SimpleNamespace(get_score=lambda messages, answer: sum(map(ord, answer)) % 17 / 17)
            with patch.object(sys, "argv", argv + extra), patch.object(trainer_utils, "LMForRewardModel", return_value=reward), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script), run_name="__main__")

        with patch.object(trainer_utils, "lm_checkpoint", capture):
            run([])
        baseline = torch.load(resume, map_location="cpu", weights_only=True)
        for step in (1, 2, 3, 5):
            shutil.copyfile(checkpoint_paths[step], resume)
            run(["--from_resume", "1"])
            actual = torch.load(resume, map_location="cpu", weights_only=True)
            for key, expected in baseline["model"].items():
                self.assertTrue(torch.equal(actual["model"][key], expected), (stage, step, key))
            self.assertEqual(actual["optimizer"]["param_groups"], baseline["optimizer"]["param_groups"])
            for param, state in baseline["optimizer"]["state"].items():
                for key in state:
                    self.assertTrue(torch.equal(actual["optimizer"]["state"][param][key], state[key]),
                                    (stage, step, param, key))
            for name in ("scheduler", "critic_scheduler", "critic_model", "critic_optimizer"):
                if name in baseline:
                    torch.testing.assert_close(actual[name], baseline[name], rtol=0, atol=0)
