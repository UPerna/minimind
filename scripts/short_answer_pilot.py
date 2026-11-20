"""Train a small multilingual short-answer adapter on the released MiniMind-3 dense model."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.model_lora import apply_lora, merge_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.convert_model import repair_generation_config


def training_rows():
    def translations(prompts, answers):
        if isinstance(answers, str):
            answers = [answers] * 3
        for prompt, answer in zip(prompts, answers):
            yield {"conversations": [{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": answer}]}

    cities = [
        ("Japan", "Giappone", "日本", "Tokyo", "Tokyo", "东京"),
        ("Spain", "Spagna", "西班牙", "Madrid", "Madrid", "马德里"),
        ("Egypt", "Egitto", "埃及", "Cairo", "Il Cairo", "开罗"),
        ("Kenya", "Kenya", "肯尼亚", "Nairobi", "Nairobi", "内罗毕"),
        ("Canada", "Canada", "加拿大", "Ottawa", "Ottawa", "渥太华"),
        ("Greece", "Grecia", "希腊", "Athens", "Atene", "雅典"),
    ]
    for en, it, zh, *answers in cities:
        yield from translations((f"Name the capital city of this country. Country: {en}. Give just its name.",
                                 f"Scrivi soltanto il nome della capitale del seguente paese: {it}.",
                                 f"写出这个国家的首都名称：{zh}。只写名称。"), answers)
    for code in ("TUMA", "SELK", "HAFI", "NURO", "KIMU", "PATO"):
        yield from translations((f"Read the tag [ID={code}]. Return just the ID.",
                                 f"Leggi il cartellino [ID={code}]. Restituisci soltanto l'ID.",
                                 f"读取标签[ID={code}]，只写出ID。"), code)
    for a, b in ((14, 5), (33, 7), (62, 21), (87, 32), (24, 9), (53, 18)):
        yield from translations((f"Give only the result of {a} - {b}.",
                                 f"Scrivi soltanto il risultato di {a} - {b}.",
                                 f"只写出{a}减{b}的结果。"), str(a - b))
    for kg in (1, 2, 3, 4, 5, 6):
        yield from translations((f"Convert {kg} kg to grams. Give just the number.",
                                 f"Converti {kg} kg in grammi. Scrivi soltanto il numero.",
                                 f"把{kg}千克换算成克。只写数字。"), str(1000 * kg))
    for a, b in ((8, 4), (12, 3), (18, 6), (35, 7), (42, 7), (63, 9)):
        yield from translations((f"Divide {a} by {b}. Give just the number.",
                                 f"Dividi {a} per {b}. Scrivi soltanto il numero.",
                                 f"把{a}除以{b}。只写数字。"), str(a // b))


def run(base, parent, device, threads, *, train_file=None, max_seq_len=96):
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float16)
    if hf.config.model_type != "qwen3":
        raise ValueError("This pilot uses the standard Qwen3 export of the dense MiniMind-3 release")
    native = MiniMindForCausalLM(MiniMindConfig())
    native.load_state_dict(hf.state_dict(), strict=True)
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="short-answer-", dir=parent)).resolve()
    print(f"Experiment directory: {work}", flush=True)
    (work / "out").mkdir()
    (work / "trainer").mkdir()
    tokenizer.save_pretrained(work / "model")
    torch.save(hf.state_dict(), work / "out/full_sft_768.pth")
    data = work / "train.jsonl"
    if train_file is None:
        data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in training_rows()), encoding="utf-8")
    else:
        data.write_bytes(train_file.read_bytes())
    command = [sys.executable, str(ROOT / "trainer/train_lora.py"), "--device", device,
               "--data_path", str(data), "--from_weight", "full_sft", "--lora_name", "short_answer",
               "--lora_rank", "8", "--lora_alpha", "16", "--epochs", "3", "--batch_size", "4",
               "--max_seq_len", str(max_seq_len), "--learning_rate", "0.001", "--num_workers", "0",
               "--log_interval", "10", "--save_interval", "10000"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    subprocess.run(command, cwd=work / "trainer", env=env, check=True)
    apply_lora(native, rank=8, lora_alpha=16)
    merged = work / "out/merged.pth"
    merge_lora(native, work / "out/short_answer_768.pth", merged)
    hf.load_state_dict(torch.load(merged, map_location="cpu", weights_only=True), strict=True)
    export = work / "export"
    hf.save_pretrained(export)
    tokenizer.save_pretrained(export)
    repair_generation_config(export)
    print(f"Adapter: {work / 'out/short_answer_768.pth'}\nTransformers export: {export}", flush=True)
    return export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Local standard dense MiniMind-3 release")
    parser.add_argument("--work-dir", required=True, type=Path, help="Parent directory for a new experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-file", type=Path, help="Optional chat-format JSONL curriculum")
    parser.add_argument("--max-seq-len", type=int, default=96)
    args = parser.parse_args()
    if args.threads < 1 or args.max_seq_len < 2:
        parser.error("--threads must be positive and --max-seq-len must be at least two")
    torch.set_num_threads(args.threads)
    run(args.base, args.work_dir, args.device, args.threads, train_file=args.train_file, max_seq_len=args.max_seq_len)
