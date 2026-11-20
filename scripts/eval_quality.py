"""Run matched multilingual prompts against a local Transformers export."""

import argparse
import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CASES = Path(__file__).resolve().parents[1] / "evals" / "quality_cases.json"


def matches_answer(reply, case, language):
    answer = unicodedata.normalize("NFKC", reply).strip().removesuffix("。").removesuffix(".").strip()
    if "number" in case:
        if language == "it":
            answer = answer.replace(",", ".")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", answer, re.ASCII):
            return False
        return Decimal(answer) == Decimal(case["number"])
    accepted = case["answers"]
    if isinstance(accepted, dict):
        accepted = accepted[language]
    if case.get("case_sensitive"):
        return answer in accepted
    return answer.casefold() in {item.casefold() for item in accepted}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(model_dir, cases_path, limit, device):
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).to(device).eval()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    rows = []
    totals = {}
    with torch.inference_mode():
        for case in cases:
            for language, prompt in case["prompts"].items():
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, open_thinking=False,
                    return_dict=True, return_tensors="pt",
                ).to(device)
                output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                        do_sample=False, max_new_tokens=limit,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
                tokens = output[0, inputs.input_ids.shape[1]:].tolist()
                ended = bool(tokens and tokens[-1] in eos_ids)
                answer = tokenizer.decode(tokens[:-1] if ended else tokens, skip_special_tokens=False)
                passed = matches_answer(answer, case, language)
                summary = totals.setdefault(language, {"correct": 0, "total": 0, "ended_with_eos": 0})
                summary["correct"] += int(passed)
                summary["total"] += 1
                summary["ended_with_eos"] += int(ended)
                rows.append({"case": case["id"], "language": language, "answer": answer, "correct": passed,
                             "generated_tokens": len(tokens), "stop": "eos" if ended else "token_limit"})
    names = ("*.safetensors", "pytorch_model*.bin", "*.index.json", "config.json", "generation_config.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "*.jinja")
    files = sorted({path for name in names for path in model_dir.glob(name)})
    return {"cases_sha256": file_sha256(cases_path), "model_files_sha256": {p.name: file_sha256(p) for p in files},
            "torch": torch.__version__, "transformers": transformers.__version__, "device": device,
            "dtype": "float32", "max_new_tokens": limit, "greedy": True, "open_thinking": False,
            "summary": totals, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local standard Transformers model directory")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", required=True, type=Path, help="Write the scores and verbatim answers to JSON")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, mps or cuda:0")
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads")
    args = parser.parse_args()
    if min(args.threads, args.max_new_tokens) < 1:
        parser.error("threads and max-new-tokens must be positive")
    torch.set_num_threads(args.threads)
    report = evaluate(args.model, args.cases, args.max_new_tokens, args.device)
    report["threads"] = args.threads
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
