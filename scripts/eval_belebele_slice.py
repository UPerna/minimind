"""Run a passage-matched, option-order check on a fixed slice of the Belebele test set."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_quality import file_sha256


REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
LANGUAGES = {
    "en": ("eng_Latn", "15af884c2b5994fdc71199e79d553a1864b08ce1b31894aa87f6be925a064a31"),
    "it": ("ita_Latn", "008c504767f4f2eaea6938a5ec75a617634cea8386b64ebfb16ee7b9a12aba27"),
    "zh": ("zho_Hans", "f3b7915eaa27e44225979ffd38708c5beee63d28bea27a91253e5fe68e2fa147"),
}
INSTRUCTION = {
    "en": "Read the passage and choose the answer to the question. Give only its letter.",
    "it": "Leggi il brano e scegli la risposta alla domanda. Scrivi soltanto la lettera.",
    "zh": "阅读短文并选择问题的正确答案。只写选项字母。",
}
LETTERS = "ABCD"


def read_data(directory):
    data = {}
    for lang, (code, digest) in LANGUAGES.items():
        path = directory / (code + ".jsonl")
        if file_sha256(path) != digest:
            raise ValueError(f"{path} differs from the pinned Belebele release")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        data[lang] = {(r["link"], r["question_number"]): r for r in rows}
    return data


def selected_keys(data, salt="minimind-reading-v1", excluded=()):
    common = set.intersection(*(set(rows) for rows in data.values()))
    def order(key):
        return hashlib.sha256(f"{salt}|{key[0]}|{key[1]}".encode()).digest()
    selected, counts = [], Counter()
    links = {key[0] for key in excluded}
    passages = {data["en"][key]["flores_passage"] for key in excluded}
    for key in sorted(common, key=order):
        golds = {int(rows[key]["correct_answer_num"]) - 1 for rows in data.values()}
        passage = data["en"][key]["flores_passage"]
        if len(golds) != 1 or key[0] in links or passage in passages:
            continue
        gold = golds.pop()
        if counts[gold] == 6:
            continue
        selected.append(key)
        links.add(key[0])
        passages.add(passage)
        counts[gold] += 1
        if len(selected) == 24:
            break
    if len(selected) != 24:
        raise ValueError("Could not find six distinct passages per gold position")
    return selected


def prompt(row, language, rotation):
    choices = [f"{label}. {row['mc_answer' + str(1 + (i + rotation) % 4)]}" for i, label in enumerate(LETTERS)]
    return "\n\n".join((INSTRUCTION[language], row["flores_passage"], row["question"], "\n".join(choices)))


@torch.inference_mode()
def answer_scores(model, tokenizer, batch, device):
    """Mean next-token log probability of each answer, excluding prompt and padding."""
    inputs = tokenizer.pad([{"input_ids": prefix + answer} for prefix, answer in batch],
                           padding=True, return_tensors="pt").to(device)
    mask = inputs.attention_mask
    keep = max(len(answer) for _, answer in batch) + 1
    logits = model(inputs.input_ids, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp_min(0),
                   use_cache=False, logits_to_keep=keep).logits
    scores = []
    for row, (_, answer) in enumerate(batch):
        if not answer:
            raise ValueError("Empty answer candidate")
        n = len(answer)
        scores.append(logits[row, -n-1:-1].log_softmax(-1).gather(1, inputs.input_ids[row, -n:, None]).mean().item())
    return scores


def likelihood_check(base, data_dir, device, batch_size):
    data = read_data(data_dir)
    keys = selected_keys(data, "minimind-likelihood-v1", excluded=selected_keys(data))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float32).to(device).eval()
    items = []
    for key in keys:
        for lang in LANGUAGES:
            row = data[lang][key]
            for condition in ("passage", "question_only"):
                text = row["question"]
                if condition == "passage":
                    text = row["flores_passage"] + "\n\n" + text
                prefix = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                      add_generation_prompt=True, open_thinking=False)
                for option in range(4):
                    answer = tokenizer.encode(row[f"mc_answer{option + 1}"], add_special_tokens=False)
                    items.append(((key, lang, condition, option), (prefix, answer)))
    items.sort(key=lambda item: sum(map(len, item[1])))
    likelihoods = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        scores = answer_scores(model, tokenizer, [seq for _, seq in batch], device)
        likelihoods.update((item[0], score) for item, score in zip(batch, scores))
    rows = []
    for key in keys:
        for lang in LANGUAGES:
            gold = int(data[lang][key]["correct_answer_num"]) - 1
            row = {"link": key[0], "question_number": key[1], "language": lang, "gold": LETTERS[gold]}
            for condition in ("passage", "question_only"):
                scores = [likelihoods[key, lang, condition, option] for option in range(4)]
                row[condition] = LETTERS[max(range(4), key=scores.__getitem__)]
                row[condition + "_mean_logp"] = [round(x, 6) for x in scores]
            rows.append(row)
    groups = {}
    for lang in (*LANGUAGES, "all"):
        group = rows if lang == "all" else [r for r in rows if r["language"] == lang]
        groups[lang] = {"items": len(group), "with_passage": sum(r["passage"] == r["gold"] for r in group),
                        "question_only": sum(r["question_only"] == r["gold"] for r in group),
                        "gained_with_passage": sum(r["passage"] == r["gold"] != r["question_only"] for r in group),
                        "lost_with_passage": sum(r["question_only"] == r["gold"] != r["passage"] for r in group)}
    return {"dataset": "facebook/belebele", "revision": REVISION, "weights_sha256": file_sha256(base / "model.safetensors"),
            "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32",
            "device": device, "threads": torch.get_num_threads(), "batch_size": batch_size,
            "scoring": "mean_log_probability_of_answer_tokens", "summary": groups, "results": rows}


def evaluate(base, data_dir, device, batch_size):
    data = read_data(data_dir)
    keys = selected_keys(data)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.padding_side = "left"
    letters = [tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(ids) != 1 for ids in letters) or len({ids[0] for ids in letters}) != 4:
        raise ValueError("This scoring protocol requires four distinct single-token option letters")
    letter_ids = [ids[0] for ids in letters]
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.float32).to(device).eval()
    items = [(key, lang, rot, prompt(data[lang][key], lang, rot)) for key in keys for lang in LANGUAGES for rot in (0, 1)]
    items.sort(key=lambda item: len(item[-1]))
    predictions = {}
    displayed = Counter()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            messages = [[{"role": "user", "content": item[-1]}] for item in batch]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, open_thinking=False,
                                                  padding=True, return_dict=True, return_tensors="pt").to(device)
            mask = inputs.attention_mask
            position = (mask.cumsum(1) - 1).clamp_min(0)
            logits = model(inputs.input_ids, attention_mask=mask, position_ids=position,
                           use_cache=False, logits_to_keep=1).logits[:, -1, letter_ids]
            for (key, lang, rot, _), selected in zip(batch, logits.argmax(1).tolist()):
                displayed[LETTERS[selected]] += 1
                predictions[key, lang, rot] = (selected + rot) % 4
    rows = []
    for key in keys:
        for lang in LANGUAGES:
            gold = int(data[lang][key]["correct_answer_num"]) - 1
            rows.append({"link": key[0], "question_number": key[1], "language": lang, "gold": LETTERS[gold],
                         "original": LETTERS[predictions[key, lang, 0]], "rotated": LETTERS[predictions[key, lang, 1]]})
    results = {}
    for lang in (*LANGUAGES, "all"):
        group = rows if lang == "all" else [r for r in rows if r["language"] == lang]
        results[lang] = {"items": len(group), "original_correct": sum(r["gold"] == r["original"] for r in group),
                         "rotated_correct": sum(r["gold"] == r["rotated"] for r in group),
                         "both_correct": sum(r["original"] == r["rotated"] == r["gold"] for r in group),
                         "same_answer": sum(r["original"] == r["rotated"] for r in group)}
    return {"dataset": "facebook/belebele", "revision": REVISION, "weights_sha256": file_sha256(base / "model.safetensors"),
            "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32", "device": device,
            "threads": torch.get_num_threads(), "batch_size": batch_size, "letter_token_ids": letter_ids,
            "displayed_letters": dict(sorted(displayed.items())), "summary": results, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="Directory with the three pinned JSONL files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scoring", choices=("letter", "answer"), default="letter")
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1:
        parser.error("threads and batch-size must be positive")
    torch.set_num_threads(args.threads)
    method = evaluate if args.scoring == "letter" else likelihood_check
    result = method(args.model, args.data, args.device, args.batch_size)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
