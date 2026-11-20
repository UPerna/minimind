"""Generate aligned English, Italian and Chinese examples and a repetition control."""

import argparse
import json
import random
from decimal import Decimal
from pathlib import Path


LANGUAGES = ("en", "it", "zh")
SPLITS = ("train", "development", "holdout")
PAIRS = {"train": 12, "development": 2, "holdout": 4}
FAMILIES = ("identifier", "location", "reported_amount", "subtraction", "coverage", "conversion")
# Templates 0–2 train the adapters, 3 is for development and 4–5 are held back.
TEMPLATES = {
    "record": [
        ("Read the record [{r}]. Give only the value for {f}. If it is missing, answer UNKNOWN.",
         "Leggi la scheda [{r}]. Scrivi solo il valore di {f}. Se manca, rispondi UNKNOWN.",
         "阅读记录【{r}】。只写出“{f}”的值。若未提供，回答UNKNOWN。"),
        ("Find {f} in this entry [{r}]. Copy its value alone. Use UNKNOWN for a missing field.",
         "Cerca {f} in questa voce [{r}]. Copia il solo valore. Per un campo assente usa UNKNOWN.",
         "请查找以下条目中的“{f}”【{r}】。只抄写其值，缺失时写UNKNOWN。"),
        ("Entry [{r}]. What is listed under {f}? Return the value or UNKNOWN.",
         "Voce [{r}]. Quale valore compare sotto {f}? Rispondi con il valore o con UNKNOWN.",
         "条目【{r}】。“{f}”一栏写了什么？仅回答该值或UNKNOWN。"),
        ("An extract shows [{r}]. Which value belongs to {f}? Write UNKNOWN if absent.",
         "L'estratto riporta [{r}]. Qual è il valore associato a {f}? Se assente scrivi UNKNOWN.",
         "摘录内容为【{r}】。哪一项数值或名称属于“{f}”？没有就写UNKNOWN。"),
        ("Return {f} from this slip [{r}]. Use the value as printed, or UNKNOWN if unavailable.",
         "Da questa distinta [{r}] ricava {f}. Riporta il valore come appare, oppure UNKNOWN se assente.",
         "请从单据【{r}】中读出“{f}”。按原样回答，找不到就写UNKNOWN。"),
        ("Requested field: {f}. Source [{r}]. Reply with the entry alone or UNKNOWN.",
         "Campo richiesto: {f}. Fonte [{r}]. Restituisci la sola voce oppure UNKNOWN.",
         "所需字段为“{f}”，资料为【{r}】。只填写该项，资料未列出则写UNKNOWN。"),
    ],
    "subtraction": [
        ("Calculate {a} minus {b}. Number only.", "Calcola {a} meno {b}. Solo il numero.", "计算{a}减{b}。只写数字。"),
        ("Subtract {b} from {a}. Give only the result.", "Sottrai {b} da {a}. Scrivi solo il risultato.", "从{a}中减去{b}。只写结果。"),
        ("Stock was {a}. We used {b}. How many remain? Just the number.", "Le scorte erano {a}. Ne abbiamo usate {b}. Quante restano? Solo il numero.", "库存原有{a}件，用了{b}件。还剩几件？只写数字。"),
        ("What remains of {a} after removing {b}? Return a number.", "Quanto resta di {a} dopo aver tolto {b}? Rispondi con un numero.", "从{a}中取走{b}后还剩多少？只回答数字。"),
        ("Of {a} tickets, {b} were used. How many are left? Number only.", "Su {a} biglietti ne sono stati usati {b}. Quanti ne restano? Solo il numero.", "共有{a}张票，已用{b}张。还剩多少？只写数字。"),
        ("Give the difference when {b} is taken away from {a}. Number only.", "Indica la differenza ottenuta togliendo {b} da {a}. Solo il numero.", "被减数为{a}，减数为{b}。只写差值。"),
    ],
    "coverage": [
        ("Cash available is {a}, debt service is {b}. Divide cash by debt service. Give the ratio only.", "La cassa disponibile è {a} e il servizio del debito è {b}. Dividi la cassa per il servizio del debito. Solo il rapporto.", "可用现金为{a}，应付债务本息为{b}。将前者除以后者，只写比率。"),
        ("A project pays {b} in debt service with {a} available. What is cash divided by payments? Number only.", "Un progetto paga {b} per il debito e dispone di {a}. Quanto fa la cassa divisa per i pagamenti? Solo il numero.", "项目偿债金额为{b}，可用于偿债的现金为{a}。只写现金除以偿债金额的结果。"),
        ("CFADS = {a}. Debt service = {b}. Report CFADS / debt service as a number.", "CFADS = {a}. Servizio del debito = {b}. Esprimi il CFADS diviso per il servizio del debito con un numero.", "可偿债现金流={a}，债务本息={b}。只回答前者与后者的比值。"),
        ("Divide available cash of {a} by scheduled debt payments of {b}. What is the coverage multiple?", "Dividi la cassa disponibile di {a} per i pagamenti del debito di {b}. Qual è il multiplo di copertura?", "把可用现金{a}除以计划偿债额{b}。覆盖倍数是多少？只写数字。"),
        ("Debt payments total {b}. Cash available for them totals {a}. Calculate cash divided by payments. Number only.", "I pagamenti del debito ammontano a {b}. La cassa disponibile è {a}. Calcola la cassa divisa per i pagamenti. Solo il numero.", "本息支出共{b}，可用现金共{a}。计算现金与本息之比，只写数字。"),
        ("A loan has CFADS of {a} and debt service of {b}. What is DSCR (CFADS divided by debt service)? Reply with only the ratio.", "Un finanziamento ha CFADS pari a {a} e servizio del debito pari a {b}. Qual è il DSCR, cioè CFADS diviso per servizio del debito? Solo il rapporto.", "某项目可偿债现金流为{a}，应付本息为{b}。偿债备付率等于现金流除以本息。只回答比率。"),
    ],
    "conversion": [
        ("Convert {a} {u} to {v}. Number only.", "Converti {a} {u} in {v}. Solo il numero.", "把{a}{u}换算为{v}。只写数字。"),
        ("How many {v} are in {a} {u}? Give only a number.", "Quanti {v} ci sono in {a} {u}? Scrivi solo un numero.", "{a}{u}等于多少{v}？只写数字。"),
        ("Quantity: {a} {u}. Express it in {v}. Number only.", "Quantità: {a} {u}. Esprimila in {v}. Solo il numero.", "已知数量为{a}{u}，用{v}表示。只写数字。"),
        ("Rewrite the measurement {a} {u} in {v}. Answer with a number.", "Riscrivi la misura {a} {u} in {v}. Rispondi con un numero.", "请将测量值{a}{u}改用{v}表示。只回答数字。"),
        ("A label lists {a} {u}. Which number should appear if the label uses {v}?", "Un'etichetta riporta {a} {u}. Che numero va scritto se si usano i {v}?", "标签上标为{a}{u}。若改用{v}，应填写哪个数字？"),
        ("The scale changes from {u} to {v}. What does {a} become? Just the number.", "Si passa da {u} a {v}. A quale numero corrisponde {a}? Scrivi solo il numero.", "单位从{u}改为{v}。原数值{a}应改写成多少？只写数字。"),
    ],
}

FIELDS = {
    "identifier": [("order code", "codice ordine", "订单代码"), ("shipment code", "codice spedizione", "发货代码"),
                   ("batch code", "codice lotto", "批次代码"), ("return code", "codice reso", "退货代码")],
    "location": [("meeting city", "città della riunione", "会议城市"), ("dinner city", "città della cena", "晚宴城市"),
                 ("hotel city", "città dell'albergo", "酒店城市"), ("fair city", "città della fiera", "展会城市")],
    "reported_amount": [("sales", "vendite", "销售额"), ("operating costs", "costi operativi", "运营成本"),
                        ("capital spending", "investimenti", "资本支出"), ("rent", "affitto", "租金")],
}

CITIES = {
    "train": [("Milan", "Milano", "米兰"), ("Paris", "Parigi", "巴黎"), ("Vienna", "Vienna", "维也纳"),
              ("London", "Londra", "伦敦"), ("Athens", "Atene", "雅典"), ("Berlin", "Berlino", "柏林"),
              ("Dublin", "Dublino", "都柏林"), ("Lisbon", "Lisbona", "里斯本")],
    "development": [("Madrid", "Madrid", "马德里"), ("Seoul", "Seul", "首尔"), ("Prague", "Praga", "布拉格"),
                    ("Tokyo", "Tokyo", "东京"), ("Venice", "Venezia", "威尼斯"), ("Lima", "Lima", "利马")],
    "holdout": [("Oslo", "Oslo", "奥斯陆"), ("Lyon", "Lione", "里昂"), ("Riga", "Riga", "里加"),
                ("Bern", "Berna", "伯尔尼"), ("Turin", "Torino", "都灵"), ("Baku", "Baku", "巴库"),
                ("Accra", "Accra", "阿克拉"), ("Perth", "Perth", "珀斯")],
}

UNITS = [(("litres", "litri", "升"), ("millilitres", "millilitri", "毫升")),
         (("kilograms", "chilogrammi", "千克"), ("grams", "grammi", "克"))]


def decimal_text(value, language="en"):
    value = format(Decimal(value), "f")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value.replace(".", ",") if language == "it" else value


def record_pair(split, family, i, rng, template):
    s = SPLITS.index(split)
    if family == "location":
        values = rng.sample(CITIES[split], 4)
    else:
        if family == "identifier":
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
            raw = ["".join(rng.choices(alphabet, k=4)) for _ in range(4)]
        else:
            raw = [str(100 + 3 * n + s) for n in rng.sample(range(300), 4)]
        assert len(set(raw)) == 4
        values = [(v, v, v) for v in raw]
    target, other, third, replacement = rng.sample(FIELDS[family], 4)
    positions = rng.sample(range(3), 3)
    for variant in range(2):
        missing = variant == 1 and i % 2 == 0
        target_value = values[0] if missing else values[variant]
        rows = [(replacement if missing else target, target_value), (other, values[2]), (third, values[3])]
        prompts, answers = {}, {}
        for j, language in enumerate(LANGUAGES):
            record = " | ".join(f"{rows[k][0][j]} = {rows[k][1][j]}" for k in positions)
            prompts[language] = template[j].format(r=record, f=target[j])
            answers[language] = ["UNKNOWN" if missing else values[variant][j]]
        yield {"prompts": prompts, "answers": answers, "case_sensitive": family == "identifier" or missing,
               "missing": missing}


def numerical_pair(split, family, i, rng, template):
    s = SPLITS.index(split)
    b = 4 + (7 * i + 3 * s) % 19
    if family == "subtraction":
        a = 34 + 3 * i + s
        values = [(a, a - b), (a + 9, a + 9 - b)]
    elif family == "coverage":
        b = 10 * (5 + 3 * i + s)
        tenths = rng.sample(range(6, 23), 2)
        values = [(b * t // 10, Decimal(t) / 10) for t in tenths]
    else:
        quantities = [Decimal(10 + 3 * i + s) / 100, Decimal(67 + 3 * i + s) / 100]
        units = UNITS[i % 2]
        reverse = i % 4 in (2, 3)
        if reverse:
            units = (units[1], units[0])
        values = [(a * 1000, a) if reverse else (a, a * 1000) for a in quantities]
    for a, answer in values:
        prompts = {}
        for j, language in enumerate(LANGUAGES):
            args = {"a": decimal_text(a, language), "b": str(b)}
            if family == "conversion":
                args.update(u=units[0][j], v=units[1][j])
            prompts[language] = template[j].format(**args)
        yield {"prompts": prompts, "number": decimal_text(answer)}


def make_cases(split):
    cases = []
    for family in FAMILIES:
        for i in range(PAIRS[split]):
            rng = random.Random(f"grounded-v1/{split}/{family}/{i}")
            forms = TEMPLATES["record" if family in FIELDS else family]
            index = {"train": i % 3, "development": 3, "holdout": 4 + i % 2}[split]
            pair = f"{split}-{family}-{i:02d}"
            generator = record_pair if family in FIELDS else numerical_pair
            for variant, case in enumerate(generator(split, family, i, rng, forms[index])):
                cases.append({"id": f"{pair}-{variant}", "pair": pair, "category": family, **case})
    return cases


def write_datasets(directory):
    try:
        from scripts.short_answer_pilot import training_rows
    except ModuleNotFoundError:
        from short_answer_pilot import training_rows
    directory.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split in SPLITS:
        cases = make_cases(split)
        prompts = [p for c in cases for p in c["prompts"].values()]
        if len(prompts) != len(set(prompts)) or seen.intersection(prompts):
            raise ValueError(f"Duplicate prompts in {split}")
        seen.update(prompts)
        (directory / f"{split}.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if split == "train":
            expanded = []
            for c in cases:
                for language, prompt in c["prompts"].items():
                    answer = decimal_text(c["number"], language) if "number" in c else c["answers"][language][0]
                    expanded.append({"conversations": [{"role": "user", "content": prompt},
                                                       {"role": "assistant", "content": answer}]})
            old = list(training_rows())
            control = [old[i % len(old)] for i in range(len(expanded))]
            for name, rows in (("expanded", expanded), ("repeated", control)):
                random.Random(20261120).shuffle(rows)
                (directory / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    write_datasets(parser.parse_args().directory)
