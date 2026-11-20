# Which training examples help MiniMind?

Changing the examples raised the exact-answer score from **13/144 to 32/144** at the same training budget. The expanded adapter gave both correct answers on only **1 of 72** pairs in which one input changed. It also tied the original dense model on a separate reading test. Keep this adapter experimental and prioritize correct use of source values in the next training run.

## Training comparison

The previous 90-example adapter returned one of its training answers verbatim on 31 of its 52 errors in the public diagnostic. For this experiment, one control repeated those examples. A second adapter trained on 432 English, Italian and Chinese chat records with answers calculated from known inputs. Both used the same dense checkpoint, rank 8, alpha 16, seed 42, four CPU threads, 160-token training window and 324 optimizer updates. The underlying scenarios, wording and values for the new holdout were fixed before training. [Training protocol](../evals/grounded-v1/protocol.md)

| Model | Development, 72 prompts | Holdout, 144 prompts | Both variants correct, 72 pairs | Earlier public set, 54 prompts |
| --- | ---: | ---: | ---: | ---: |
| Original dense release | 0 | 0 | 0 | 0 |
| Previous 90-example adapter | 0 | 19 | 4 | 2 |
| Repeated examples, matched budget | 2 | 13 | 2 | 1 |
| Expanded curriculum, matched budget | **19** | **32** | **1** | **4** |

On the holdout, the expanded adapter scored 10/48 in English, 8/48 in Italian and 14/48 in Chinese. It gained 32 answers and lost 13 relative to the matched control. Of its 32 correct answers, 23 answered questions with a value and nine correctly identified a missing field. It also answered `UNKNOWN` on 17 of 54 records where the requested field was present. All figures count translated prompts. The 144 prompts represent 24 paired scenarios in three languages. [Item-level predictions](../evals/grounded-v1/predictions.csv) · [Full scores and model hashes](../evals/grounded-v1/results.json)

| Holdout task, 24 prompts each | Repeated examples | Expanded curriculum |
| --- | ---: | ---: |
| Copy an identifier | 5 | 6 |
| Locate a city in a record | 8 | 3 |
| Read a reported amount | 0 | 7 |
| Subtract two amounts | 0 | 2 |
| Calculate debt-service coverage | 0 | 2 |
| Convert metric units | 0 | 12 |

For example, one English record gave the requested identifier as `KMGJ`. In its paired record that field changed to `RBUJ`. The expanded adapter returned the same distractor, `NGWB`, both times. In a subtraction pair, the correct answers were 26 and 35. It answered 26 twice. These examples explain why both variants of a pair must count when evaluating whether the model follows the input.

The CSV stores each response as a JSON string so line breaks and whitespace can be recovered exactly.

## Independent reading checks

Belebele supplies parallel, human-authored reading questions with four choices. On 24 passages in three languages, the expanded adapter selected the correct letter 19/72 times in the original option order and 16/72 after rotating the options. It answered zero items correctly under both orderings and selected only A or D. A constant letter scores 18/72 in either ordering on this slice. [External test protocol](../evals/grounded-v1/reading-protocol.md) · [Belebele dataset](https://huggingface.co/datasets/facebook/belebele)

A follow-up used **24 different passages**. It scored the tokens of each complete candidate answer and tested each question again after removing the passage.

| Model | With passage, 72 items | Question alone, 72 items | Answers gained / lost by adding passage |
| --- | ---: | ---: | ---: |
| Original dense release | 23 | 21 | 13 / 11 |
| Expanded curriculum | 23 | 24 | 14 / 15 |

The full-answer scorer gave both models 23/72 with the passage. The passage improved some individual answers and hurt a similar number. These two small slices cover reading under specific scoring methods. A broader claim about conversational quality or the original model’s pretraining data requires additional evidence. [External predictions and option scores](../evals/grounded-v1/reading)

## Next training priorities

1. **Teach the model to identify the evidence.** Build four versions of a record. Change the requested value, change a distractor, shuffle fields, and remove the requested field. Train the model to return the field name, its source text and its value. Score correct changes, correct invariance and unnecessary refusals separately. Reserve entire sources and layouts for testing. Include naturally written Italian documents and questions alongside English and Chinese.
2. **Use the existing calculator for arithmetic.** Train the model to identify the operation, both operands and their units. For debt coverage, verify that available cash is the numerator and debt service the denominator before calling the calculator. Test wrong roles, missing amounts and zero denominators. Score the complete path from question to answer, including tool selection and operand extraction. [Existing tool interface](../scripts/eval_toolcall.py) · [Program-aided language models](https://proceedings.mlr.press/v202/gao23f.html)
3. **Make broader training earn its cost.** Run a small controlled comparison with licensed, naturally written multilingual passages and source-backed answers. Compare against the released model, a compute-matched control and a small multilingual reference model. Keep a fresh reading sample and a blind conversation review for the final decision. Track Italian gains alongside Chinese retention. A useful candidate should improve paired answers and show a benefit from supplying the relevant passage.

## Run the experiment

Use the [dense MiniMind-3 release at f92512d](https://huggingface.co/jingyaogong/minimind-3/tree/f92512d4cd6142fa9acc0d6022375049a8974bf6). To try the saved adapter in a local Transformers directory, run the following from the fork’s root.

```bash
python scripts/short_answer_pilot.py --base /path/to/minimind-3 --work-dir runs \
  --merge-only evals/grounded-v1/grounded_v1_768.safetensors
```

The 398,496-byte adapter reconstructed the same model file as the training run. Its merged SHA-256 was `03f7af72b45ef1952cbf9ccc7d8ddec5d6ab41470c768d4628c1e2887aa8235a`.

To recreate the two training arms and their questions, run the following commands.

```bash
python scripts/grounded_curriculum.py runs/data
python scripts/compare_curricula.py cases runs/questions.json
python scripts/short_answer_pilot.py --base /path/to/minimind-3 --work-dir runs/control \
  --train-file runs/data/repeated.jsonl --max-seq-len 160 --threads 4
python scripts/short_answer_pilot.py --base /path/to/minimind-3 --work-dir runs/expanded \
  --train-file runs/data/expanded.jsonl --max-seq-len 160 --threads 4
```

Use the printed export paths with `scripts/eval_quality.py --model PATH --cases runs/questions.json --batch-size 8 --threads 2 --output REPORT`. Save each report as `base.json`, `previous.json`, `repeated.json` or `expanded.json` in the same directory. The previous adapter is the [original 90-example pilot](model-quality.md). Run `python scripts/compare_curricula.py summarize REPORT_DIR OUTPUT_DIR` to reproduce the scores and CSV.

For the external checks, download `eng_Latn.jsonl`, `ita_Latn.jsonl` and `zho_Hans.jsonl` from the [pinned dataset release](https://huggingface.co/datasets/facebook/belebele/tree/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data). Run `scripts/eval_belebele_slice.py --model PATH --data SOURCE_DIR --output REPORT` for the letter check. Add `--scoring answer --threads 3` for the full-answer check on the separate passages.
