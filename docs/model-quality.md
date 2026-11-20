# Measuring short-answer quality

The released 64M dense checkpoint returned zero exact answers on 54 prompts covering 18 distinct questions in English, Italian and Chinese. A small LoRA adapter trained on 90 separate examples returned two exact answers. Its clearest effect was on stopping. The base emitted EOS within 32 tokens on 5 prompts, and the adapter did so on 50.

| Matched prompts | Base correct | Adapter correct | Base stopped at EOS | Adapter stopped at EOS |
| --- | ---: | ---: | ---: | ---: |
| English, 18 | 0 | 1 | 3 | 18 |
| Italian, 18 | 0 | 0 | 1 | 18 |
| Chinese, 18 | 0 | 1 | 1 | 14 |
| **All 54** | **0** | **2** | **5** | **50** |

The adapter answered the Italian-capital question in Chinese with “罗马” and gave the formula for carbon dioxide in English as “CO2”. It still answered the Italian question about Portugal’s capital with “Nairobi”. The pilot remains an optional experiment. Its results give little support for switching the default checkpoint.

## What the scores measure

The first set contains six questions, each translated into three languages. The second contains twelve new questions in the same families. Both cover factual recall, subtraction, unit conversion, debt-service coverage and copying an identifier. A correct response must give the requested answer alone. For example, an explanatory sentence can fail even when it contains the right city. Numeric scoring accepts an Italian decimal comma, while identifiers and chemical symbols preserve case. The runner saves the verbatim text, token count and stop reason for every response.

All generations used the same chat template, greedy decoding, thinking disabled, a 32-token limit, float32 and two CPU threads. The JSON reports identify the weights, tokenizer, configuration and question files by SHA-256. The translations share questions, and the two hits provide little evidence about wider language or reasoning ability. These are public development cases. Training on their expected answers would compromise future comparisons.

| Set | Questions | Original model outputs | Adapter outputs |
| --- | --- | --- | --- |
| Initial, 18 prompts | [Question file](../evals/quality_cases.json) | [Base responses](../evals/minimind3-dense-baseline.json) | [Pilot responses](../evals/short-answer-pilot.json) |
| Transfer, 36 prompts | [Question file](../evals/transfer_cases.json) | [Base responses](../evals/minimind3-dense-transfer.json) | [Pilot responses](../evals/short-answer-transfer.json) |

## Reproduce the comparison

Use a local copy of the [original dense checkpoint at f92512d](https://huggingface.co/jingyaogong/minimind-3/tree/f92512d4cd6142fa9acc0d6022375049a8974bf6). Run these commands from the repository root. Substitute your checkpoint path and the export directory printed by the pilot. Allow about 1.2 GB of free disk space per training run.

```bash
python scripts/eval_quality.py --model /path/to/minimind-3 --output base.json
python scripts/eval_quality.py --model /path/to/minimind-3 --cases evals/transfer_cases.json --output base-transfer.json
python scripts/short_answer_pilot.py --base /path/to/minimind-3 --work-dir /tmp/minimind-pilots
python scripts/eval_quality.py --model /path/to/pilot/export --output pilot.json
python scripts/eval_quality.py --model /path/to/pilot/export --cases evals/transfer_cases.json --output pilot-transfer.json
```

The recipe creates 30 training questions in three languages and calls the fork’s LoRA trainer. It uses rank 8, alpha 16, three epochs, batches of four, a 96-token training window and an initial learning rate of 0.001. Two independent CPU training runs produced the same merged model SHA-256, `8fb619437daff233903cc06644f28ad7458193806a53b71f79240b827767ad05`. The calibration data SHA-256 was `952caaede83c042b8bab625a01d31ce730d1c522490e13695eaa747f8b53b99a`.

## Next experiments

1. **Increase answer accuracy.** Compare the base, this pilot, and a candidate trained on a broader, deduplicated English–Italian–Chinese corpus with verified facts, arithmetic answers and source passages. Before training, freeze a separate set of unseen facts, values and document templates. Use questions with longer answers too, and score correctness separately from formatting.
2. **Check what the model retains.** Measure Chinese conversation and tool-call validity alongside English and Italian. Compare candidates on the same items and report paired gains and losses. Keep translated variants together when estimating uncertainty. Evaluate the chosen candidate once on a fresh holdout.
3. **Measure the deployment cost.** Compare the selected candidate on CPU, Apple Silicon and a CUDA device using the same prompt lengths and output limits. Record time to first token, generated tokens per second, peak memory and any changes to the answers after quantization. Run distributed and mixed-precision training checks before recommending a recipe for multiple GPUs.
