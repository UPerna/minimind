# Multilingual curriculum comparison

Both training arms start from the same released MiniMind-3 dense checkpoint. Each uses 432 chat records, three epochs, batches of four, a 160-token training window, seed 42, learning rate 0.001 and rank-8 LoRA with alpha 16. This matches 324 optimizer updates and the padded training capacity. The two corpora can contain different numbers of answer tokens.

| Arm | Training examples |
| --- | --- |
| Repetition control | The previous pilot’s 90 chat records, repeated to fill 432 slots |
| Expanded curriculum | 144 distinct examples, each in English, Italian and Chinese |

The expanded curriculum covers copying an identifier, locating a city in a record, reading a reported amount, subtraction, debt coverage and metric conversion. Each scenario produces two questions with the same wording. One input changes the correct answer. Some record pairs remove the requested field while retaining its value under another label. The required answer for an absent field is `UNKNOWN`. These are synthetic records with a known answer.

Development contains 72 prompts. The holdout contains 144 prompts, or 24 paired scenarios in three languages. The holdout uses two new wording patterns for each task, new records and operands, and different city names. The existing public 54 prompts serve as a separate check on transfer. General conversation and real financial documents are outside this test.

**Primary comparison.** Compare the expanded curriculum with the matched repetition control on exact answers on the holdout. Also evaluate the original dense checkpoint, the original 90-example adapter and both new models on the same cases. Use greedy decoding, thinking disabled, 32 new tokens and the existing exact-answer grader. Freeze both recipes before observing scores. Run the holdout once per model, with no selection or adjustment based on its answers.

**Secondary outcomes.** Report correct answers by language, task and presence of evidence. Count a pair as correct only when both responses match their expected answers. Report how often the model invents an answer on missing-field cases, and how often it unnecessarily answers `UNKNOWN` on answerable records. Summarize changes on the earlier 54 public prompts separately. Translations and related scenarios share underlying data.

| Generated file | SHA-256 before training |
| --- | --- |
| expanded.jsonl | `976fdd8e5a799188cf7790c96e06d070d6fa5d9c48e897031947bb5c0ad4270c` |
| repeated.jsonl | `4845e4898ffb3a8a289fd8fc0a75f4a8ac66f3b6f1df5d711f5ff192ef830f15` |
| development.json | `78f8b7e141dd4351e837b49bf6d1f580aa85bf52cbe6296113c06ea3bf174537` |
| holdout.json | `7ced28709f479b6d8a342189cead662e335b135f5f0e0bd1ba95bddf1ae83606` |
