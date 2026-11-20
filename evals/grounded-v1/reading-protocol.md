# External reading check

Before scoring the models, choose a fixed slice of the [Belebele test set](https://huggingface.co/datasets/facebook/belebele/tree/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data). It contains independently authored multiple-choice questions with parallel English, Italian and Simplified Chinese versions. This check provides a separate measure of reading transfer. The curriculum uses no Belebele examples.

Use 24 distinct source passages, one question per passage, with six gold answers at each option position. Select aligned questions by sorting their source-link and question-number keys by a fixed SHA-256 ordering. Keep only matching gold positions across languages. The script checks all three source-file hashes.

For each model, score the same 72 translated items twice. The second presentation moves the first option to the end. At the first response token, select the most likely of the four letter tokens. Compare both choices in their original answer coordinates. Report original-order accuracy, rotated-order accuracy, and the count correct under both orderings. A constant letter scores 18/72 in either ordering on this balanced slice.

This test covers a small subset and one scoring method. Option-order sensitivity and a constant-letter baseline help interpret the result. The raw report contains item identifiers, gold labels and predictions. Obtain the passage text from the [official dataset](https://huggingface.co/datasets/facebook/belebele).

## Follow-up after the letter test

The original-order experiment showed a strong preference for particular letters. Before running a second check, fix 24 additional balanced passages using the hash seed `minimind-likelihood-v1`. Exclude every previously selected source link and English passage. Compare the original dense model and the expanded adapter. Score the mean conditional log probability of each complete candidate answer and choose the highest. Exclude prompt, padding and EOS tokens from the mean. Run each question with and without its passage and count which answers change from wrong to right and from right to wrong. This follow-up uses a new sample and cannot be subtracted from the first sample as an estimate of a scoring-method effect.
