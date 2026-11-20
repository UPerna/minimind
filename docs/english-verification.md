# English edition verification

Verified on 20 November 2026 with Python 3.12.14, PyTorch 2.6.0 on CPU, Transformers 4.57.6, Datasets 3.6.0, NumPy 1.26.4, and PyArrow 25.0.1.

The source baseline was [MiniMind master at cc9629e](https://github.com/jingyaogong/minimind/tree/cc9629e5f464ef46ef9411bfac916364e0a6dff1). Inference checks used the [published dense checkpoint at f92512d](https://huggingface.co/jingyaogong/minimind-3/tree/f92512d4cd6142fa9acc0d6022375049a8974bf6).

| Check | Observed result |
| --- | --- |
| Released dense checkpoint | All 63,912,192 parameters loaded into the native model. For the prompt `A test of the native model.`, the maximum absolute difference from the Transformers logits was 0 in float32. The largest difference between cached and full-sequence logits for the final position was about 0.0000204. |
| Command-line inference | The eight built-in English prompts completed on CPU with the published checkpoint and a two-token generation limit. This exercises the interface and generation path. |
| API | A chat request completed in both streaming and non-streaming modes using the released checkpoint. |
| Streamlit | The app discovered the model, rendered the English controls and composer, and retained the thinking toggle after a rerun. |
| Training | A four-record CPU pretraining run and a two-conversation supervised fine-tuning run completed with a one-layer, 64-hidden-size model. Both wrote weights. |
| Language fixtures | All six translation pairs worked in both directions. The weather tool recognized 15 English city names and ten original Chinese spellings. Tokenization and decoding were checked on Chinese, English, and mixed text. |
| Repository content | The two English manual paths have identical content. Both tokenizer files are byte-identical to upstream. The English chat visuals replay the earlier demonstrations, including their original errors. |

## Run the regression checks

Run from the repository root after installing the training dependencies.

```bash
python -m unittest discover -s tests -v
```

The suite uses the bundled tokenizer and locally initialized models. It checks chat formatting, tool inputs, saved training state, loss masking, export equivalence and the scoring of the published multilingual results.

## Training and generation repairs

| Reproduction | Before | After |
| --- | --- | --- |
| Restart pretraining from the first of two accumulated microbatches | 14 of 14 weight tensors differed from uninterrupted training. Maximum absolute difference 0.000643026. | All 14 matched exactly. The regression suite compares optimizer slots as well as weights and covers four save positions, including the final short update. |
| Fine-tune on a long prompt with the entire answer beyond the training window | Zero supervised tokens and a NaN loss. | The same record retained 11 target tokens, with a loss of 8.55460 in the small CPU reproduction. An input with no answer yields zero loss. A window with no targets leaves weights and AdamW state untouched. |
| Force the first generated token to EOS on the published dense Transformers export with a four-token budget | The model generated EOS four times. | After local metadata repair, it generated EOS once and stopped. New dense and MoE test exports also stopped on the first EOS and matched native logits in float32. |
| Install the full requirements on a clean Linux CPU environment with Python 3.12 | Unconstrained PyArrow conflicted with NumPy 1.26.4. The old ujson pin required a source build that failed here. | The committed pins installed with CPU PyTorch 2.6.0. The package checker found all 122 installed packages compatible. |

The supervised trainers average over the number of answer or prediction tokens in each optimizer update. Checkpoints record the latest completed update. On resume, the loader replays any unfinished microbatches with the recorded-position data order and deterministic batch sampling. Use the same data, seed, training settings and process layout for an exact comparison. Older files infer the update position from the saved batch number. Changing process count or batch size maps the position by examples consumed and can change the optimization trajectory.

The suite restarts all eight entry points from batches 1, 2, 3 and 5 of a five-batch CPU epoch and compares final weights and optimizer state with uninterrupted runs. It also checks the PPO critic and learning-rate schedulers. The PPO trainer flushes pending minibatches at each rollout boundary. Other preference trainers use the actual length of the final accumulation window. The three rollout trainers use native PyTorch generation and a local test reward for this check. Full reward-model training, external rollout servers, hardware-specific mixed precision and full-corpus GPU runs require separate validation.

For a local copy of the released dense checkpoint, run the repair once before using plain Transformers generation. Existing explicit EOS lists and other generation options are preserved.

```bash
python scripts/convert_model.py --repair-generation-config /path/to/minimind-3
```

To compare distributed gradients and AdamW states against a single-process reference on a machine with working loopback networking, run the two-rank check.

```bash
torchrun --standalone --nproc-per-node=2 tests/distributed_accumulation.py
```

The [multilingual experiment](model-quality.md) includes the released model's answers, the adapter's answers, scoring rules and next experiments.
