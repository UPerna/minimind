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

## Run the localization checks

From the repository root, with the training dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The tests use the tokenizer bundled with the repository and run without downloading weights. They check the token IDs, chat template, multilingual round trips, tool inputs, and English tool results.

## Limits observed in training

- Resuming after a completed optimizer update reproduced the uninterrupted run exactly. Restoring a checkpoint saved halfway through a two-step accumulation cycle changed all 14 final weight tensors. The largest difference in the small CPU test was 0.000643.
- When truncation removed every supervised assistant token, the corresponding SFT loss was `NaN`. The neighboring example with five target tokens had finite loss.

Full-corpus training, distributed GPU training, the MoE release, and complete reinforcement-learning runs were outside this verification. The linked corpora and released model weights retain their original languages.
