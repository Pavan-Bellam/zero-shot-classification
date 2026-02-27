# Zero-Shot Text Classification

BiEncoder, PolyEncoder, and ColBERT models for zero-shot text classification, built with PyTorch and Hugging Face Transformers.

## What this is

This project implements three encoder architectures for zero-shot text classification:

- **BiEncoder** - encodes text and labels independently into a shared embedding space using a single BERT encoder, then scores via dot product. Simplest and fastest.
- **PolyEncoder** - uses learnable code vectors that attend over text token outputs to produce richer context representations, then lets each label attend over those context vectors before scoring. More expressive than BiEncoder while staying cheaper than full cross-attention.
- **ColBERT** - keeps per-token embeddings for both text and labels, scores via MaxSim (each label token finds its best-matching text token). Most fine-grained interaction without full cross-encoding.

At inference time, all three models compute similarity scores between a text and any set of labels, including labels never seen during training, making them zero-shot.

## Project structure

```
├── model.py                    # BiEncoderModel, PolyEncoderModel, ColBERTModel, and load_model utility
├── dataset.py                  # Dataset class with custom collator for in-batch contrastive learning
├── utils.py                    # Data loading, sample-level splitting, and pair expansion
├── config.yml                  # Model, training, data generation, and hub config
├── scripts/
│   ├── create_dataset.py       # Synthetic data generation via LLM
│   ├── train.py                # Training script with wandb logging
│   ├── eval.py                 # Zero-shot evaluation via semantic search
│   └── push_to_hub.py          # Push trained checkpoint to HuggingFace Hub
├── data/
│   └── synthetic_data.json     # 1000 generated training samples
└── README.md
```

## Training approach

**Negative sampling via batch structure** - each batch item is a (text, label) pair. `forward()` computes a B×B similarity matrix between all text and label embeddings. The target matrix is an identity matrix adjusted for duplicate texts (same text, different labels) and duplicate labels (different texts, same label). These get marked as positives to avoid contradictory training signals. Everything else off-diagonal is a negative, so negatives come for free from other samples in the batch. This gives B² training signals per batch instead of ~B.

### Model selection

Set `model_type` in `config.yml` to switch between architectures:

```yaml
train:
  model_type: colbert      # bi_encoder, poly_encoder, or colbert
  n_code_vectors: 16        # poly_encoder only
  colbert_dim: 64           # colbert only
```

The `load_model()` utility auto-detects the model type from a saved checkpoint's config and loads the correct class.

## Design decisions

### ColBERT: learnable score scale

ColBERT L2-normalizes its per-token embeddings, so MaxSim returns cosine similarities bounded to [-1, 1]. Feeding these directly as logits to BCE gives `sigmoid([-1, 1]) = [0.27, 0.73]`, which means the model can never produce confident predictions and the loss plateaus. A learnable `score_scale` parameter (initialized to 10.0) multiplies the MaxSim output before the loss, giving sigmoid the full dynamic range. The model learns the optimal scale during training.

### PolyEncoder: code vector initialization

The learnable code vectors that attend over BERT token outputs must be initialized at BERT's scale (`std=0.02`), not the default `torch.randn` scale (`std=1.0`). With `std=1.0`, each code vector has norm ~27.7 vs BERT output norms of ~10-15. The resulting attention scores are so large that softmax becomes nearly one-hot. The context vector just copies a single token instead of being a meaningful weighted average. Small changes in BERT outputs during fine-tuning shift which token gets selected, causing noisy and unstable training. Scaling down to `std=0.02` (norm ~0.55) produces smooth attention distributions and stable optimization.

## Synthetic data generation

`scripts/create_dataset.py` generates training data using the OpenAI API with structured outputs (Pydantic models):

1. Generates a pool of broad, high-level topics (e.g. "Finance", "Sports", "Biology")
2. Randomly samples combinations of 3 topics each as label sets
3. For each combination, asks the LLM to write a sentence that relates to all 3 labels
4. Labels are taken from the topic pool, not generated per-sentence. This guarantees broad, consistent labels
5. Uses async requests with configurable concurrency
6. Produces 1000 samples total, saved to `data/synthetic_data.json`

Each sample looks like:

```json
{
  "text": "A startup sells tiny chips that guide tourists through a city.",
  "labels": [
    "Entrepreneurship",
    "Microelectronics",
    "Tourism"
  ]
}
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requriements.txt
```

To regenerate the synthetic data, add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your-key-here
```

Then run:

```bash
python scripts/create_dataset.py
```

## Training

Training uses [wandb](https://wandb.ai) for experiment tracking. Log in first:

```bash
wandb login
```

Then run:

```bash
python scripts/train.py
```

This reads all settings from `config.yml`, trains for the configured number of steps, validates every 50 steps, and saves checkpoints to `checkpoints/<wandb-run-name>/best/` (lowest val loss) and `checkpoints/<wandb-run-name>/last/` (final step).

## Evaluation

Evaluate a trained checkpoint on the held-out test set:

```bash
python -m scripts.eval --model_path checkpoints/<run-name>/best

# or evaluate on validation split:
python -m scripts.eval --model_path checkpoints/<run-name>/best --split val
```

Evaluation works as actual zero-shot classification: each test text is scored against **all** unique labels in the dataset via semantic similarity, and the top-scoring labels are selected as predictions.

Two metrics are reported:

- **top1_accuracy** - for each text, is the single highest-scoring label one of the ground truth labels? This measures whether the model's best guess is correct.
- **accuracy_at_k** - for each text with k ground truth labels, take the top-k predictions and measure what fraction match. This measures how well the model recovers all relevant labels, not just the best one.

## Push to HuggingFace Hub

After training, push a saved run:

```bash
python scripts/push_to_hub.py --run checkpoints/<run-name> --repo_id your-username/model-name
```

The model is uploaded as a private repository by default. Pass `--public` to make it public.

## Benchmarks

Data: train: 2565 | val: 135 | test: 100

| Architecture | Top-1 Accuracy | Accuracy@k | Converged at | Steps/sec |
|-------------|----------------|------------|-------------|-----------|
| BiEncoder | 0.8300 | 0.6200 | ~450 | 10.0 |
| PolyEncoder | 0.8500 | 0.6967 | ~300 | 7.9 |
| ColBERT | 0.8600 | 0.6967 | ~650 | 9.6 |

Evaluated using `scripts/eval.py` on the test split. Each text is scored against all 300 unique labels via semantic similarity.

## Pros and cons

| | BiEncoder | PolyEncoder | ColBERT |
|---|-----------|-------------|---------|
| **Speed (training)** | Fastest. Just two mean pools + one matmul | Slowest. Two rounds of attention (codes over tokens, then labels over codes) add overhead | Fast. One extra linear projection + matmul, fixed cost regardless of sequence length |
| **Speed (inference)** | Fastest. Label embeddings can be precomputed and reused, scoring is a single matmul | Slower. Each label must attend over text context vectors, so scoring is not a simple matmul | Depends on sequence length. Per-token embeddings mean larger tensors, but MaxSim is parallelizable |
| **Memory** | Minimal. One [D] vector per text/label | Slightly more. [m, D] context vectors per text + [D] per label | Most. [seq_len, dim] per text/label, but dim is smaller (64 vs 768) |

**BiEncoder** is the simplest approach. It encodes text and labels independently and collapses each into a single vector through mean pooling. That pooling step loses information since the whole input is squeezed into one representation, but it makes a solid baseline and is the fastest option.

**PolyEncoder** keeps more information by using m code vectors to attend over text tokens instead of collapsing everything into one. Then each label attends over those m context vectors, so the label actually influences the final text representation. This means the model can weight different parts of the text differently depending on the label it is comparing against.

**ColBERT** goes further by not collapsing at all. It keeps per-token embeddings for both text and labels, and scores through MaxSim where each label token finds its best-matching text token. This gives the most fine-grained interaction since no information is thrown away during encoding. The tradeoff is storing full token-level embeddings instead of single vectors.

At this data scale (~1000 samples, short sentences), speed differences are modest (~8-10 steps/sec for all three). The gap would widen with longer texts or larger batches, where ColBERT's per-token storage and PolyEncoder's attention rounds become more costly.

## Trained models

- [BiEncoder](https://huggingface.co/PavanBellam/zero-cl-bert)
- [PolyEncoder](https://huggingface.co/PavanBellam/zero-cl-poly)
- [ColBERT](https://huggingface.co/PavanBellam/zero-cl-cobert)
