# Evolution of Zero-Shot Text Classification

Zero-shot text classification means classifying text into categories the model has never seen during training. Over the past few years, three main approaches have emerged: NLI-based, embedding-based, and LLM-based. Below is how each developed and what the key papers found.

---

## 1. NLI-Based Approaches

### Yin et al. (2019) - Classification as Entailment

Yin et al. proposed treating classification as a textual entailment problem. Take an NLI model (trained to predict if a hypothesis follows from a premise) and use it for classification by turning labels into hypotheses. For example, to check if a text is about "sports," form the pair: `premise = [text]`, `hypothesis = "This text is about sports."` If the model predicts entailment, assign that label. Different templates handle different tasks like "This text expresses ?" for emotion and "The people there need ?" for situations.

Before this paper, zero-shot classification research was scattered. Definitions were too narrow, classifiers treated labels as numbers without understanding their meaning, and evaluation setups were inconsistent. Yin et al. unified things by defining two settings: *label-partially-unseen* (some labels seen in training, test on all) and *label-fully-unseen* (no task-specific training at all). They benchmarked on Yahoo (topics, 10 classes), UnifyEmotion (9 classes), and Situation Typing (11 classes, multi-label).

Their BERT-based NLI ensemble reached 45.7% accuracy on topics, 25.2% F1 on emotion, and 38.0% F1 on situation in the fully-unseen setting. These numbers were not great, but the idea worked. This approach became the basis for HuggingFace's `zero-shot-classification` pipeline and Facebook's `bart-large-mnli`.

### Laurer et al. (2024) - Universal Classifiers

Laurer et al. addressed a key weakness of the NLI approach. Early classifiers were trained only on NLI datasets (MNLI, SNLI, etc.), which contain generic entailment examples, not task-specific ones. So the models struggled on actual classification tasks.

Their fix was to train on diverse classification datasets reformulated into NLI format, similar to how instruction-tuning improves LLMs. They simplified NLI to binary (entailment vs. not-entailment) and assembled 33 datasets covering 389 classes across sentiment, emotion, intent, toxicity, topics, spam, and more. They combined 5 NLI datasets (~885k pairs) with 28 classification datasets. They aggressively cleaned the data. CleanLab removed ~135k noisy texts (~17%), and they downsampled each class to 500 examples and each dataset to 5,000, reducing over 1M texts to 51,731.

Result: +9.4% higher mean balanced accuracy across 28 tasks compared to NLI-only training, in a leave-one-out zero-shot evaluation. Their DeBERTa-v3 model (~300M parameters) was orders of magnitude cheaper than billion-parameter LLMs. Training took ~10 hours on a single A100 GPU (~50 EUR). Their earlier models had been downloaded over 55 million times on HuggingFace by December 2023.

### Gao, Ghosh & Gimpel (2023) - Label Description Training

Instead of improving the model, Gao et al. improved what the model learns about labels. Their method, LabelDescTraining, finetunes models on tiny datasets that *describe* the labels using related terms, dictionary definitions, and Wikipedia sentences, rather than annotated input texts.

For topic classification, each label got ~6 descriptions: the label itself, three related terms, a dictionary definition, and the first Wikipedia sentence. Total training data was tiny. 24 examples for 4-class topic classification, 84 for 14-class DBPedia.

With RoBERTa-large, this improved average zero-shot accuracy from 58.8% to 77.7% across nine benchmarks (+18.9%). Some individual gains: Yelp-2 went from 70.6% to 94.6%, IMDB from 74.1% to 92.1%, DBPedia from 63.9% to 86.6%. The method also made models much less sensitive to prompt template choice (standard deviations dropped from 9.7-17.0 to 0.8-6.4 across 14 templates).

Their 355M-parameter RoBERTa-large with LabelDescTraining outperformed GPT-3.5 (text-davinci-003) zero-shot on AGNews, DBPedia, Yelp-2, SST-5, and IMDB. A few minutes of label curation beat a model orders of magnitude larger.

---

## 2. Embedding-Based Approaches

These methods place texts and labels in a shared embedding space, turning classification into a nearest-neighbor search.

### WC-SBERT (Chi et al., 2024) - Wikipedia Categories

WC-SBERT trains on label pairs from Wikipedia categories instead of document texts. Starting from the all-mpnet-base-v2 model, they scraped 1.56M unique categories from 6.4M Wikipedia pages and generated training pairs from co-occurring categories using Multiple Negative Ranking (MNR) Loss. They pre-computed embeddings for all 6.45M Wikipedia texts, then used a self-training loop where Wikipedia texts are matched to target labels by similarity.

It achieved state-of-the-art on AG News (0.815) and Yahoo! Answers (0.627), beating both the previous self-training baseline and GPT-3.5. It was 10-40x faster at inference and 5,000-9,700x more efficient in fine-tuning per sample compared to text-based self-training.

The weakness was that it struggled with semantically similar labels. On DBPedia, "Animal" accuracy was only 39.44% (most misclassified as "Plant"), and "Artist" was 26.96% (misclassified as "Album"). When labels are too close in meaning, cosine similarity cannot tell them apart.

### GLiClass (Stepanov et al., 2025) - Uni-Encoder

GLiClass tackles the accuracy-efficiency tradeoff directly. Cross-encoders (NLI-based) give rich joint representations but must process each text-label pair separately, so throughput degrades linearly with the number of labels. Bi-encoders encode texts and labels independently for speed but lose the interaction signal. GLiClass uses a *uni-encoder* instead. It prepends each label with a `<<LABEL>>` token, concatenates all labels with the text, and feeds everything through a single DeBERTa-v3 encoder in one forward pass.

Training has three stages: pre-training on 1.2M examples, mid-training with an adapted PPO for multi-label classification, and post-training with LoRA using synthetic data from GPT-4o.

The large model (439M params) hit F1 of 0.7193 across 14 benchmarks, beating the best cross-encoder baseline (0.6821) by +5.5%. Some big individual gains: +31.8 F1 on Financial Phrasebank, +16.1 on Enron Spam. The efficiency advantage was clear. Scaling from 1 to 128 labels, GLiClass throughput dropped only 7-20%, while cross-encoders dropped 98%. The edge variant (32.7M params) ran at 97.29 examples/sec, which is 16x faster than the DeBERTa-v3-large cross-encoder.

With just 8 labeled examples per class, the smallest model improved by +50% relative F1, making it useful for few-shot scenarios too.

---

## 3. LLM-Based Approaches

Large language models brought a new option: just prompt the model. But research found the picture was more complicated than expected.

### CARP (Sun et al., 2023) - Structured Prompting

Sun et al. found that LLMs underperform fine-tuned models on text classification because the task requires handling complex linguistics like negation, irony, and contrast that generic reasoning approaches are not built for. Their solution, CARP (Clue And Reasoning Prompting), breaks classification into three steps: (1) identify surface clues like keywords and tone, (2) reason over those clues considering deeper phenomena, (3) make the final decision.

With a fine-tuned kNN retriever for demonstrations and weighted voting across multiple runs, CARP with InstructGPT-3 (175B params) achieved SOTA on 4/5 benchmarks: 97.39 on SST-2, 96.40 on AGNews, 98.78 on R8, 96.95 on R52. These numbers surpassed even fully supervised fine-tuned models.

In low-resource settings, the gap was huge. With 16 examples per class, CARP got 90.23% on R8 while fine-tuned RoBERTa got 11.29% (basically random).

The catch is that it needs multiple LLM calls per example, a fine-tuned retriever trained on the full labeled training set, and all experiments used a proprietary API (text-davinci-003). So the impressive numbers come with significant cost and data requirements.

### Gretz et al. (2023) - TTC23 Benchmark

Gretz et al. from IBM Research evaluated what different approaches actually achieve across diverse conditions. They created TTC23, a benchmark of 23 topical classification datasets spanning general, finance, legal, and medical domains, with 4 to 150 classes.

Key findings from comparing NLI, QA, and instruction-tuned approaches (110M to 11B params):
- Off-the-shelf, Flan-T5-XXL (11B) scored best on multi-class (macro-F1 64.79), but the much smaller DeBERTa-Large-NLI (435M) was competitive
- Task-specific fine-tuning had a huge impact. Fine-tuned DeBERTa-Large-NLI gained ~10 F1 points (54.40 to 64.00), matching the 25x-larger off-the-shelf Flan-T5-XL
- No significant correlation (-0.03) between test category similarity to training categories and F1 scores. The models learned the general task, not specific categories
- Even the best model's macro-F1 of 67.32 was "typically still not satisfactory in practice"

### Lepagnol et al. (2024) - Does Size Matter?

Lepagnol et al. asked whether model size actually correlates with zero-shot classification performance. They tested 72 models (77M to 40B params) across 15 datasets.

The answer was mostly no. On 10/15 datasets, there was no significant correlation between size and performance. For IMDB, the correlation was actually *negative*, meaning larger models did worse. A 163M-parameter model (LaMini-GPT-124M) hit 0.734 on AGNews, beating the reported SOTA of 0.625. A 783M-parameter model (LaMini-Flan-T5) hit 0.933 on IMDB and 0.977 on Yelp, far exceeding prior SOTAs.

Other findings: architecture choice (encoder-decoder vs. decoder-only) mattered more than size on 7/15 datasets. Instruction-tuning helped encoder-decoder models (p=0.0086) but not decoder-only ones (p=0.6693). A well-chosen small model can beat a 40B-parameter model while being 50x cheaper to deploy.

---

## 4. Cross-Paradigm Benchmarking: BTZSC (2025)

For years, each paradigm evaluated itself on its own datasets, making comparison hard. The BTZSC benchmark (under review at ICLR 2026) changed this by evaluating 38 models from four families (NLI cross-encoders, embedding models, rerankers, and instruction-tuned LLMs) across 22 datasets under strict zero-shot conditions.

The results were surprising:

- **Rerankers**, which were designed for information retrieval and not classification, came out on top. Qwen3-Reranker-8B hit F1 of 0.72 and accuracy of 0.76, beating the best NLI cross-encoder by +12 F1 points. Even the smaller Qwen3-Reranker-0.6B surpassed all NLI cross-encoders.

- **Embedding models** offered the best accuracy-latency tradeoff. gte-large-en-v1.5 reached F1 of 0.62, beating all NLI cross-encoders despite lacking cross-attention. But scaling showed diminishing returns. The 8B variant barely improved over the 0.6B.

- **NLI cross-encoders**, the dominant approach since 2019, showed signs of plateauing. The best one (deberta-v3-large-nli-triplet) reached F1 of only 0.60, trailing rerankers, LLMs, and even embedding models.

- **LLMs** at 4-12B parameters were competitive. Qwen3-4B reached F1 of 0.65, Mistral-Nemo-Instruct-12B hit 0.67. But sub-billion LLMs failed badly (gemma-3-270m-it at F1 of 0.28), confirming a minimum scale threshold for generative approaches.

Across all paradigms, sentiment was easiest (F1 ~0.88-0.90), topic and intent were in the middle, and emotion remained largely unsolved (best F1 of 0.49).

---

## Where My Work Fits

My project implements BiEncoder, PolyEncoder, and ColBERT architectures for zero-shot classification, trained with in-batch contrastive learning on synthetic data. The training computes a B×B similarity matrix between all text and label embeddings in a batch. The target is an identity matrix, adjusted for duplicates: if two batch items share the same text (with different labels) or the same label (with different texts), those off-diagonal cells are also marked as positives to avoid contradictory signals. Everything else off-diagonal is a negative. This gives B² training signals per batch instead of ~B. WC-SBERT (MNR Loss) and GLiClass (token-level contrastive) also use contrastive learning for zero-shot classification.
