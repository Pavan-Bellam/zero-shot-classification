import torch
import argparse
import yaml

from model import load_model, PolyEncoderModel, ColBERTModel
from utils import load_samples, split_samples, get_all_labels


def compute_scores(model, texts, all_labels, device, batch_size=32):
    """Compute [N_text, N_label] score matrix. Works with both model types."""
    model.eval()

    with torch.no_grad():
        if isinstance(model, ColBERTModel):
            # ColBERT: per-token embeddings + MaxSim scoring
            text_emb, text_mask = model.encode_raw(texts, batch_size)
            label_emb, label_mask = model.encode_raw(all_labels, batch_size)
            scores = torch.sigmoid(model.score_scale * model.maxsim(text_emb, text_mask, label_emb, label_mask))
        elif isinstance(model, PolyEncoderModel):
            # encode labels via mean pooling
            label_embs = []
            for i in range(0, len(all_labels), batch_size):
                emb = model._encode_raw_labels(all_labels[i:i + batch_size])
                label_embs.append(emb)
            label_embeddings = torch.cat(label_embs, dim=0)

            # encode texts via code attention → context vectors
            text_contexts = []
            for i in range(0, len(texts), batch_size):
                ctx = model._encode_raw_texts(texts[i:i + batch_size])
                text_contexts.append(ctx)
            text_contexts = torch.cat(text_contexts, dim=0)

            scores = torch.sigmoid(model.score(text_contexts, label_embeddings))
        else:
            # BiEncoder: encode both sides independently, matmul
            label_embs = []
            for i in range(0, len(all_labels), batch_size):
                emb = model.encode_texts(all_labels[i:i + batch_size])
                label_embs.append(emb)
            label_embeddings = torch.cat(label_embs, dim=0)

            text_embs = []
            for i in range(0, len(texts), batch_size):
                emb = model.encode_texts(texts[i:i + batch_size])
                text_embs.append(emb)
            text_embeddings = torch.cat(text_embs, dim=0)

            scores = torch.sigmoid(text_embeddings @ label_embeddings.T)

    return scores


def evaluate(model, test_samples, all_labels, device, batch_size=32):
    """
    Zero-shot evaluation: for each text, score it against ALL labels,
    pick top-k (where k = number of true labels), check overlap.
    """
    texts = [s["text"] for s in test_samples]
    scores = compute_scores(model, texts, all_labels, device, batch_size)

    top1_correct = 0
    total_correct = 0
    total_labels = 0

    for i, sample in enumerate(test_samples):
        true_labels = set(sample["labels"])
        k = len(true_labels)

        top_indices = scores[i].topk(k).indices.tolist()
        predicted = set(all_labels[j] for j in top_indices)

        correct = len(predicted & true_labels)
        total_correct += correct
        total_labels += k

        # top-1: is the single best label correct?
        best = all_labels[scores[i].argmax().item()]
        if best in true_labels:
            top1_correct += 1

    accuracy_at_k = total_correct / total_labels
    top1_accuracy = top1_correct / len(test_samples)

    return {"top1_accuracy": top1_accuracy, "accuracy_at_k": accuracy_at_k}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF repo id")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    with open("config.yml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path).to(device)

    seed = config.get('seed', 422)
    samples = load_samples(config["data"]["path"])
    train_samples, val_samples, test_samples = split_samples(samples, seed)

    if args.split == "val":
        test_samples = val_samples

    all_labels = get_all_labels(samples)

    print(f"Evaluating on {args.split} split ({len(test_samples)} samples, {len(all_labels)} unique labels)")
    metrics = evaluate(model, test_samples, all_labels, device)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
